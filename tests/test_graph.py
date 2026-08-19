"""
测试 LangGraph 图：节点级 + 图级（不依赖外部服务与知识库）
"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

import graph.nodes as nodes
from graph.build import build_graph
from conftest import FakeVectorDB, FakeReranker


class TestRewriteQuery:
    """查询改写节点"""

    def test_rewrite_returns_4_queries(self, patch_llm):
        """1 个问题 → 3 变体 + 原问题 = 4 查询"""
        patch_llm(["变体一\n变体二\n变体三"])
        result = nodes.rewrite_query({"query": "什么是RAG？"})
        queries = result["rewritten_queries"]
        assert len(queries) == 4
        assert "什么是RAG？" in queries

    def test_rewrite_dedup(self, patch_llm):
        """变体与原始问题重复时去重"""
        patch_llm(["什么是RAG？\n相同\n相同"])
        result = nodes.rewrite_query({"query": "什么是RAG？"})
        queries = result["rewritten_queries"]
        assert len(set(queries)) == len(queries)  # 无重复
        assert "什么是RAG？" in queries


class TestRetrieveContext:
    """检索节点"""

    def test_no_context_returns_fallback(self, monkeypatch):
        """检索全空 → 拒答文本 + 空 sources + 发 token 事件"""
        monkeypatch.setattr("graph.nodes.get_vector_db", lambda: FakeVectorDB([]))
        monkeypatch.setattr("graph.nodes.bm25_retrieve", lambda q, top_k: [])
        writer = MagicMock()
        monkeypatch.setattr("graph.nodes.get_stream_writer", lambda: writer)

        result = nodes.retrieve_context({"query": "不存在的问题", "rewritten_queries": ["不存在的问题"]})
        assert result["answer"] == nodes.FALLBACK_ANSWER
        assert result["sources"] == []
        writer.assert_called_once()

    def test_retrieve_returns_ranked_sources(self, monkeypatch):
        """有上下文 → 重排后返回 sources（按分数降序）"""
        docs = [
            Document(page_content="无人机续航30分钟", metadata={"source": "无人机.txt", "page": 0}),
            Document(page_content="无人机有智能跟拍模式", metadata={"source": "无人机.txt", "page": 0}),
        ]
        monkeypatch.setattr("graph.nodes.get_vector_db", lambda: FakeVectorDB(docs))
        monkeypatch.setattr("graph.nodes.bm25_retrieve", lambda q, top_k: [])
        monkeypatch.setattr("graph.nodes.get_reranker", lambda: FakeReranker([0.9, 0.2]))

        result = nodes.retrieve_context({"query": "无人机续航", "rewritten_queries": ["无人机续航"]})
        assert len(result["context_docs"]) == 2
        assert result["context_docs"][0].page_content == "无人机续航30分钟"  # 高分在前
        assert result["sources"][0]["index"] == 1
        assert result["sources"][0]["source"] == "无人机.txt"
        assert result["sources"][0]["page"] == 0

    def test_low_score_docs_filtered(self, monkeypatch):
        """分数低于阈值的文档被过滤"""
        docs = [
            Document(page_content="相关内容", metadata={"source": "a.txt", "page": 0}),
            Document(page_content="低分噪音", metadata={"source": "b.txt", "page": 0}),
        ]
        monkeypatch.setattr("graph.nodes.get_vector_db", lambda: FakeVectorDB(docs))
        monkeypatch.setattr("graph.nodes.bm25_retrieve", lambda q, top_k: [])
        monkeypatch.setattr("graph.nodes.get_reranker", lambda: FakeReranker([0.9, 0.005]))

        result = nodes.retrieve_context({"query": "相关内容", "rewritten_queries": ["相关内容"]})
        assert len(result["context_docs"]) == 1
        assert result["context_docs"][0].page_content == "相关内容"


class TestGraph:
    """图级：完整流转 + 条件边"""

    def test_graph_stream_full_flow(self, monkeypatch, patch_llm):
        """改写→检索→生成 全链路，custom 事件产出 token，updates 携带 sources"""
        patch_llm(['{"intent": "knowledge"}', "改写一\n改写二\n改写三", "这是生成的答案"])
        monkeypatch.setattr("graph.nodes.get_vector_db", lambda: FakeVectorDB([
            Document(page_content="RAG是检索增强生成", metadata={"source": "t.txt", "page": 0})]))
        monkeypatch.setattr("graph.nodes.bm25_retrieve", lambda q, top_k: [])
        monkeypatch.setattr("graph.nodes.get_reranker", lambda: FakeReranker([0.9]))

        graph = build_graph()
        chunks = list(graph.stream({"query": "什么是RAG？"}, stream_mode=["custom", "updates"], version="v2"))

        tokens = [c["data"] for c in chunks if c["type"] == "custom"]
        assert "".join(tokens) == "这是生成的答案"

        # updates 中应出现 sources
        sources_seen = []
        for c in chunks:
            if c["type"] == "updates":
                for node_state in c["data"].values():
                    if node_state and node_state.get("sources"):
                        sources_seen = node_state["sources"]
        assert len(sources_seen) == 1
        assert sources_seen[0]["source"] == "t.txt"

    def test_graph_no_context_ends_early(self, monkeypatch, patch_llm):
        """检索为空 → 拒答并结束，不执行生成节点"""
        patch_llm(["改写一\n改写二\n改写三"])
        monkeypatch.setattr("graph.nodes.get_vector_db", lambda: FakeVectorDB([]))
        monkeypatch.setattr("graph.nodes.bm25_retrieve", lambda q, top_k: [])

        graph = build_graph()
        chunks = list(graph.stream({"query": "不存在"}, stream_mode=["custom", "updates"], version="v2"))

        tokens = [c["data"] for c in chunks if c["type"] == "custom"]
        assert "".join(tokens) == nodes.FALLBACK_ANSWER

        # 生成节点不应执行
        for c in chunks:
            if c["type"] == "updates":
                assert "generate_answer" not in c["data"]
