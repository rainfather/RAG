"""
意图路由测试：规则预判 + LLM 分类 + 图级路由流转
"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

import graph.intent as intent
from graph.build import build_graph
from conftest import FakeVectorDB, FakeReranker, FakeLLM


class TestRuleBasedClassify:
    """关键词规则预判"""

    def test_handoff_keywords(self):
        assert intent.rule_based_classify("我要转人工客服") == intent.INTENT_HANDOFF
        assert intent.rule_based_classify("请给我转接人工") == intent.INTENT_HANDOFF
        assert intent.rule_based_classify("有人工客服电话吗") == intent.INTENT_HANDOFF

    def test_operation_keywords(self):
        assert intent.rule_based_classify("帮我查一下我的订单") == intent.INTENT_OPERATION
        assert intent.rule_based_classify("我要申请退款") == intent.INTENT_OPERATION
        assert intent.rule_based_classify("修改收货地址") == intent.INTENT_OPERATION

    def test_chitchat_keywords(self):
        assert intent.rule_based_classify("你好") == intent.INTENT_CHITCHAT
        assert intent.rule_based_classify("谢谢") == intent.INTENT_CHITCHAT

    def test_knowledge_not_matched(self):
        """知识问题不命中任何规则 → None（交给 LLM）"""
        assert intent.rule_based_classify("什么是RAG？") is None
        assert intent.rule_based_classify("无人机续航时间是多少") is None

    def test_priority_handoff_over_operation(self):
        """含多个关键词时按优先级：handoff > operation"""
        assert intent.rule_based_classify("我的订单有问题，转人工客服") == intent.INTENT_HANDOFF


class TestLLMClassify:
    """LLM 分类兜底"""

    def test_parses_valid_json(self, monkeypatch):
        fake = FakeLLM(responses=['{"intent": "knowledge"}'])
        monkeypatch.setattr("graph.nodes.get_llm", lambda: fake)
        assert intent.llm_classify("什么是向量数据库？") == intent.INTENT_KNOWLEDGE

    def test_parses_json_with_code_fence(self, monkeypatch):
        fake = FakeLLM(responses=['```json\n{"intent": "operation"}\n```'])
        monkeypatch.setattr("graph.nodes.get_llm", lambda: fake)
        assert intent.llm_classify("帮我退款") == intent.INTENT_OPERATION

    def test_invalid_json_falls_back_to_knowledge(self, monkeypatch):
        fake = FakeLLM(responses=["抱歉，我无法理解"])
        monkeypatch.setattr("graph.nodes.get_llm", lambda: fake)
        assert intent.llm_classify("随便问问") == intent.INTENT_KNOWLEDGE

    def test_llm_error_falls_back_to_knowledge(self, monkeypatch):
        """LLM 调用异常 → 保守降级 knowledge"""
        def boom(*a, **kw):
            raise RuntimeError("LLM down")
        monkeypatch.setattr("graph.nodes.get_llm", boom)
        assert intent.llm_classify("随便问问") == intent.INTENT_KNOWLEDGE


class TestRouteIntentNode:
    """路由节点：规则优先，规则未命中才调 LLM"""

    def test_rule_hit_skips_llm(self, monkeypatch):
        """规则命中时不调用 LLM"""
        monkeypatch.setattr("graph.nodes.get_llm", boom := MagicMock())
        result = intent.route_intent({"query": "转人工客服"})
        assert result["intent"] == intent.INTENT_HANDOFF
        boom.assert_not_called()

    def test_llm_fallback_when_rule_miss(self, monkeypatch):
        fake = FakeLLM(responses=['{"intent": "knowledge"}'])
        monkeypatch.setattr("graph.nodes.get_llm", lambda: fake)
        result = intent.route_intent({"query": "什么是RAG？"})
        assert result["intent"] == intent.INTENT_KNOWLEDGE


class TestNonKnowledgeResponse:
    """非知识意图的响应节点"""

    def test_handoff_response_text(self, monkeypatch):
        writer = MagicMock()
        monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: writer)
        result = intent.non_knowledge_response({"intent": intent.INTENT_HANDOFF})
        assert result["answer"] == intent.HANDOFF_RESPONSE
        assert result["sources"] == []
        writer.assert_called_once_with(intent.HANDOFF_RESPONSE)


class TestGraphRouting:
    """图级：意图路由条件边流转"""

    def _stub_retrieval(self, monkeypatch, docs=None):
        monkeypatch.setattr("graph.nodes.get_vector_db", lambda: FakeVectorDB(docs or []))
        monkeypatch.setattr("graph.nodes.bm25_retrieve", lambda q, top_k: [])

    def test_knowledge_goes_rag_flow(self, monkeypatch):
        """knowledge 意图 → 完整 RAG 链路（分类 + 改写 + 生成）"""
        fake = FakeLLM(responses=[
            '{"intent": "knowledge"}',      # route_intent（规则未命中 → LLM）
            "改写一\n改写二\n改写三",          # rewrite_query
            "RAG是检索增强生成技术。",         # generate_answer
        ])
        monkeypatch.setattr("graph.nodes.get_llm", lambda: fake)
        self._stub_retrieval(monkeypatch, docs=[
            Document(page_content="RAG是检索增强生成", metadata={"source": "t.txt", "page": 0})])
        monkeypatch.setattr("graph.nodes.get_reranker", lambda: FakeReranker([0.9]))

        chunks = list(build_graph().stream(
            {"query": "什么是RAG？"}, stream_mode=["custom", "updates"], version="v2"))

        tokens = [c["data"] for c in chunks if c["type"] == "custom"]
        assert "".join(tokens) == "RAG是检索增强生成技术。"
        # 经过 RAG 流程 → 有 sources
        sources_seen = []
        for c in chunks:
            if c["type"] == "updates":
                for s in c["data"].values():
                    if s and s.get("sources"):
                        sources_seen = s["sources"]
        assert len(sources_seen) == 1

    def test_handoff_goes_non_knowledge_response(self, monkeypatch):
        """handoff 意图（规则命中，不调 LLM）→ 非知识响应节点"""
        monkeypatch.setattr("graph.nodes.get_llm", boom := MagicMock())
        self._stub_retrieval(monkeypatch)

        chunks = list(build_graph().stream(
            {"query": "我要转人工客服"}, stream_mode=["custom", "updates"], version="v2"))

        tokens = [c["data"] for c in chunks if c["type"] == "custom"]
        assert "".join(tokens) == intent.HANDOFF_RESPONSE
        boom.assert_not_called()  # 规则命中，LLM 完全不被调用
        # 未走 RAG 流程
        for c in chunks:
            if c["type"] == "updates":
                assert "rewrite_query" not in c["data"]

    def test_operation_goes_non_knowledge_response(self, monkeypatch):
        """operation 意图 → 提示文案"""
        self._stub_retrieval(monkeypatch)
        chunks = list(build_graph().stream(
            {"query": "帮我查订单"}, stream_mode=["custom", "updates"], version="v2"))
        tokens = [c["data"] for c in chunks if c["type"] == "custom"]
        assert "".join(tokens) == intent.OPERATION_RESPONSE
