"""
测试 rag_chat 门面（LangGraph 迁移版，保持旧版行为语义）
覆盖：缓存命中返回字符串、无上下文拒答、有历史跳过缓存、纯函数
"""
from unittest.mock import patch

import graph.nodes as nodes
import graph.rag_chat as rag_chat_module
from graph.nodes import _docs_to_source_dicts, _format_history, FALLBACK_ANSWER
from conftest import FakeLLM, FakeVectorDB


class TestDocsToSourceDicts:
    """Document → source dict 格式转换"""

    def test_basic_conversion(self, sample_documents):
        """基本转换：Document 列表转 dict 列表"""
        result = _docs_to_source_dicts(sample_documents)
        assert len(result) == 3
        assert result[0]["index"] == 1
        assert result[1]["index"] == 2
        assert result[2]["index"] == 3

    def test_fields_mapping(self, sample_documents):
        """验证字段映射正确"""
        result = _docs_to_source_dicts(sample_documents)
        first = result[0]
        assert "content" in first
        assert "source" in first
        assert "page" in first
        assert first["source"] == "test_doc_1.txt"
        assert first["page"] == 0

    def test_empty_list(self):
        """空列表返回空列表"""
        result = _docs_to_source_dicts([])
        assert result == []

    def test_missing_metadata_fields(self):
        """缺失 metadata 字段时使用默认值"""
        from langchain_core.documents import Document
        doc = Document(page_content="内容", metadata={})
        result = _docs_to_source_dicts([doc])
        assert result[0]["source"] == "未知来源"
        assert result[0]["page"] == "未知页码"


class TestFormatHistory:
    """对话历史格式化"""

    def test_empty_history(self):
        """空历史返回默认文本"""
        result = _format_history(None)
        assert "无历史" in result
        result = _format_history([])
        assert "无历史" in result

    def test_formats_user_and_assistant(self):
        """正确格式化用户和助手的对话"""
        history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "问题2"},
        ]
        result = _format_history(history)
        assert "用户：问题1" in result
        assert "助手：回答1" in result
        assert "用户：问题2" in result

    def test_truncates_long_content(self):
        """过长内容被截断"""
        long_text = "X" * 350  # > 300 chars, triggers truncation
        history = [{"role": "assistant", "content": long_text}]
        result = _format_history(history, max_turns=1)
        assert "..." in result
        assert len(result) < len(long_text)

    def test_only_recent_turns_kept(self):
        """只保留最近 N 轮"""
        history = []
        for idx in range(1, 7):
            history.append({"role": "user", "content": f"Q{idx}"})
            history.append({"role": "assistant", "content": f"A{idx}"})

        result = _format_history(history, max_turns=2)
        # 只保留最近 2 轮 = 4 条消息
        assert "Q1" not in result  # 旧消息被丢弃
        assert "Q5" in result
        assert "Q6" in result


def _stub_empty_retrieval(monkeypatch):
    """检索组件全空：get_vector_db 返回空库、bm25 返回空"""
    monkeypatch.setattr("graph.nodes.get_vector_db", lambda: FakeVectorDB([]))
    monkeypatch.setattr("graph.nodes.bm25_retrieve", lambda q, top_k: [])
    monkeypatch.setattr("graph.nodes.get_llm", lambda: FakeLLM(responses=["改写一\n改写二\n改写三"]))


class TestRagChatNoCache:
    """rag_chat 门面行为（缓存/拒答/历史）"""

    @patch("graph.rag_chat.get_cached_answer")
    def test_cache_hit_returns_string(self, mock_get_cache):
        """缓存命中时返回字符串而非流"""
        mock_get_cache.return_value = {
            "answer": "缓存中的答案",
            "sources": [{"index": 1, "content": "测试", "source": "test.txt", "page": 1}]
        }
        answer, sources = rag_chat_module.rag_chat("任意问题", use_cache=True)
        assert isinstance(answer, str)
        assert answer == "缓存中的答案"
        assert len(sources) == 1
        assert sources[0]["index"] == 1

    def test_no_context_returns_fallback(self, mock_redis, monkeypatch):
        """知识库无相关内容时返回拒答文本（作为单个 token 发出）"""
        _stub_empty_retrieval(monkeypatch)
        answer, sources = rag_chat_module.rag_chat("无答案的问题", use_cache=False)
        assert not isinstance(answer, str)          # 生成器
        assert "".join(answer) == FALLBACK_ANSWER   # 消费后得到拒答文本
        assert sources == []

    def test_cache_skipped_with_history(self, mock_redis, monkeypatch):
        """有对话历史时跳过缓存"""
        _stub_empty_retrieval(monkeypatch)
        with patch("graph.rag_chat.get_cached_answer") as mock_get_cache:
            answer, sources = rag_chat_module.rag_chat("追问", use_cache=True, history=[
                {"role": "user", "content": "什么是RAG？"},
                {"role": "assistant", "content": "RAG是..."},
            ])
            # 有历史时不调用缓存
            mock_get_cache.assert_not_called()
            # 仍走完整图，检索空 → 拒答
            assert "".join(answer) == FALLBACK_ANSWER

    def test_normal_flow_returns_stream_and_sources(self, mock_redis, monkeypatch):
        """正常流程：生成器产出 LLM 答案，sources 被填充"""
        monkeypatch.setattr("graph.nodes.get_vector_db", lambda: FakeVectorDB([
            __import__("langchain_core.documents", fromlist=["Document"]).Document(
                page_content="无人机续航时间长达30分钟",
                metadata={"source": "无人机.txt", "page": 0})
        ]))
        monkeypatch.setattr("graph.nodes.bm25_retrieve", lambda q, top_k: [])
        monkeypatch.setattr("graph.nodes.get_reranker", lambda: __import__(
            "tests.conftest", fromlist=["FakeReranker"]).FakeReranker([0.9]))
        # 共享实例：路由分类 + rewrite + generate 三个节点消费同一个 responses 队列
        fake = FakeLLM(responses=[
            '{"intent": "knowledge"}',  # route_intent（规则未命中 → LLM 分类）
            "改写一\n改写二\n改写三",     # rewrite_query
            "无人机续航时间长达30分钟。",  # generate_answer
        ])
        monkeypatch.setattr("graph.nodes.get_llm", lambda: fake)
        answer, sources = rag_chat_module.rag_chat("无人机续航多久", use_cache=False)
        text = "".join(answer)
        assert text == "无人机续航时间长达30分钟。"
        assert len(sources) == 1
        assert sources[0]["source"] == "无人机.txt"
