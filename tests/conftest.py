"""
pytest 全局配置 & fixtures
"""
import pytest
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class FakeLLM(BaseChatModel):
    """流式/非流式 FakeChatModel：按序弹出 responses（替代真实 LLM，测试不依赖 API）"""

    responses: list[str]

    @property
    def _llm_type(self) -> str:
        return "fake-llm"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = self.responses.pop(0) if self.responses else "默认答案"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        text = self.responses.pop(0) if self.responses else "默认答案"
        for ch in text:
            yield ChatGenerationChunk(message=AIMessageChunk(content=ch))


class FakeRetriever:
    """假检索器：固定返回给定文档"""

    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


class FakeVectorDB:
    """假 FAISS 库：as_retriever 返回 FakeRetriever"""

    def __init__(self, docs):
        self.docs = docs

    def as_retriever(self, **kwargs):
        return FakeRetriever(self.docs)


class FakeReranker:
    """假 reranker：按给定分数序列打分"""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return [self.scores[i % len(self.scores)] for i in range(len(pairs))]


@pytest.fixture(scope="session")
def sample_query():
    """测试用的标准查询"""
    return "什么是RAG？"


@pytest.fixture(scope="session")
def sample_documents():
    """构造测试用的 Document 对象列表"""
    return [
        Document(
            page_content="RAG（Retrieval-Augmented Generation）是一种结合检索与生成的技术。",
            metadata={"source": "test_doc_1.txt", "page": 0}
        ),
        Document(
            page_content="向量数据库用于存储文档的嵌入表示，支持高效的语义搜索。",
            metadata={"source": "test_doc_2.txt", "page": 0}
        ),
        Document(
            page_content="BM25是一种基于词频的稀疏检索算法，适合关键词匹配。",
            metadata={"source": "test_doc_3.txt", "page": 0}
        ),
    ]


@pytest.fixture
def mock_redis(monkeypatch):
    """
    模拟 Redis 不可用的场景，确保降级逻辑正常。
    所有测试默认 mock Redis，需要真实 Redis 的测试用 @pytest.mark.requires_redis
    """
    import cache as cache_module

    def mock_get_client():
        raise ConnectionError("Mock Redis connection failed")

    monkeypatch.setattr(cache_module, "get_redis_client", mock_get_client)
    return cache_module


@pytest.fixture(scope="module")
def kb_ready():
    """知识库未构建时跳过端到端测试（先运行 python data_process.py）"""
    import os
    from config import VECTOR_DB_PATH, BM25_INDEX_PATH

    if not (os.path.exists(VECTOR_DB_PATH) and os.path.exists(BM25_INDEX_PATH)):
        pytest.skip("知识库未构建，跳过端到端测试（先运行 python data_process.py）")


@pytest.fixture
def patch_llm(monkeypatch):
    """把 graph.nodes.get_llm 替换为 FakeLLM；返回工厂函数 _make(responses) -> FakeLLM"""

    def _make(responses):
        fake = FakeLLM(responses=list(responses))
        monkeypatch.setattr("graph.nodes.get_llm", lambda: fake)
        return fake

    return _make
