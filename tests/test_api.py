"""
测试 FastAPI 接口端点
使用 TestClient，不需要启动真实服务
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """FastAPI 测试客户端"""
    return TestClient(app)


class TestHealthCheck:
    """健康检查接口"""

    def test_health_returns_200(self, client):
        """健康检查应返回 200"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "status" in data["data"]
        assert "redis" in data["data"]

    def test_health_has_version(self, client):
        """健康检查应包含版本号"""
        response = client.get("/api/health")
        assert response.json()["data"]["version"] == "1.0.0"


class TestDocumentsEndpoint:
    """文档列表接口"""

    def test_list_documents(self, client):
        """文档列表接口应返回 200"""
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "documents" in data["data"]


class TestChatEndpoint:
    """问答接口"""

    @patch("main.rag_chat")
    @patch("main.get_cached_answer")
    def test_chat_returns_answer(self, mock_cache, mock_rag_chat, client):
        """正常问答流程"""
        mock_cache.return_value = None
        mock_rag_chat.return_value = (
            "RAG是检索增强生成技术...",
            [{"index": 1, "content": "RAG技术...", "source": "test.txt", "page": 1}]
        )
        response = client.post("/api/chat", json={"query": "什么是RAG？"})
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["answer"] == "RAG是检索增强生成技术..."
        assert len(data["data"]["sources"]) == 1

    @patch("main.get_cached_answer")
    def test_chat_cache_hit(self, mock_cache, client):
        """缓存命中直接返回"""
        mock_cache.return_value = {
            "answer": "缓存的答案",
            "sources": [{"index": 1, "content": "...", "source": "cached.txt", "page": 0}]
        }
        response = client.post("/api/chat", json={"query": "缓存测试"})
        assert response.status_code == 200
        data = response.json()
        assert data["from_cache"] is True
        assert data["data"]["answer"] == "缓存的答案"

    def test_chat_empty_query(self, client):
        """空查询应被拒绝"""
        response = client.post("/api/chat", json={"query": ""})
        # FastAPI Pydantic 校验应当返回 422；若穿透则业务层也会报错
        # 无论哪种结果，都应是非正常 status（非 0 code）
        if response.status_code == 200:
            assert response.json()["code"] != 0
        else:
            assert response.status_code == 422


class TestUploadEndpoint:
    """文档上传接口"""

    def test_upload_invalid_extension(self, client):
        """上传不支持的文件格式应返回 400"""
        response = client.post(
            "/api/upload",
            files={"file": ("test.exe", b"fake content", "application/octet-stream")}
        )
        assert response.status_code == 400
        assert "仅支持" in response.json()["detail"]

    def test_upload_txt_file(self, client):
        """上传 TXT 文件应成功"""
        response = client.post(
            "/api/upload",
            files={"file": ("test_doc.txt", b"test document content", "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["filename"] == "test_doc.txt"


class TestCacheEndpoints:
    """缓存管理接口"""

    def test_clear_cache(self, client):
        """清空缓存接口应返回 200"""
        response = client.post("/api/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

    def test_cache_stats(self, client):
        """缓存统计接口应返回 200"""
        response = client.get("/api/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "cache_count" in data["data"] or "error" in data["data"]
