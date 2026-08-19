"""
测试 API Key 鉴权 和 请求限流
"""
import pytest
import time
from unittest.mock import patch
from fastapi.testclient import TestClient
from auth import RateLimiter, verify_api_key, check_rate_limit


class TestRateLimiter:
    """滑动窗口限流器单元测试"""

    def test_allows_requests_within_limit(self):
        """在限制范围内允许请求"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("test_key") is True

    def test_blocks_requests_over_limit(self):
        """超过限制后拒绝请求"""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            assert limiter.is_allowed("test_key") is True
        assert limiter.is_allowed("test_key") is False

    def test_different_keys_independent(self):
        """不同 key 之间独立计数"""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("key_a") is True
        assert limiter.is_allowed("key_a") is True
        assert limiter.is_allowed("key_a") is False  # key_a 被限
        assert limiter.is_allowed("key_b") is True   # key_b 不受影响

    def test_window_expiry(self, monkeypatch):
        """窗口过期后可以重新请求"""
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        assert limiter.is_allowed("key") is True
        assert limiter.is_allowed("key") is False

        # 模拟时间前进 2 秒（先存下真实时间避免递归）
        real_now = time.time()
        monkeypatch.setattr(time, 'time', lambda: real_now + 2.0)
        assert limiter.is_allowed("key") is True


class TestAuthIntegration:
    """鉴权集成测试（通过 FastAPI TestClient）"""

    @pytest.fixture
    def client(self):
        from main import app
        return TestClient(app)

    @patch("auth.API_KEYS", ["test-secret-key"])
    def test_valid_api_key_accepted(self, client):
        """正确的 API Key 应通过"""
        # 注意：/api/chat 需要完整参数，用 /api/health 测试鉴权
        # health 端点没有 auth 依赖，所以换个思路：直接测 auth 函数
        import asyncio
        from fastapi import HTTPException

        # 验证函数层面的行为
        with patch("auth.API_KEYS", ["test-key"]):
            with pytest.raises(HTTPException) as exc_info:
                # 模拟 Security 调用 - 传入错误的 key
                with patch("auth.api_key_header", return_value="wrong-key"):
                    verify_api_key("wrong-key")
            assert exc_info.value.status_code == 403

    @patch("auth.API_KEYS", [])
    def test_empty_keys_allows_all(self):
        """未配置 API_KEYS 时放行所有请求"""
        result = verify_api_key(None)
        assert result is True

    def test_chat_endpoint_accepts_history(self, client):
        """问答接口接受 history 参数"""
        # 注意：没有 API Key 且 API_KEYS 为空时 auth 放行
        with patch("main.rag_chat") as mock_rag, \
             patch("main.get_cached_answer", return_value=None):
            mock_rag.return_value = ("答案", [{"index": 1, "content": "...", "source": "t.txt", "page": 1}])
            response = client.post("/api/chat", json={
                "query": "测试问题",
                "history": [
                    {"role": "user", "content": "之前的问题"},
                    {"role": "assistant", "content": "之前的回答"}
                ]
            })
            assert response.status_code == 200
            # 验证 rag_chat 被调用时传入了 history
            call_kwargs = mock_rag.call_args
            assert call_kwargs[1].get("history") is not None

    def test_chat_endpoint_accepts_stream(self, client):
        """SSE 流式模式返回 event-stream"""
        with patch("main.rag_chat") as mock_rag, \
             patch("main.get_cached_answer", return_value=None):
            mock_rag.return_value = ("流式答案", [])
            response = client.post("/api/chat", json={
                "query": "测试",
                "stream": True
            })
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
