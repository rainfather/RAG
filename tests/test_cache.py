"""
测试 Redis 缓存工具类
覆盖：缓存读写、过期、清空、降级、key 生成
"""
import pytest
import time
from cache import (
    get_cached_answer,
    set_cached_answer,
    clear_all_cache,
    get_cache_stats,
    ping_redis,
    _make_cache_key,
)


class TestCacheKey:
    """缓存 key 生成"""

    def test_same_query_same_key(self):
        """相同问题（忽略首尾空格和大小写）应该生成相同的 key"""
        k1 = _make_cache_key("什么是RAG")
        k2 = _make_cache_key("  什么是rag  ")
        assert k1 == k2

    def test_different_query_different_key(self):
        """不同问题应该生成不同的 key"""
        k1 = _make_cache_key("什么是RAG")
        k2 = _make_cache_key("什么是嵌入模型")
        assert k1 != k2

    def test_key_has_prefix(self):
        """key 应该以 rag:chat: 开头"""
        key = _make_cache_key("测试")
        assert key.startswith("rag:chat:")

    def test_key_length_consistent(self):
        """SHA256 生成的 key 长度固定"""
        key = _make_cache_key("任意长度的问题")
        assert len(key) == 64 + len("rag:chat:")  # 64 hex chars + prefix


class TestCacheDegradation:
    """缓存降级：Redis 不可用时不应抛异常"""

    def test_get_returns_none_on_failure(self, mock_redis):
        """Redis 挂了，get_cached_answer 应该返回 None 而不是抛异常"""
        result = get_cached_answer("测试问题")
        assert result is None

    def test_set_does_not_raise_on_failure(self, mock_redis):
        """Redis 挂了，set_cached_answer 应该静默失败"""
        try:
            set_cached_answer("测试", {"answer": "测试答案", "sources": []})
        except Exception as e:
            pytest.fail(f"set_cached_answer 不应抛出异常: {e}")

    def test_clear_returns_zero_on_failure(self, mock_redis):
        """Redis 挂了，clear_all_cache 应该返回 0"""
        count = clear_all_cache()
        assert count == 0

    def test_stats_returns_error_on_failure(self, mock_redis):
        """Redis 挂了，get_cache_stats 应该返回 error 信息"""
        stats = get_cache_stats()
        assert "error" in stats

    def test_ping_returns_false_on_failure(self, mock_redis):
        """Redis 挂了，ping_redis 应该返回 False"""
        assert ping_redis() is False


@pytest.mark.requires_redis
class TestCacheWithRedis:
    """需要真实 Redis 的集成测试（运行前确保 Redis 6+ 启动）"""

    @pytest.fixture(autouse=True)
    def _check_redis(self):
        """每个测试前检查 Redis 是否可用，不可用则跳过"""
        if not ping_redis():
            pytest.skip("Redis 不可用或版本过旧（需要 Redis 6+），跳过集成测试")

    def test_set_and_get(self):
        """写入缓存后能正确读出"""
        query = "集成测试查询"
        data = {"answer": "测试答案", "sources": [{"index": 1, "content": "测试"}]}

        set_cached_answer(query, data)
        result = get_cached_answer(query)

        assert result is not None
        assert result["answer"] == "测试答案"
        assert len(result["sources"]) == 1

    def test_cache_miss(self):
        """未缓存的问题应该返回 None"""
        result = get_cached_answer("一个从未问过的问题_xyz")
        assert result is None

    def test_clear_all(self):
        """清空后所有缓存不可访问"""
        set_cached_answer("清空测试", {"answer": "test", "sources": []})
        deleted = clear_all_cache()
        assert deleted >= 1
        assert get_cached_answer("清空测试") is None

    def test_cache_stats(self):
        """缓存统计正常工作"""
        set_cached_answer("统计测试", {"answer": "stats", "sources": []})
        stats = get_cache_stats()
        assert "cache_count" in stats
        assert stats["cache_count"] >= 1
        assert "expire_seconds" in stats

    def test_ping_ok(self):
        """Redis 正常连接时 ping 返回 True"""
        assert ping_redis() is True
