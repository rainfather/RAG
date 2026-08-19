"""
Redis缓存工具类
作用：缓存相同问题的答案，提速+省钱
原理：问题当key，答案当value，存在Redis里，过期自动删除
"""

import redis
import json
import hashlib
import logging
from redis import ConnectionPool
from config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD,
    REDIS_MAX_CONNECTIONS, CACHE_EXPIRE_SECONDS
)

# 使用 logging 代替 print，方便生产环境日志收集
logger = logging.getLogger("rag.redis")

# 全局连接池和客户端单例
_pool = None
_redis_client = None


def get_redis_client():
    """获取Redis连接，全局单例，带连接池和重试"""
    global _pool, _redis_client
    if _redis_client is None:
        _pool = ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            max_connections=REDIS_MAX_CONNECTIONS,
            decode_responses=True,         # 自动把bytes转成字符串
            socket_connect_timeout=3,      # 连接超时3秒
            socket_keepalive=True,         # TCP keepalive，防止空闲断连
            retry_on_timeout=True,         # 超时自动重试
            health_check_interval=30,      # 每30秒检查连接健康状态
        )
        _redis_client = redis.Redis(
            connection_pool=_pool,
            retry_on_error=[ConnectionError, TimeoutError],
        )
    return _redis_client


def ping_redis() -> bool:
    """检查Redis是否连通，用于健康检查"""
    try:
        return get_redis_client().ping()
    except Exception as e:
        logger.warning(f"Redis ping 失败：{e}")
        return False


def _make_cache_key(query: str) -> str:
    """
    生成缓存key
    用SHA256把问题转成固定长度的字符串，作为Redis的key
    为什么不用问题直接当key？因为问题可能很长，浪费空间
    """
    clean_query = query.strip().lower()
    sha_hash = hashlib.sha256(clean_query.encode("utf-8")).hexdigest()
    return f"rag:chat:{sha_hash}"


def get_cached_answer(query: str):
    """
    从缓存里取答案
    找到了返回字典，没找到返回None
    """
    try:
        r = get_redis_client()
        key = _make_cache_key(query)
        cached = r.get(key)
        if cached:
            return json.loads(cached)
        return None
    except Exception as e:
        # Redis挂了也不能影响正常问答，打个日志继续走正常流程
        logger.warning(f"Redis缓存读取失败：{e}")
        return None


def set_cached_answer(query: str, answer_data: dict):
    """
    把答案存进缓存
    answer_data: 要存的数据，比如{"answer": "...", "sources": [...]}
    """
    try:
        r = get_redis_client()
        key = _make_cache_key(query)
        value = json.dumps(answer_data, ensure_ascii=False)
        r.setex(key, CACHE_EXPIRE_SECONDS, value)
    except Exception as e:
        logger.warning(f"Redis缓存写入失败：{e}")


def clear_all_cache():
    """
    清空所有RAG相关的缓存
    知识库更新后调用，避免缓存脏数据
    使用SCAN代替KEYS，避免阻塞Redis
    """
    try:
        r = get_redis_client()
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match="rag:chat:*", count=100)
            if keys:
                r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info(f"已清空 {deleted} 条Redis缓存")
        return deleted
    except Exception as e:
        logger.warning(f"Redis缓存清空失败：{e}")
        return 0


def get_cache_stats():
    """获取缓存统计信息，方便排查问题（使用SCAN，不阻塞）"""
    try:
        r = get_redis_client()
        count = 0
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match="rag:chat:*", count=100)
            count += len(keys)
            if cursor == 0:
                break
        return {
            "cache_count": count,
            "expire_seconds": CACHE_EXPIRE_SECONDS
        }
    except Exception as e:
        logger.warning(f"Redis缓存统计获取失败：{e}")
        return {"error": str(e)}
