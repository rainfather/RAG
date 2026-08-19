"""
API Key 鉴权 + 请求频率限制
- verify_api_key: 验证 X-API-Key 请求头
- RateLimiter: 内存滑动窗口限流器
- check_rate_limit: FastAPI 依赖，检查频率限制
"""
import time
import logging
from collections import defaultdict
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from config import API_KEYS, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW

logger = logging.getLogger("rag.auth")

# ========== API Key 验证 ==========
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ========== 滑动窗口限流器 ==========
class RateLimiter:
    """基于内存的滑动窗口限流器，按 API Key 独立计数"""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """检查请求是否被允许，顺便清理过期记录"""
        now = time.time()
        window_start = now - self.window_seconds
        # 清理窗口外的过期记录
        self._requests[key] = [t for t in self._requests[key] if t > window_start]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True

    @property
    def total_tracked_keys(self) -> int:
        """当前追踪的 key 数量（用于监控）"""
        return len(self._requests)


# 全局单例
_limiter = RateLimiter(max_requests=RATE_LIMIT_MAX, window_seconds=RATE_LIMIT_WINDOW)


def verify_api_key(api_key: str = Security(api_key_header)):
    """
    验证 API Key
    - 未配置 API_KEYS（空列表）：放行所有请求
    - 配置了 API_KEYS：仅白名单内的 key 可访问
    """
    if not API_KEYS:
        return True
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="无效或缺失的 API Key，请联系管理员获取")
    return True


def check_rate_limit(api_key: str = Security(api_key_header)):
    """
    请求频率限制
    - 有 API Key 按 key 限流
    - 无 API Key 统一按 "anonymous" 限流
    """
    key = api_key if api_key else "anonymous"
    if not _limiter.is_allowed(key):
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁（{RATE_LIMIT_MAX} 次/{RATE_LIMIT_WINDOW}秒），请稍后再试"
        )
    return True
