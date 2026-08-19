"""
LangFuse 可观测性集成 — LLM 调用链追踪 & 性能监控
- 自动追踪检索、重排序、LLM 生成等所有环节
- 记录耗时、token 用量、输入输出
- 无 LangFuse 配置时自动降级，不影响业务

接入方式 (二选一):
  云端: https://cloud.langfuse.com 注册获取 Public/Secret Key
  自建: docker compose up langfuse 然后配置 LANGFUSE_HOST

环境变量:
  LANGFUSE_PUBLIC_KEY  — 项目公钥
  LANGFUSE_SECRET_KEY  — 项目密钥
  LANGFUSE_HOST        — 自建服务地址 (可选，默认 cloud.langfuse.com)
"""
import logging
import os

logger = logging.getLogger("rag.tracing")

# 全局单例
_langfuse_handler = None
_langfuse_available = False
_checked = False


def _init_langfuse():
    """延迟初始化 LangFuse（避免启动时就连接）"""
    global _langfuse_handler, _langfuse_available, _checked

    if _checked:
        return

    _checked = True
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not (public_key and secret_key):
        logger.info("LangFuse 未配置 (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)，追踪功能已禁用")
        return

    try:
        from langfuse.langchain import CallbackHandler

        _langfuse_handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            # 自动记录所有 LangChain 组件调用的输入/输出/耗时
        )
        _langfuse_available = True
        logger.info("✅ LangFuse 追踪已启用 → %s", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    except ImportError:
        logger.warning("⚠️ langfuse 未安装，追踪不可用。安装: pip install langfuse")
    except Exception as e:
        logger.warning(f"⚠️ LangFuse 初始化失败: {e}")


def get_langfuse_handler():
    """
    获取 LangChain CallbackHandler。
    返回 None 表示追踪不可用（未配置或初始化失败），调用方应优雅降级。
    """
    _init_langfuse()
    return _langfuse_handler if _langfuse_available else None


def get_trace_metadata(session_id: str = None, user_id: str = None, tags: list[str] = None):
    """
    构造 LangChain 追踪元数据。
    调用方可同时传入给 chat() 等入口函数，实现按会话/用户筛选。
    """
    metadata = {}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if user_id:
        metadata["langfuse_user_id"] = user_id
    if tags:
        metadata["langfuse_tags"] = tags
    return metadata


def is_tracing_enabled() -> bool:
    """检查追踪是否已启用"""
    _init_langfuse()
    return _langfuse_available
