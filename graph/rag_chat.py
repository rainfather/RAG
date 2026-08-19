"""
rag_chat 门面 — 对外接口保持与旧版 rag_core.rag_chat 一致

签名：rag_chat(query, use_cache=True, history=None) -> (answer_or_stream, sources)
- 缓存命中：返回 (字符串答案, sources)
- 未命中：返回 (token 生成器, sources_ref)
  - 生成器逐个产出 token（含拒答场景——拒答文本作为单个 token 发出）
  - sources_ref 是可变列表，生成器消费过程中被填充（retrieve 节点完成后）
- 缓存写入由调用方负责（main.py / app.py 与旧版一致）
"""
from cache import get_cached_answer
from graph.build import build_graph

# 编译后的图（模块级单例）
_agent = None


def get_agent():
    """获取编译后的 LangGraph 图（单例）"""
    global _agent
    if _agent is None:
        _agent = build_graph()
    return _agent


def rag_chat(query, use_cache=True, history=None):
    """
    RAG 总入口（兼容旧版签名）
    - use_cache=True：优先走 Redis 缓存（只有无历史时才有效）
    - history：多轮对话历史 [{"role":"user","content":"..."}, ...]
    - 返回格式统一为 (answer_or_stream, sources_list_of_dicts)
    """
    # 0. 缓存检查（有历史时跳过缓存，因为答案依赖上下文）——与旧版语义一致
    if use_cache and not history:
        cached = get_cached_answer(query)
        if cached:
            return cached["answer"], cached.get("sources", [])

    # 1. 构造初始状态，流式执行图
    state = {"query": query, "history": history or []}
    sources_ref: list[dict] = []

    def stream_answer():
        """逐 token 产出；sources_ref 在 retrieve 节点完成后被填充"""
        # version="v2"：LangGraph 1.x 多模式 stream 返回 {"type","data"} 字典格式
        for chunk in get_agent().stream(state, stream_mode=["custom", "updates"], version="v2"):
            if chunk["type"] == "custom":
                yield chunk["data"]
            elif chunk["type"] == "updates":
                for _node_name, node_state in chunk["data"].items():
                    if node_state and node_state.get("sources"):
                        sources_ref.clear()
                        sources_ref.extend(node_state["sources"])

    return stream_answer(), sources_ref
