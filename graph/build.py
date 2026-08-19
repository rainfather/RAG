"""
LangGraph 图构建 — 意图路由 + RAG 问答图

流程：
  route_intent（意图路由）
    ├─ knowledge → rewrite_query → retrieve_context → (有上下文? generate_answer : END)
    └─ handoff / operation / chitchat → non_knowledge_response → END
缓存检查在门面层（graph/rag_chat.py）同步完成——与旧版 rag_chat 行为一致。
"""
from langgraph.graph import StateGraph, END
from graph.state import RagState
from graph.nodes import rewrite_query, retrieve_context, generate_answer
from graph.intent import route_intent, route_condition, non_knowledge_response


def has_context(state: RagState) -> str:
    """条件边：检索无结果（拒答文本已写入 answer）→ 直接结束，否则生成"""
    if state.get("answer"):
        return "end"
    return "generate"


def build_graph():
    """构建并编译意图路由 + RAG 图"""
    g = StateGraph(RagState)
    g.add_node("route_intent", route_intent)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("retrieve_context", retrieve_context)
    g.add_node("generate_answer", generate_answer)
    g.add_node("non_knowledge_response", non_knowledge_response)

    g.set_entry_point("route_intent")
    # 意图路由：knowledge 走 RAG 流程，其余意图走非知识响应
    g.add_conditional_edges(
        "route_intent",
        route_condition,
        {
            "knowledge": "rewrite_query",
            "non_knowledge": "non_knowledge_response",
        },
    )
    g.add_edge("rewrite_query", "retrieve_context")
    g.add_conditional_edges(
        "retrieve_context",
        has_context,
        {"end": END, "generate": "generate_answer"},
    )
    g.add_edge("generate_answer", END)
    g.add_edge("non_knowledge_response", END)

    return g.compile()
