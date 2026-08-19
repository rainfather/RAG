"""
LangGraph 状态定义 — RAG 问答图
"""
from typing import TypedDict, Optional
from langchain_core.documents import Document


class RagState(TypedDict, total=False):
    """RAG 图状态：节点间传递的共享数据"""
    query: str
    history: list[dict]                 # 对话历史 [{"role","content"}, ...]
    intent: str                         # 意图路由结果（knowledge/handoff/operation/chitchat）
    rewritten_queries: list[str]        # 查询改写结果（含原问题）
    context_docs: list[Document]        # 重排序后的上下文文档
    sources: list[dict]                 # 统一 sources 格式 [{"index","content","source","page"}]
    answer: str                         # 最终答案（生成节点累计）
