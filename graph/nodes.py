"""
LangGraph 节点函数 — RAG 问答图的三个核心节点

节点职责（每个节点独立可测）：
1. rewrite_query     — LLM 查询改写：1 个问题 → 3 个变体 + 原问题 = 4 查询
2. retrieve_context  — 混合检索（FAISS + BM25）→ 去重 → Reranker 精排 → top3
3. generate_answer   — 组装 prompt（history + context）→ LLM 流式生成，
                       节点内 get_stream_writer 逐 token 发送（stream_mode="custom"）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.config import get_stream_writer

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME, LLM_TEMPERATURE, CONVERSATION_MAX_TURNS
from retrieval import get_vector_db, get_bm25, get_reranker, bm25_retrieve
from tracing import get_langfuse_handler
from graph.state import RagState

# ========== LLM 单例 ==========
_llm = None


def get_llm():
    """全局单例 LLM，避免重复创建客户端"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            streaming=True
        )
    return _llm


def _invoke_config():
    """LangFuse 追踪配置：有 handler 则挂 callbacks，无则空 config"""
    handler = get_langfuse_handler()
    return {"callbacks": [handler]} if handler else {}


# ========== 1. 查询改写节点 ==========
REWRITE_PROMPT = ChatPromptTemplate.from_template("""
你是查询改写助手。请将用户的问题改写成3个不同表述的检索查询，用于知识库检索。
要求：
1. 每个查询意思相同，但用词、句式不同
2. 适合做关键词检索，多用专业术语
3. 只输出3行，每行一个查询，不要其他解释

用户问题：{query}
""")


def rewrite_query(state: RagState) -> dict:
    """把用户问题改写成多个不同表述的查询，提升召回率"""
    query = state["query"]
    chain = REWRITE_PROMPT | get_llm() | StrOutputParser()
    result = chain.invoke({"query": query}, config=_invoke_config())
    # 按行分割，过滤空行
    queries = [line.strip() for line in result.strip().split('\n') if line.strip()]
    # 加上原始问题，一共 4 个查询
    queries.append(query)
    return {"rewritten_queries": list(set(queries))}


# ========== 2. 检索节点 ==========
FALLBACK_ANSWER = "知识库中未找到相关内容，无法回答该问题。"


def _docs_to_source_dicts(docs):
    """将 Document 对象列表转为统一的 source dict 格式（与旧 rag_core 一致）"""
    result = []
    for i, doc in enumerate(docs, 1):
        result.append({
            "index": i,
            "content": doc.page_content,
            "source": doc.metadata.get("source", "未知来源"),
            "page": doc.metadata.get("page", "未知页码")
        })
    return result


def retrieve_context(state: RagState) -> dict:
    """多查询改写 + 混合检索 + 重排序精排（原 rag_core.retrieve_context 逻辑）"""
    queries = state["rewritten_queries"]

    all_docs = []
    for q in queries:
        # 1. 向量语义召回
        db = get_vector_db()
        retriever = db.as_retriever(search_kwargs={"k": 10})
        vec_docs = retriever.invoke(q)

        # 2. BM25关键词召回
        bm25_docs = bm25_retrieve(q, top_k=10)

        all_docs.extend(vec_docs)
        all_docs.extend(bm25_docs)

    # 3. 全部合并去重
    seen = set()
    merged_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged_docs.append(doc)

    if not merged_docs:
        # 无上下文：写入拒答文本并作为 token 发出（保持调用方 isinstance 分支兼容）
        writer = get_stream_writer()
        writer(FALLBACK_ANSWER)
        return {"context_docs": [], "sources": [], "answer": FALLBACK_ANSWER}

    # 4. 重排序精排（用原始问题作为 query，与旧 rag_core 行为一致）
    reranker = get_reranker()
    pairs = [[state["query"], doc.page_content] for doc in merged_docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(merged_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in ranked[:3] if score > 0.01]

    # 统一 sources 格式（与旧 _docs_to_source_dicts 一致）
    sources = _docs_to_source_dicts(top_docs)

    return {"context_docs": top_docs, "sources": sources}


# ========== 3. 生成节点 ==========
GENERATE_PROMPT = ChatPromptTemplate.from_template("""
你是严谨的知识问答助手，必须严格根据下方参考资料回答用户问题。
规则：
1. 答案完全来自参考资料，禁止编造任何资料外的信息
2. 若参考资料无相关内容，直接回答"根据现有资料无法回答该问题"
3. 关键信息后标注对应资料编号，例如：[参考资料1]
4. 若当前问题含"刚才""上面""那个"等指代词，请结合对话历史推断用户指代的具体内容

对话历史：
{history}

参考资料：
{context}

用户当前问题：{query}
""")


def _format_history(history: list[dict] | None, max_turns: int = None) -> str:
    """将对话历史列表格式化为 Prompt 文本，只保留最近 N 轮"""
    if not history:
        return "（无历史对话）"
    if max_turns is None:
        max_turns = CONVERSATION_MAX_TURNS
    # 保留最近 N 轮（每轮 = 1 user + 1 assistant = 2 条消息）
    recent = history[-(max_turns * 2):]
    lines = []
    for turn in recent:
        role_label = "用户" if turn["role"] == "user" else "助手"
        # 截断过长的内容，避免 Prompt 过长
        content = turn["content"][:300] + "..." if len(turn["content"]) > 300 else turn["content"]
        lines.append(f"{role_label}：{content}")
    return "\n".join(lines)


def generate_answer(state: RagState) -> dict:
    """基于检索上下文 + 对话历史，调用 LLM 生成答案；节点内逐 token 流式发送"""
    query = state["query"]
    context_docs = state.get("context_docs", [])

    # 拼接上下文，附带来源信息
    context_text = ""
    for i, doc in enumerate(context_docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "未知页码")
        context_text += f"【参考资料{i}】来源：{source} 第{page}页\n内容：{doc.page_content}\n\n"

    history_text = _format_history(state.get("history"))

    chain = GENERATE_PROMPT | get_llm() | StrOutputParser()

    writer = get_stream_writer()
    full_answer = []
    for chunk in chain.stream(
        {"context": context_text, "query": query, "history": history_text},
        config=_invoke_config()
    ):
        full_answer.append(chunk)
        writer(chunk)

    return {"answer": "".join(full_answer)}
