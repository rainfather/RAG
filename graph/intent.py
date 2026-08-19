"""
意图路由（Intent Routing）—— 智能客服入口分流

用户问题先进路由分类，再决定走哪条处理链路：

    knowledge  → 知识问答（RAG：rewrite → retrieve → generate）
    handoff    → 转人工（用户明确要求人工服务）
    operation  → 操作类（查订单/退款/改地址——当前返回提示，为工具调用预留）
    chitchat   → 闲聊寒暄（友好回应并引导提问）

分类策略：关键词规则预判（快、稳、免费）→ 未命中时 LLM 分类兜底（灵活）。
规则负责"必须即时可靠"的意图（转人工、操作类），LLM 负责覆盖灵活表达。
"""
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from graph.state import RagState
from graph import nodes as graph_nodes

# ========== 意图定义 ==========
INTENT_KNOWLEDGE = "knowledge"   # 知识问答 → RAG 流程
INTENT_HANDOFF = "handoff"       # 转人工
INTENT_OPERATION = "operation"   # 操作类（暂不支持，转人工；未来接工具调用）
INTENT_CHITCHAT = "chitchat"     # 闲聊寒暄

# 非知识意图的响应文案（用户可见）
HANDOFF_RESPONSE = "好的，正在为您转接人工客服，请稍候。转接后您可以描述具体问题，人工客服会为您处理。"
OPERATION_RESPONSE = "该操作（订单/退款/地址等）当前暂不支持自助办理，已为您记录并转接人工客服处理。"
CHITCHAT_RESPONSE = "您好！我是智能客服助手，可以为您解答产品、使用等知识问题。请问有什么可以帮您？"

# ========== 1. 规则预判（关键词 → 意图） ==========
# 按"必须可靠"优先排序：handoff > operation > chitchat
RULE_KEYWORDS: list[tuple[str, list[str]]] = [
    (INTENT_HANDOFF, ["转人工", "人工客服", "人工服务", "找人工", "转接人工", "真人客服", "客服电话"]),
    (INTENT_OPERATION, ["查订单", "订单", "退款", "退货", "改地址", "修改地址", "收货地址",
                        "取消订单", "物流", "快递", "发票", "重置密码", "开通", "充值", "下单"]),
    (INTENT_CHITCHAT, ["你好", "您好", "嗨", "hi", "hello", "在吗", "谢谢", "感谢", "再见", "拜拜", "辛苦了"]),
]


def rule_based_classify(query: str) -> str | None:
    """关键词规则预判：命中返回意图，未命中返回 None（交给 LLM 兜底）"""
    for intent, keywords in RULE_KEYWORDS:
        for kw in keywords:
            if kw in query:
                return intent
    return None


# ========== 2. LLM 分类兜底 ==========
CLASSIFY_PROMPT = ChatPromptTemplate.from_template("""
你是智能客服的意图分类器。请判断用户问题的意图，只输出 JSON，格式：
{{"intent": "knowledge" 或 "handoff" 或 "operation" 或 "chitchat"}}

意图定义：
- knowledge：询问知识、产品信息、使用说明等（如"什么是RAG""无人机续航多久"）
- handoff：明确要求人工客服、转接、投诉（如"我要找人工"）
- operation：需要办理业务（如"查我的订单""申请退款""修改收货地址"）
- chitchat：问候、寒暄、感谢（如"你好""谢谢"）

只输出 JSON，不要输出其他内容。

用户问题：{query}
""")


def _parse_intent_json(text: str) -> str | None:
    """解析 LLM 输出的 JSON，容错：去掉代码块围栏/多余文字，找不到合法意图返回 None"""
    text = text.strip()
    # 去掉 ```json ... ``` 围栏
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
        intent = data.get("intent")
    except json.JSONDecodeError:
        # 尝试从文本中截取第一个 { ... } 片段
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            data = json.loads(text[start:end + 1])
            intent = data.get("intent")
        except json.JSONDecodeError:
            return None
    return intent if intent in (INTENT_KNOWLEDGE, INTENT_HANDOFF, INTENT_OPERATION, INTENT_CHITCHAT) else None


def llm_classify(query: str) -> str:
    """LLM 分类兜底：失败时保守返回 knowledge（宁可走 RAG 检索，也不误转人工）"""
    try:
        chain = CLASSIFY_PROMPT | graph_nodes.get_llm() | StrOutputParser()
        result = chain.invoke({"query": query}, config=graph_nodes._invoke_config())
        intent = _parse_intent_json(result)
        return intent if intent else INTENT_KNOWLEDGE
    except Exception:
        # LLM 构造/调用失败（如未配置 API Key、网络异常）时保守降级：
        # 走知识问答（RAG 检索不到会拒答，比误判转人工安全）
        return INTENT_KNOWLEDGE


# ========== 3. 路由节点 ==========
def route_intent(state: RagState) -> dict:
    """意图路由节点：规则预判 → LLM 兜底，写入 state["intent"]"""
    query = state["query"]
    intent = rule_based_classify(query)
    if intent is None:
        intent = llm_classify(query)
    return {"intent": intent}


def route_condition(state: RagState) -> str:
    """条件边：knowledge → RAG 流程；其他意图 → 非知识响应节点"""
    intent = state.get("intent", INTENT_KNOWLEDGE)
    if intent == INTENT_KNOWLEDGE:
        return "knowledge"
    return "non_knowledge"


# ========== 4. 非知识响应节点 ==========
INTENT_RESPONSES = {
    INTENT_HANDOFF: HANDOFF_RESPONSE,
    INTENT_OPERATION: OPERATION_RESPONSE,
    INTENT_CHITCHAT: CHITCHAT_RESPONSE,
}


def non_knowledge_response(state: RagState) -> dict:
    """非知识意图的响应：按意图输出对应文案（流式发 token + 写入 answer）"""
    from langgraph.config import get_stream_writer

    intent = state.get("intent", INTENT_CHITCHAT)
    response = INTENT_RESPONSES.get(intent, CHITCHAT_RESPONSE)

    writer = get_stream_writer()
    writer(response)
    return {"answer": response, "sources": []}
