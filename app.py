import streamlit as st
from graph.rag_chat import rag_chat
from cache import set_cached_answer
import os
from config import VECTOR_DB_PATH

st.set_page_config(page_title="简易RAG问答系统", page_icon="📚", layout="wide")
st.title("📚 简易RAG知识问答系统")
st.caption("LangChain + FAISS + BGE + DeepSeek")

# 侧边栏状态提示
with st.sidebar:
    st.header("知识库状态")
    if os.path.exists(VECTOR_DB_PATH):
        st.success("✅ 向量知识库已就绪")
    else:
        st.error("❌ 未检测到知识库，请先运行 data_process.py")
    st.divider()
    st.markdown("""
    **使用步骤**
    1. 将PDF/TXT放入 `data` 文件夹
    2. 运行 `python data_process.py` 建库
    3. 回到页面开始提问
    """)

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 查看参考资料来源"):
                for src in msg["sources"]:
                    st.write(f"**资料{src['index']}**：{src['source']} 第{src['page']}页")
                    st.write(src["content"])
                    st.divider()

# 用户输入与回答生成
if user_query := st.chat_input("请输入你的问题..."):
    # 构建对话历史（不含当前问题）
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("检索知识库中..."):
            answer_stream, sources = rag_chat(user_query, history=history)

            if isinstance(answer_stream, str):
                # 缓存命中或固定回复，直接展示
                response = answer_stream
                st.markdown(response)
            else:
                # 缓存未命中，流式生成并收集完整文本
                response = st.write_stream(answer_stream)
                # 将本次结果写入缓存（无历史时）
                if sources and not history:
                    set_cached_answer(user_query, {
                        "answer": response,
                        "sources": sources
                    })

            # 展示参考来源
            if sources:
                with st.expander("📄 查看参考资料来源"):
                    for src in sources:
                        st.write(f"**资料{src['index']}**：{src['source']} 第{src['page']}页")
                        st.write(src["content"])
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })
