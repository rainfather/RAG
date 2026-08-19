"""
端到端测试：真实检索（已构建知识库）+ FakeLLM 生成
不 mock 中间层——验证 query → 改写 → 混合检索 → 重排 → 生成 → sources 完整链路
"""
import pytest

pytestmark = pytest.mark.requires_kb


def test_e2e_full_pipeline(kb_ready, mock_redis, monkeypatch):
    """真实检索链路 + FakeLLM：完整图执行，产出答案与 sources"""
    from conftest import FakeLLM
    import graph.rag_chat as rag_chat_module

    fake = FakeLLM(responses=[
        "改写一\n改写二\n改写三",  # rewrite_query 节点
        "无人机探索者X100续航时间长达30分钟，支持智能跟拍、轨迹飞行和一键返航。",  # generate_answer 节点
    ])
    monkeypatch.setattr("graph.nodes.get_llm", lambda: fake)

    answer, sources = rag_chat_module.rag_chat("无人机续航时间", use_cache=False)

    text = "".join(answer)
    assert len(text) > 0
    assert len(sources) > 0
    src = sources[0]
    assert src["index"] == 1
    assert "content" in src
    assert "source" in src


def test_e2e_with_history_skips_cache(kb_ready, mock_redis, monkeypatch):
    """带历史时跳过缓存，走完整图"""
    from conftest import FakeLLM
    import graph.rag_chat as rag_chat_module

    fake = FakeLLM(responses=[
        "改写一\n改写二\n改写三",
        "根据资料，RAG是检索增强生成技术。",
    ])
    monkeypatch.setattr("graph.nodes.get_llm", lambda: fake)

    answer, sources = rag_chat_module.rag_chat(
        "那它有什么优势？",
        use_cache=True,
        history=[{"role": "user", "content": "什么是RAG？"},
                 {"role": "assistant", "content": "RAG是检索增强生成技术。"}],
    )
    text = "".join(answer)
    assert len(text) > 0
