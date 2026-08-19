"""
SSE 流式接口端到端测试
补旧项目缺失的流式覆盖：验证事件序列 token...done 且 done 携带 sources
"""
import json

import pytest
from fastapi.testclient import TestClient

from main import app

pytestmark = pytest.mark.requires_kb


def _parse_sse(body: str) -> list[dict]:
    """解析 SSE 文本为事件列表"""
    events = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    return [json.loads(e) for e in events]


def test_sse_stream_token_then_done(kb_ready, mock_redis, monkeypatch):
    """流式问答：先 token 事件，最后 done 事件且带 sources"""
    from conftest import FakeLLM

    fake = FakeLLM(responses=[
        '{"intent": "knowledge"}',  # route_intent（规则未命中 → LLM 分类）
        "改写一\n改写二\n改写三",     # rewrite_query 节点
        "无人机探索者X100续航时间长达30分钟。",  # generate_answer 节点
    ])
    monkeypatch.setattr("graph.nodes.get_llm", lambda: fake)

    client = TestClient(app)
    with client.stream("POST", "/api/chat", json={"query": "无人机续航时间", "stream": True}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    parsed = _parse_sse(body)
    assert len(parsed) >= 2
    types = [p["type"] for p in parsed]
    assert "token" in types
    assert parsed[-1]["type"] == "done"
    assert len(parsed[-1].get("sources", [])) > 0
    # token 拼接后应等于 FakeLLM 生成的答案
    tokens = "".join(p["content"] for p in parsed if p["type"] == "token")
    assert tokens == "无人机探索者X100续航时间长达30分钟。"
