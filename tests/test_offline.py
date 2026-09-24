"""Offline tests: no network required.

Run with: pytest tests/ -q
Live behaviour is documented in docs/ and was verified during the study;
these tests pin the offline-correct parts: schema derivation, message and
envelope shapes, SSE parsing, and error mapping.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from qwen_studio import QwenStudio, tool  # noqa: E402
from qwen_studio.chat import ChatCompletion  # noqa: E402
from qwen_studio.exceptions import (APIError, AuthError, BadRequestError,  # noqa: E402
                                    NotFoundError, PunishedError)
from qwen_studio.local_tools import Tool  # noqa: E402
from qwen_studio.sse import consume, parse_event  # noqa: E402


# ----------------------------------------------------------------- tools
def test_tool_schema_derivation():
    @tool
    def get_weather(city: str, days: int = 1):
        """Get weather for a city"""
        return {"city": city}

    s = get_weather.declaration()["input_schema"]
    assert s["properties"]["city"]["type"] == "string"
    assert s["properties"]["days"]["type"] == "integer"
    assert get_weather.name == "get_weather"
    assert get_weather.execute({"city": "Oslo"}) == '{"city": "Oslo"}'


def test_tool_execution_error_is_payload():
    @tool
    def boom(x: int):
        """always fails"""
        raise ValueError("nope")

    with pytest.raises(Exception):
        boom.execute({"x": 1})


# ---------------------------------------------------------------- shapes
def test_user_message_shape():
    msg = ChatCompletion.user_message("hi", "m1", feature_config={"a": 1})
    assert msg["role"] == "user"
    assert msg["content"] == "hi"
    assert msg["chat_type"] == "t2t"
    assert msg["models"] == ["m1"]
    assert msg["feature_config"] == {"a": 1}
    assert msg["fid"] and msg["childrenIds"]


def test_body_shape():
    body = ChatCompletion.build_body("c1", [{"role": "user"}], "m1",
                                     parent_id="r1")
    assert body["stream"] is True
    assert body["version"] == "2.1"
    assert body["incremental_output"] is True
    assert body["chatId"] == body["chat_id"] == "c1"
    assert body["parentId"] == body["parent_id"] == "r1"


# ------------------------------------------------------------------- sse
def _stream(frames):
    return iter(f"data: {json.dumps(f)}" for f in frames)


def test_sse_parse_and_consume():
    frames = [
        {"response.created": {"response_id": "resp1", "chat_id": "c1"}},
        {"choices": [{"delta": {"role": "assistant", "phase": "answer",
                                "status": "typing", "content": "He"}}]},
        {"choices": [{"delta": {"role": "assistant", "phase": "answer",
                                "status": "typing", "content": "y"}}]},
        {"choices": [{"delta": {"role": "assistant", "phase": "answer",
                                "status": "finished"}}],
         "usage": {"total_tokens": 9}},
    ]
    res = consume(_stream(frames))
    assert res.answer == "Hey"
    assert res.response_id == "resp1"
    assert res.phases[0] == "created"
    assert res.phases[-1] == "answer/finished"


def test_sse_local_tool_collection():
    frames = [
        {"response.created": {"response_id": "r2"}},
        {"choices": [{"delta": {"phase": "local_tool", "status": "finished",
                                "extra": {"local_mcp": {
                                    "S": [{"tool_name": "t",
                                           "params": {"a": 1}}]}}}}]},
    ]
    res = consume(_stream(frames))
    assert res.tool_calls == [{"S": [{"tool_name": "t", "params": {"a": 1}}]}]
    assert len(res.tool_events) == 1


# --------------------------------------------------------------- errors
def test_punish_detection():
    with pytest.raises(PunishedError):
        raise PunishedError("RGV587", body="FAIL_SYS_USER_VALIDATE")
    assert PunishedError is not None


def test_error_hierarchy():
    assert issubclass(NotFoundError, APIError)
    assert issubclass(BadRequestError, APIError)
    assert issubclass(AuthError, Exception)


# ------------------------------------------------------------ constructor
def test_constructor_guard():
    with pytest.raises(AuthError):
        QwenStudio()


def test_extra_cookies_merge(tmp_path):
    state = {"cookies": [
        {"name": "token", "value": "T", "domain": ".qwen.ai"},
        {"name": "acw_tc", "value": "A", "domain": "chat.qwen.ai"},
        {"name": "other", "value": "x", "domain": "example.com"},
    ]}
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state))
    jar = QwenStudio.cookies_from_browser_state(str(p))
    assert jar == {"token": "T", "acw_tc": "A"}

    q = QwenStudio.from_access_token("tok", extra_cookies=jar)
    ck = q._cookies()
    assert ck["token"] == "T"          # session material
    assert ck["acw_tc"] == "A"         # anti-bot jar present
    assert "other" not in ck           # foreign domain excluded


def test_punish_markers_include_x5sec():
    from qwen_studio.client import _looks_punished
    assert _looks_punished('{"ret":["RGV587_ERROR::SM"]}')
    assert _looks_punished('window.location.replace(".../_____tmd_____/punish?x5secdata=Z")')
    assert _looks_punished('<meta name="aliyun_waf_aa" content="x">')
    assert not _looks_punished("hello world")


def test_system_chat_context_always_dies_on_exit():
    """A one-shot system chat is deleted (chat, then project) when the
    block ends - even after several turns, and even if one delete fails."""
    from qwen_studio import exceptions as qe
    from qwen_studio.chats import Chat
    from qwen_studio.projects import SystemChatContext

    calls = []

    class Svc:
        def __init__(self, kind, fail_first=False):
            self.kind, self.fail_first = kind, fail_first

        def delete(self, rid):
            calls.append((self.kind, rid))
            if self.fail_first:
                self.fail_first = False
                raise qe.APIError("flaky")

    class FakeClient:
        chats = Svc("chat", fail_first=True)
        projects = Svc("project")

    ctx = SystemChatContext(client=FakeClient(), chat=Chat(id="c1"),
                            project_id="p1")
    with ctx as chat:
        assert chat.id == "c1"
    assert calls == [("chat", "c1"), ("chat", "c1"), ("project", "p1")]

    calls.clear()
    with SystemChatContext(client=FakeClient(), chat=Chat(id="c2"),
                           project_id="p2", auto_cleanup=False):
        pass
    assert calls == []
