"""Offline tests for the OpenAI-compatible proxy layer (no network).

Covers: message-view canonicalisation, the 1-hour history router (exact /
prefix / replay decisions + TTL), the replay document, OpenAI chunk
aggregation, tool mapping, model resolution, data-URL and multipart parsing,
and error mapping - all against a stub backend.
"""

import json
import time
import uuid

import pytest

from qwen_studio import exceptions as qe
from qwen_studio.files import FileRef
from qwen_studio.openai_api import (OpenAICompatService, QwenBackend,
                                    SessionRouter, aggregate_chunks,
                                    render_history_document, view_key,
                                    view_message, views_of,
                                    _data_url_to_attachment)
from qwen_studio.openai_server import _Multipart, map_exception


# --------------------------------------------------------------------- stub
class StubBackend:
    """Scripted stand-in for QwenBackend."""

    def __init__(self, turns=None, models=None):
        self.turns = list(turns or [])          # one event-list per stream_turn
        self.stream_calls = []                  # (chat_id, model, prompt, files, tools)
        self.created = []                       # (model, system)
        self.uploads = []                       # (filename, ct)
        self.cleaned = []                       # (chat_ids, project_id)
        self.chat_seq = 0
        self.models = models or ["qwen3.7-plus", "qwen3.5-vl-plus"]

    def model_ids(self):
        return list(self.models)

    def resolve_model(self, requested, has_images=False):
        requested = (requested or "").strip()
        if requested in self.models:
            model = requested
        else:
            model = self.models[0]
        if has_images and "vl" not in model:
            model = next((m for m in self.models if "vl" in m), model)
        return model

    def declare_tools(self, tools):
        bucket = {}
        for t in tools or []:
            f = (t or {}).get("function") or {}
            if f.get("name"):
                bucket[f["name"]] = {"description": f.get("description") or "",
                                     "input_schema": f.get("parameters") or {}}
        return {"openai_tools": bucket} if bucket else {}

    def create_chat(self, model, system_prompt):
        self.created.append((model, system_prompt))
        self.chat_seq += 1
        cid = f"chat-{self.chat_seq}"
        return cid, (f"proj-{self.chat_seq}" if system_prompt.strip() else None)

    def upload(self, data, filename, content_type):
        self.uploads.append((filename, content_type))
        return FileRef(id=f"qwen-{uuid.uuid4().hex[:6]}", url="https://x/y",
                       name=filename, content_type=content_type)

    def cleanup(self, chat_ids, project_id):
        self.cleaned.append((list(chat_ids), project_id))

    def stream_turn(self, chat_id, model, prompt, *, files_entries=None,
                    tools_decl=None, thinking=False):
        self.stream_calls.append((chat_id, model, prompt, files_entries,
                                  tools_decl))
        events = self.turns.pop(0) if self.turns else [
            {"type": "answer", "text": "stub"}]
        yield from events
        yield {"type": "done", "response_id": "resp-1"}


def make_service(backend, **kw):
    svc = OpenAICompatService(backend, ttl=kw.pop("ttl", 3600.0),
                              replay_mode=kw.pop("replay_mode", "both"), **kw)
    return svc


def chat(body, svc):
    kind, payload = svc.handle_chat(body, stream=False)
    assert kind == "json"
    return payload


# -------------------------------------------------------------------- views
def test_view_message_normalises_parts():
    m = {"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
        {"type": "file", "file": {"file_id": "file-1"}},
        {"type": "bogus"}]}
    v = view_message(m)
    assert v["role"] == "user"
    assert v["content"][0] == {"type": "text", "text": "hi"}
    assert v["content"][1] == {"type": "image_url",
                               "url": "data:image/png;base64,AA"}
    assert v["content"][2]["file_id"] == "file-1"


def test_view_message_tool_calls_and_tool_role():
    a = view_message({"role": "assistant", "content": None, "reasoning_content": "x",
                      "tool_calls": [{"id": "call_1", "type": "function",
                                      "function": {"name": "f",
                                                   "arguments": "{\"a\":1}"}}]})
    assert a["tool_calls"][0]["id"] == "call_1"
    assert "reasoning_content" not in a          # decorations are ignored
    t = view_message({"role": "tool", "tool_call_id": "call_1",
                      "content": "result"})
    assert t["tool_call_id"] == "call_1"
    # same semantic message from a client that adds extra fields -> same view
    echoed = view_message({"role": "tool", "tool_call_id": "call_1",
                           "content": "result", "name": "f",
                           "extra_field": 1})
    assert view_key(t) == view_key(echoed)


# ------------------------------------------------------------------- router
def test_router_new_cached_native_replay():
    r = SessionRouter(ttl=3600)
    sys_user = [{"role": "system", "content": "S"},
                {"role": "user", "content": "hi"}]
    v = views_of(sys_user)
    d = r.decide(v)
    assert d.mode == "new"

    from qwen_studio.openai_api import Session, keys_of
    s = Session(id="s1", views=list(v), keys=keys_of(v))
    r.add(s)

    # exact resend -> cached
    assert r.decide(views_of(sys_user)).mode == "cached"
    # stateless retry: incoming history is the prefix just before our reply
    s.append({"role": "assistant", "content": "hello"})
    s.last_assistant = {"id": "x", "choices": [{"message": {
        "role": "assistant", "content": "hello"}, "finish_reason": "stop"}]}
    assert r.decide(views_of(sys_user)).mode == "cached"
    # prefix + user tail -> native
    more = sys_user + [{"role": "assistant", "content": "hello"},
                       {"role": "user", "content": "more"}]
    d2 = r.decide(views_of(more))
    assert d2.mode == "native" and d2.session is s
    assert len(d2.tail) == 1
    # prefix + tool tail -> native at router level (service replays it)
    s.append(view_message({"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "f", "arguments": "{}"}}]}))
    tool_tail = sys_user + [{"role": "assistant", "content": "hello"},
                            {"role": "assistant", "content": None,
                             "tool_calls": [{"id": "call_1", "type": "function",
                                             "function": {"name": "f",
                                                          "arguments": "{}"}}]},
                            {"role": "tool", "tool_call_id": "call_1",
                             "content": "42"}]
    assert r.decide(views_of(tool_tail)).mode == "native"
    # edited history (assistant tail unknown) -> replay
    assert r.decide(views_of(more + [{"role": "assistant", "content": "?"}])
                    ).mode == "replay"
    # unrelated history -> replay
    assert r.decide(views_of([{"role": "user", "content": "a"},
                              {"role": "assistant", "content": "b"},
                              {"role": "user", "content": "c"}])).mode == "replay"


def test_router_ttl_expiry_calls_cleanup():
    expired = []
    r = SessionRouter(ttl=10, on_expire=lambda s: expired.append(s.id))
    from qwen_studio.openai_api import Session, keys_of
    v = views_of([{"role": "user", "content": "hi"}])
    s = Session(id="s1", views=v, keys=keys_of(v))
    s.last_used = time.time() - 100
    r.add(s)
    gone = r.sweep()
    assert [x.id for x in gone] == ["s1"] and expired == ["s1"]
    assert len(r) == 0


# ------------------------------------------------------------------ service
def test_service_new_turn_with_system_prompt_uses_project():
    be = StubBackend(turns=[[{"type": "answer", "text": "hello!"}]])
    svc = make_service(be)
    out = chat({"model": "unknown-model", "stream": False,
                "messages": [{"role": "system", "content": "be terse"},
                             {"role": "user", "content": "hi"}]}, svc)
    assert out["choices"][0]["message"]["content"] == "hello!"
    assert out["choices"][0]["finish_reason"] == "stop"
    # system prompt enforced server-side via a project, model resolved to default
    assert be.created[0] == ("qwen3.7-plus", "be terse")
    assert be.created[0][0] == out["model"]


def test_service_exact_resend_is_cached():
    be = StubBackend(turns=[[{"type": "answer", "text": "A"}]])
    svc = make_service(be)
    body = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
    first = chat(body, svc)
    second = chat(body, svc)
    assert first["id"] == second["id"]                # served from memory
    assert len(be.stream_calls) == 1                  # no new upstream turn
    assert be.created and len(be.created) == 1


def test_service_continuation_routes_to_same_chat():
    be = StubBackend(turns=[[{"type": "answer", "text": "A"}],
                            [{"type": "answer", "text": "B"}]])
    svc = make_service(be)
    m1 = [{"role": "user", "content": "hi"}]
    chat({"model": "m", "messages": m1}, svc)
    m2 = m1 + [{"role": "assistant", "content": "A"},
               {"role": "user", "content": "more"}]
    chat({"model": "m", "messages": m2}, svc)
    assert [c[0] for c in be.stream_calls] == ["chat-1", "chat-1"]
    assert be.created and len(be.created) == 1        # one upstream conversation
    assert be.stream_calls[1][2] == "more"            # only the delta was sent


def test_service_unseen_history_replays_with_history_file():
    be = StubBackend(turns=[[{"type": "answer", "text": "ok"}]])
    svc = make_service(be, replay_mode="both")
    msgs = [{"role": "system", "content": "S"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}]
    out = chat({"model": "m", "messages": msgs}, svc)
    assert out["choices"][0]["message"]["content"] == "ok"
    assert any(fn == "conversation-history.md" for fn, _ in be.uploads)
    # replay prompt contains the history inline (mode=both) and the latest msg
    prompt = be.stream_calls[0][2]
    assert "[2] USER" in prompt and "[3] ASSISTANT" in prompt
    assert "Latest user message:" in prompt
    # the replay chat is a brand new conversation inside the project
    assert be.created[0][1] == "S"


def test_service_tool_roundtrip_uses_replay():
    be = StubBackend(turns=[
        [{"type": "tool_call", "name": "get_weather",
          "arguments": {"city": "Dhaka"}}],
        [{"type": "answer", "text": "sunny 30C"}]])
    svc = make_service(be)
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "w",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}}}}]
    m1 = [{"role": "user", "content": "weather in Dhaka?"}]
    out = chat({"model": "m", "messages": m1, "tools": tools}, svc)
    msg = out["choices"][0]["message"]
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {
        "city": "Dhaka"}
    call_id = msg["tool_calls"][0]["id"]

    # client executes the tool and posts the standard OpenAI tool message
    m2 = m1 + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": "get_weather",
                      "arguments": "{\"city\": \"Dhaka\"}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": "{\"t\": 30}"}]
    out2 = chat({"model": "m", "messages": m2, "tools": tools}, svc)
    assert out2["choices"][0]["message"]["content"] == "sunny 30C"
    # second turn = replay: new chat, history file uploaded, tools re-declared
    assert be.stream_calls[1][0] != be.stream_calls[0][0]
    assert any(fn == "conversation-history.md" for fn, _ in be.uploads)
    assert be.stream_calls[1][4]["openai_tools"]["get_weather"]


def test_service_image_part_uploads_and_picks_vision_model():
    be = StubBackend(turns=[[{"type": "answer", "text": "seen"}]])
    svc = make_service(be)
    png = ("data:image/png;base64,"
           "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8"
           "BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    out = chat({"model": "m", "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": png}}]}]}, svc)
    assert out["choices"][0]["message"]["content"] == "seen"
    _, model, prompt, files, _ = be.stream_calls[0]
    assert "vl" in model                                   # vision routing
    assert files and files[0]["file_class"] == "vision"
    assert "what is this?" in prompt


def test_service_streaming_chunks_shape_and_tool_finish():
    be = StubBackend(turns=[[{"type": "answer", "text": "he"},
                             {"type": "answer", "text": "llo"}]])
    svc = make_service(be)
    kind, gen = svc.handle_chat({"model": "m", "stream": True,
                                 "messages": [{"role": "user",
                                               "content": "hi"}]})
    assert kind == "stream"
    chunks = list(gen)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant",
                                                "content": "he"}
    assert chunks[1]["choices"][0]["delta"]["content"] == "llo"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    # session recorded; exact resend re-serves as cached json
    kind2, payload = svc.handle_chat({"model": "m", "stream": True,
                                      "messages": [{"role": "user",
                                                    "content": "hi"}]})
    assert kind2 == "json" and payload["choices"][0]["message"]["content"] == "hello"


def test_stream_cached_replays_content_chunks():
    be = StubBackend(turns=[[{"type": "answer", "text": "cached text"}]])
    svc = make_service(be)
    full = chat({"model": "m", "messages": [{"role": "user", "content": "q"}]},
                svc)
    chunks = list(svc.stream_cached(full))
    assert any(c["choices"][0]["delta"].get("content") == "cached text"
               for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_upstream_failure_drops_session():
    be = StubBackend(turns=[[{"type": "answer", "text": "ok"}]])
    svc = make_service(be)
    chat({"model": "m", "messages": [{"role": "user", "content": "a"}]}, svc)
    # simulate upstream failure on the next turn, then restore
    orig_stream = be.stream_turn

    def boom(*a, **k):
        raise qe.APIError("upstream boom")

    be.stream_turn = boom
    m2 = [{"role": "user", "content": "a"},
          {"role": "assistant", "content": "ok"},
          {"role": "user", "content": "b"}]
    with pytest.raises(qe.APIError):
        chat({"model": "m", "messages": m2}, svc)
    # session dropped: the retry rebuilds from scratch (replay), not cached
    be.stream_turn = orig_stream
    be.turns = [[{"type": "answer", "text": "rebuilt"}]]
    out = chat({"model": "m", "messages": m2}, svc)
    assert out["choices"][0]["message"]["content"] == "rebuilt"
    assert len(be.created) >= 2                       # a fresh conversation


# ----------------------------------------------------------------- helpers
def test_aggregate_chunks_content_and_tools():
    chunks = [
        {"id": "c1", "object": "chat.completion.chunk", "created": 1,
         "model": "m", "choices": [{"index": 0, "delta": {"role": "assistant"},
                                    "finish_reason": None}]},
        {"id": "c1", "object": "chat.completion.chunk", "created": 1,
         "model": "m", "choices": [{"index": 0, "delta": {"content": "he"},
                                    "finish_reason": None}]},
        {"id": "c1", "object": "chat.completion.chunk", "created": 1,
         "model": "m", "choices": [{"index": 0, "delta": {"tool_calls": [
             {"index": 0, "id": "call_1", "type": "function",
              "function": {"name": "f", "arguments": "{\"a\":"}}]},
             "finish_reason": None}]},
        {"id": "c1", "object": "chat.completion.chunk", "created": 1,
         "model": "m", "choices": [{"index": 0, "delta": {"tool_calls": [
             {"index": 0, "function": {"arguments": "1}"}}]},
             "finish_reason": None}]},
        {"id": "c1", "object": "chat.completion.chunk", "created": 1,
         "model": "m", "choices": [{"index": 0, "delta": {},
                                    "finish_reason": "tool_calls"}]},
    ]
    full = aggregate_chunks(chunks)
    assert full["choices"][0]["message"]["content"] == "he"
    assert full["choices"][0]["finish_reason"] == "tool_calls"
    tc = full["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"] == {"name": "f", "arguments": "{\"a\":1}"}


def test_render_history_document():
    v = views_of([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_9", "type": "function",
             "function": {"name": "calc", "arguments": "{\"x\":2}"}}]},
        {"role": "tool", "tool_call_id": "call_9", "content": "4"},
    ])
    doc = render_history_document(v, system_prompt="be brief")
    assert "be brief" in doc
    assert "[1] USER" in doc and "hello" in doc
    assert "requested tool call `call_9`" in doc and "calc" in doc
    assert "TOOL RESULT (for call call_9)" in doc and "4" in doc


def test_resolve_model_vision_and_default():
    class Dummy:
        def list_model_ids(self):
            return ["qwen3.7-plus", "qwen3.5-vl-plus"]
    be = QwenBackend(Dummy())
    assert be.resolve_model("qwen3.7-plus") == "qwen3.7-plus"
    assert be.resolve_model("gpt-4o") == "qwen3.7-plus"      # default
    assert be.resolve_model("gpt-4o", has_images=True) == "qwen3.5-vl-plus"


def test_data_url_parsing():
    a = _data_url_to_attachment("data:image/png;name=pic;base64,AAEC",
                                "image")
    assert a.content_type == "image/png" and a.filename == "pic.png"
    assert a.data == b"\x00\x01\x02"
    with pytest.raises(qe.BadRequestError):
        _data_url_to_attachment("data:image/png;base64,!!!not-b64!!!",
                                "image")


def test_multipart_parse():
    b = "--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.txt\"\r\n" \
        "Content-Type: text/plain\r\n\r\nhello\r\n--B--\r\n"
    got = _Multipart.parse(b.encode(), "multipart/form-data; boundary=B")
    assert got == ("a.txt", "text/plain", b"hello")


def test_map_exception_status_codes():
    assert map_exception(qe.PunishedError("x"))[0] == 503
    assert map_exception(qe.RateLimitedError("x"))[0] == 429
    assert map_exception(qe.QuotaError("x"))[0] == 429
    assert map_exception(qe.BadRequestError("x"))[0] == 400
    st, obj = map_exception(qe.NotFoundError("gone"))
    assert st == 502 and obj["error"]["type"] == "upstream_error"
    st, obj = map_exception(ValueError("boom"))
    assert st == 500 and obj["error"]["type"] == "internal_error"
