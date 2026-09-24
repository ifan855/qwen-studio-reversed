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
        self.proj_seq = 0
        self.chats = {}                         # live chat -> project
        self.projects = {}                      # live project -> instruction
        self.projects_created = []
        self.deleted_chats = []
        self.deleted_projects = []
        self.instruction_updates = []
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

    # lifecycle primitives (mirror QwenBackend)
    def create_project(self, instruction):
        self.proj_seq += 1
        pid = f"proj-{self.proj_seq}"
        self.projects[pid] = instruction
        self.projects_created.append(pid)
        return pid

    def open_chat(self, model, project_id=None):
        self.chat_seq += 1
        cid = f"chat-{self.chat_seq}"
        self.chats[cid] = project_id
        self.created.append((model, self.projects.get(project_id, "")
                             if project_id else ""))
        return cid

    def set_instruction(self, project_id, instruction):
        self.instruction_updates.append((project_id, instruction))
        self.projects[project_id] = instruction

    def delete_chat(self, chat_id):
        self.deleted_chats.append(chat_id)
        self.chats.pop(chat_id, None)

    def delete_project(self, project_id):
        self.deleted_projects.append(project_id)
        self.projects.pop(project_id, None)

    def upload(self, data, filename, content_type):
        self.uploads.append((filename, content_type))
        return FileRef(id=f"qwen-{uuid.uuid4().hex[:6]}", url="https://x/y",
                       name=filename, content_type=content_type)

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
def _committed(sid, msgs, reply="hello"):
    """A stored, completed session for ``msgs`` + an assistant ``reply``."""
    from qwen_studio.openai_api import Session
    v = views_of(msgs + [{"role": "assistant", "content": reply}])
    s = Session(id=sid, views=v, chat_id=f"chat-{sid}")
    s.last_assistant = {"id": sid, "choices": [{"message": {
        "role": "assistant", "content": reply}, "finish_reason": "stop"}]}
    return s


def test_router_new_cached_native_fork_replay():
    r = SessionRouter(ttl=3600)
    sys_user = [{"role": "system", "content": "S"},
                {"role": "user", "content": "hi"}]
    assert r.decide(views_of(sys_user)).mode == "new"

    s = _committed("s1", sys_user, "hello")
    r.add(s)
    # stateless retry: the history right before our stored reply -> cached
    assert r.decide(views_of(sys_user)).mode == "cached"
    # exact resend including our reply -> cached
    both = sys_user + [{"role": "assistant", "content": "hello"}]
    assert r.decide(views_of(both)).mode == "cached"
    # prefix + user tail -> native in the same session
    more = both + [{"role": "user", "content": "more"}]
    d = r.decide(views_of(more))
    assert d.mode == "native" and d.session is s and len(d.tail) == 1
    # prefix + tool tail -> native as well (tool results stay in the chat)
    d = r.decide(views_of(both + [{"role": "tool", "tool_call_id": "x",
                                   "content": "42"}]))
    assert d.mode == "native" and d.tail[0]["role"] == "tool"
    # client appended its own assistant message + user -> still native
    d = r.decide(views_of(more + [{"role": "assistant", "content": "?"},
                                  {"role": "user", "content": "!"}]))
    assert d.mode == "native" and len(d.tail) == 3
    # edited reply (diverges after the first user turn) -> fork
    edited = sys_user + [{"role": "assistant", "content": "EDITED"},
                         {"role": "user", "content": "x"}]
    d = r.decide(views_of(edited))
    assert d.mode == "fork" and d.session is s
    # same dialog, different system prompt -> native, flagged
    d = r.decide(views_of([{"role": "system", "content": "S2"}] + more[1:]))
    assert d.mode == "native" and d.system_changed
    # unrelated history -> replay
    assert r.decide(views_of([{"role": "user", "content": "a"},
                              {"role": "assistant", "content": "b"},
                              {"role": "user", "content": "c"}])).mode == "replay"


def test_router_matching_is_lenient_about_client_echo_formatting():
    """Clients echo our replies with cosmetic differences; routing must see
    through them instead of falling back to a replay."""
    r = SessionRouter()
    from qwen_studio.openai_api import Session
    stored = views_of([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_ours", "type": "function",
             "function": {"name": "w", "arguments": "{\"city\": \"Dhaka\", \"u\": 1}"}}]},
        {"role": "tool", "tool_call_id": "call_ours", "content": "30C"},
        {"role": "assistant", "content": "It is 30C.\n"}])
    s = Session(id="s", views=stored, chat_id="c")
    s.last_assistant = {"id": "x"}
    r.add(s)
    echoed = [
        {"role": "developer", "content": [{"type": "text", "text": "S"}]},
        {"role": "user", "content": [{"type": "text", "text": "weather?  "}]},
        {"role": "assistant", "content": "", "reasoning_content": "hmm",
         "tool_calls": [{"id": "call_RENAMED", "type": "function",
                         "function": {"name": "w",
                                      "arguments": "{\"u\":1,\"city\":\"Dhaka\"}"}}]},
        {"role": "tool", "tool_call_id": "call_RENAMED", "content": "30C",
         "name": "w"},
        {"role": "assistant", "content": [{"type": "text",
                                           "text": "It is 30C."}]},
        {"role": "user", "content": "thanks\r\n"}]
    d = r.decide(views_of(echoed))
    assert d.mode == "native" and d.session is s and len(d.tail) == 1


def test_router_prefers_longest_stored_history_not_most_recent():
    """The old router stopped at the first (most recent) session sharing a
    prefix; a longer, exact prefix held by an older session was missed."""
    r = SessionRouter()
    base = [{"role": "user", "content": "q1"}]
    full = _committed("old", base + [{"role": "assistant", "content": "a1"},
                                     {"role": "user", "content": "q2"}], "a2")
    r.add(full)
    partial = _committed("new", base, "a1")          # added later
    r.add(partial)
    msgs = base + [{"role": "assistant", "content": "a1"},
                   {"role": "user", "content": "q2"},
                   {"role": "assistant", "content": "a2"},
                   {"role": "user", "content": "q3"}]
    d = r.decide(views_of(msgs))
    assert d.mode == "native" and d.session is full and len(d.tail) == 1


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


def test_service_tool_roundtrip_stays_in_same_chat():
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
    # the result went to the SAME chat, no new conversation, no replay doc
    assert [c[0] for c in be.stream_calls] == ["chat-1", "chat-1"]
    assert len(be.created) == 1
    assert not any(fn == "conversation-history.md" for fn, _ in be.uploads)
    prompt = be.stream_calls[1][2]
    assert "get_weather" in prompt and call_id in prompt and '{"t": 30}' in prompt
    assert be.stream_calls[1][4]["openai_tools"]["get_weather"]   # re-declared
    # and the conversation keeps going natively
    m3 = m2 + [{"role": "assistant", "content": "sunny 30C"},
               {"role": "user", "content": "and tomorrow?"}]
    chat({"model": "m", "messages": m3, "tools": tools}, svc)
    assert be.stream_calls[2][0] == "chat-1"
    assert be.stream_calls[2][2] == "and tomorrow?"


def test_tool_results_plus_user_message_are_both_delivered():
    be = StubBackend(turns=[[{"type": "tool_call", "name": "f",
                              "arguments": {}}]])
    svc = make_service(be)
    m1 = [{"role": "user", "content": "go"}]
    out = chat({"model": "m", "messages": m1}, svc)
    cid = out["choices"][0]["message"]["tool_calls"][0]["id"]
    m2 = m1 + [out["choices"][0]["message"],
               {"role": "tool", "tool_call_id": cid, "content": "RESULT-7"},
               {"role": "user", "content": "also say hi"}]
    chat({"model": "m", "messages": m2}, svc)
    chat_id, _, prompt, _, _ = be.stream_calls[1]
    assert chat_id == "chat-1"
    assert "RESULT-7" in prompt and "also say hi" in prompt
    assert prompt.index("RESULT-7") < prompt.index("also say hi")


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


def test_transient_upstream_failure_keeps_conversation_in_its_chat():
    be = StubBackend(turns=[[{"type": "answer", "text": "ok"}]])
    svc = make_service(be)
    chat({"model": "m", "messages": [{"role": "user", "content": "a"}]}, svc)
    orig_stream = be.stream_turn

    def limited(*a, **k):
        raise qe.RateLimitedError("slow down")

    be.stream_turn = limited
    m2 = [{"role": "user", "content": "a"},
          {"role": "assistant", "content": "ok"},
          {"role": "user", "content": "b"}]
    with pytest.raises(qe.RateLimitedError):
        chat({"model": "m", "messages": m2}, svc)
    # retry continues in the very same chat - nothing rebuilt, nothing lost
    be.stream_turn = orig_stream
    be.turns = [[{"type": "answer", "text": "second"}]]
    out = chat({"model": "m", "messages": m2}, svc)
    assert out["choices"][0]["message"]["content"] == "second"
    assert len(be.created) == 1 and be.stream_calls[-1][0] == "chat-1"
    assert be.stream_calls[-1][2] == "b"


def test_broken_chat_is_revived_in_same_project():
    """The chat rejects the continuation (deleted upstream, bad request):
    the conversation moves to a fresh chat in the same project, with its
    history replayed, and the dead chat is cleaned up."""
    be = StubBackend(turns=[[{"type": "answer", "text": "A"}]])
    svc = make_service(be)
    m1 = [{"role": "system", "content": "S"}, {"role": "user", "content": "a"}]
    chat({"model": "m", "messages": m1}, svc)
    orig_stream = be.stream_turn

    def picky(chat_id, *a, **k):
        if chat_id == "chat-1":
            raise qe.NotFoundError("chat gone")
        return orig_stream(chat_id, *a, **k)

    be.stream_turn = picky
    be.turns = [[{"type": "answer", "text": "B"}]]
    m2 = m1 + [{"role": "assistant", "content": "A"},
               {"role": "user", "content": "b"}]
    out = chat({"model": "m", "messages": m2}, svc)
    assert out["choices"][0]["message"]["content"] == "B"
    assert be.created[-1] == ("qwen3.7-plus", "S")
    assert be.projects_created == ["proj-1"]            # project reused
    assert "chat-1" in be.deleted_chats                 # dead chat reaped
    (sess,) = svc.router.sessions()
    assert sess.chat_id == "chat-2" and sess.project_id == "proj-1"
    assert "[2] USER" in be.stream_calls[-1][2]          # history replayed
    # ...and from here on it is a normal native continuation again
    be.stream_turn = orig_stream
    chat({"model": "m", "messages": m2 + [{"role": "assistant", "content": "B"},
                                          {"role": "user", "content": "c"}]}, svc)
    assert be.stream_calls[-1][0] == "chat-2" and be.stream_calls[-1][2] == "c"


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


# ------------------------------------------------- regressions (blank/locks)
def test_interrupted_stream_resend_reruns_instead_of_blank():
    """A first-message stream that dies mid-flight must not poison the router:
    the resend used to hit the 'cached' path with an empty reply ({})."""
    be = StubBackend(turns=[[{"type": "answer", "text": "hello there"}],
                            [{"type": "answer", "text": "hello again"}]])
    svc = make_service(be)
    body = {"model": "m", "stream": True, "messages": [
        {"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]}
    kind, gen = svc.handle_chat(dict(body), stream=True)
    assert kind == "stream"
    next(gen)                      # one chunk reaches the client...
    gen.close()                    # ...then the connection dies (GeneratorExit)

    # the exact resend re-runs the turn - it is NOT served a blank cache
    kind2, payload = svc.handle_chat(dict(body), stream=False)
    assert kind2 == "json"
    assert payload["choices"][0]["message"]["content"] == "hello again"
    assert len(be.stream_calls) == 2


def test_native_continuation_keeps_full_history():
    """Turns 2..N must stay on the same upstream chat: the router needs the
    new user turns committed into the session (they used to be dropped, so
    turn 3 silently fell back to replay)."""
    be = StubBackend()
    svc = make_service(be)
    msgs = [{"role": "user", "content": "one"}]
    for word in ("two", "three", "four"):
        out = chat({"model": "m", "messages": msgs}, svc)
        reply = out["choices"][0]["message"]["content"]
        msgs = msgs + [{"role": "assistant", "content": reply},
                       {"role": "user", "content": word}]
    out = chat({"model": "m", "messages": msgs}, svc)     # 5th turn
    assert len(be.created) == 1                            # one upstream chat
    assert [c[2] for c in be.stream_calls] == ["one", "two", "three", "four"]
    # stored history is a true prefix of the client's history
    (sess,) = svc.router._sessions.values()
    assert [v["role"] for v in sess.views] == \
        ["user", "assistant", "user", "assistant", "user", "assistant",
         "user", "assistant"]


def test_stream_lock_released_when_stream_never_consumed():
    """handle_chat(stream=True) whose iterator is dropped without a single
    read must still release the global lock (generator-finally alone never
    runs for a never-started generator)."""
    be = StubBackend(turns=[[{"type": "answer", "text": "x"}]])
    svc = make_service(be)
    kind, gen = svc.handle_chat({"model": "m", "stream": True,
                                 "messages": [{"role": "user",
                                               "content": "hi"}]})
    assert kind == "stream"
    del gen                                   # abandoned, never iterated
    import gc
    gc.collect()
    assert svc._lock.acquire(timeout=1)       # would deadlock/raise if leaked
    svc._lock.release()
    # the service still serves requests afterwards
    out = chat({"model": "m", "messages": [{"role": "user", "content": "yo"}]},
               svc)
    assert out["choices"][0]["message"]["content"]


# ------------------------------------------- continuation robustness (0.4.1)
def _reply(out):
    return out["choices"][0]["message"]


def test_interrupted_native_turn_keeps_conversation_and_flags_chat():
    """A dropped connection mid-continuation used to delete the whole
    conversation upstream. Now the conversation survives; the retry goes to
    the same chat with a note that the cut-off reply was never delivered."""
    be = StubBackend(turns=[[{"type": "answer", "text": "A"}],
                            [{"type": "answer", "text": "par"},
                             {"type": "answer", "text": "tial"}],
                            [{"type": "answer", "text": "B"}]])
    svc = make_service(be)
    m1 = [{"role": "user", "content": "a"}]
    chat({"model": "m", "messages": m1}, svc)
    m2 = m1 + [{"role": "assistant", "content": "A"},
               {"role": "user", "content": "b"}]
    kind, gen = svc.handle_chat({"model": "m", "messages": m2}, stream=True)
    next(gen)
    gen.close()                                  # client vanished
    assert not be.deleted_chats                  # nothing torn down
    out = chat({"model": "m", "messages": m2}, svc)
    assert _reply(out)["content"] == "B"
    assert be.stream_calls[-1][0] == "chat-1"
    assert "cut off" in be.stream_calls[-1][2] and be.stream_calls[-1][2].endswith("b")
    (sess,) = svc.router.sessions()
    assert not sess.interrupted and sess.user_turns == 2


def test_aborted_first_turn_deletes_chat_and_project_immediately():
    be = StubBackend(turns=[[{"type": "answer", "text": "x"},
                             {"type": "answer", "text": "y"}]])
    svc = make_service(be)
    kind, gen = svc.handle_chat({"model": "m", "messages": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"}]}, stream=True)
    next(gen)
    gen.close()
    assert be.deleted_chats == ["chat-1"] and be.deleted_projects == ["proj-1"]
    assert len(svc.router) == 0 and not be.chats and not be.projects


def test_failed_first_turn_deletes_chat_and_project():
    be = StubBackend()
    svc = make_service(be)

    def boom(*a, **k):
        raise qe.APIError("upstream boom")

    be.stream_turn = boom
    with pytest.raises(qe.APIError):
        chat({"model": "m", "messages": [{"role": "system", "content": "S"},
                                         {"role": "user", "content": "hi"}]},
             svc)
    assert not be.chats and not be.projects and len(svc.router) == 0


def test_oneshot_upstream_dies_after_grace_but_memory_survives():
    be = StubBackend(turns=[[{"type": "answer", "text": "A"}],
                            [{"type": "answer", "text": "B"}],
                            [{"type": "answer", "text": "C"}]])
    svc = make_service(be, oneshot_ttl=60)
    m1 = [{"role": "system", "content": "S"}, {"role": "user", "content": "a"}]
    chat({"model": "m", "messages": m1}, svc)
    (sess,) = svc.router.sessions()
    svc.sweep(now=time.time() + 30)              # within grace: untouched
    assert be.chats and be.projects
    svc.sweep(now=time.time() + 61)              # one-shot idle -> dies
    assert be.deleted_chats == ["chat-1"] and be.deleted_projects == ["proj-1"]
    assert not be.chats and not be.projects
    assert sess.dormant and len(svc.router) == 1  # history still in memory
    # exact resend is still served from memory
    assert _reply(chat({"model": "m", "messages": m1}, svc))["content"] == "A"
    # a late follow-up revives it: fresh project + chat, history replayed
    m2 = m1 + [{"role": "assistant", "content": "A"},
               {"role": "user", "content": "b"}]
    assert _reply(chat({"model": "m", "messages": m2}, svc))["content"] == "B"
    assert sess.chat_id == "chat-2" and sess.project_id == "proj-2"
    assert be.created[-1] == ("qwen3.7-plus", "S")
    assert "Latest user message:\nb" in be.stream_calls[-1][2]
    # now a real conversation: survives the one-shot reaper, continues natively
    svc.sweep(now=time.time() + 120)
    assert not sess.dormant
    m3 = m2 + [{"role": "assistant", "content": "B"},
               {"role": "user", "content": "c"}]
    chat({"model": "m", "messages": m3}, svc)
    assert be.stream_calls[-1][:3:2] == ("chat-2", "c")


def test_default_single_turn_keeps_same_upstream_chat_until_session_ttl():
    """A single-turn conversation must keep its canonical upstream chat/project
    for the full session lifetime; early one-shot cleanup is opt-in only."""
    be = StubBackend(turns=[[{"type": "answer", "text": "A"}],
                            [{"type": "answer", "text": "B"}]])
    svc = make_service(be)
    m1 = [{"role": "system", "content": "S"}, {"role": "user", "content": "a"}]
    chat({"model": "m", "messages": m1}, svc)
    (sess,) = svc.router.sessions()

    svc.sweep(now=time.time() + 3000)
    assert not sess.dormant
    assert be.deleted_chats == [] and be.deleted_projects == []

    m2 = m1 + [{"role": "assistant", "content": "A"},
               {"role": "user", "content": "b"}]
    chat({"model": "m", "messages": m2}, svc)
    assert be.stream_calls[-1][0] == "chat-1"
    assert be.created == [("qwen3.7-plus", "S")]

    svc.sweep(now=time.time() + 3601)
    assert sess.dormant
    assert be.deleted_chats == ["chat-1"] and be.deleted_projects == ["proj-1"]
    assert len(svc.router) == 0


def test_oneshot_reaper_can_be_disabled():
    be = StubBackend()
    svc = make_service(be, oneshot_ttl=None)
    chat({"model": "m", "messages": [{"role": "user", "content": "a"}]}, svc)
    svc.sweep(now=time.time() + 3000)
    assert not be.deleted_chats                   # disabled: normal TTL only
    svc.sweep(now=time.time() + 4000)             # past the 3600 s TTL
    assert be.deleted_chats == ["chat-1"] and len(svc.router) == 0


def test_ttl_expiry_deletes_chat_and_project_and_forgets():
    be = StubBackend()
    svc = make_service(be, ttl=100)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "a"}]
    out = chat({"model": "m", "messages": msgs}, svc)
    msgs += [_reply(out), {"role": "user", "content": "b"}]
    chat({"model": "m", "messages": msgs}, svc)     # 2 user turns: not one-shot
    svc.sweep(now=time.time() + 90)
    assert not be.deleted_chats
    svc.sweep(now=time.time() + 101)
    assert be.deleted_chats == ["chat-1"] and be.deleted_projects == ["proj-1"]
    assert len(svc.router) == 0


def test_fork_reuses_project_and_project_outlives_parent():
    be = StubBackend()
    svc = make_service(be, oneshot_ttl=None)
    m1 = [{"role": "system", "content": "S"}, {"role": "user", "content": "a"}]
    out = chat({"model": "m", "messages": m1}, svc)
    m2 = m1 + [_reply(out), {"role": "user", "content": "b"}]
    chat({"model": "m", "messages": m2}, svc)
    # the user edits their 2nd message -> diverges after the first exchange
    m2e = m1 + [_reply(out), {"role": "user", "content": "b-edited"}]
    chat({"model": "m", "messages": m2e}, svc)
    assert be.stream_calls[-1][0] == "chat-2"       # new chat...
    assert be.projects_created == ["proj-1"]        # ...same project
    assert be.chats["chat-2"] == "proj-1"
    parent, fork = svc.router.sessions()
    assert parent.project is fork.project and fork.project.refs == 2
    # parent expires: its chat dies, the shared project must not
    svc.router.drop(parent)
    svc._release_upstream(parent)
    assert "chat-1" in be.deleted_chats and not be.deleted_projects
    svc.shutdown()
    assert be.deleted_projects == ["proj-1"] and not be.chats


def test_system_prompt_change_updates_project_in_place():
    be = StubBackend()
    svc = make_service(be)
    m1 = [{"role": "system", "content": "date: monday"},
          {"role": "user", "content": "a"}]
    out = chat({"model": "m", "messages": m1}, svc)
    m2 = [{"role": "system", "content": "date: tuesday"}, m1[1],
          _reply(out), {"role": "user", "content": "b"}]
    chat({"model": "m", "messages": m2}, svc)
    assert be.instruction_updates == [("proj-1", "date: tuesday")]
    assert be.stream_calls[-1][0] == "chat-1" and be.stream_calls[-1][2] == "b"
    assert len(be.created) == 1
    (sess,) = svc.router.sessions()
    assert sess.system == "date: tuesday"


def test_system_prompt_added_to_projectless_chat_opens_project_chat():
    be = StubBackend()
    svc = make_service(be)
    m1 = [{"role": "user", "content": "a"}]
    out = chat({"model": "m", "messages": m1}, svc)
    m2 = [{"role": "system", "content": "NEW"}] + m1 + [
        _reply(out), {"role": "user", "content": "b"}]
    chat({"model": "m", "messages": m2}, svc)
    assert be.created[-1] == ("qwen3.7-plus", "NEW")
    assert be.stream_calls[-1][0] == "chat-2"


def test_failed_deletions_are_retried_not_leaked():
    be = StubBackend()
    svc = make_service(be, oneshot_ttl=0)
    fails = {"n": 2}
    real = be.delete_project

    def flaky(pid):
        if fails["n"]:
            fails["n"] -= 1
            raise qe.APIError("temporary")
        real(pid)

    be.delete_project = flaky
    chat({"model": "m", "messages": [{"role": "system", "content": "S"},
                                     {"role": "user", "content": "a"}]}, svc)
    svc.sweep(now=time.time() + 1)                  # reap -> delete fails
    assert svc.graveyard and be.projects
    svc.sweep(now=time.time() + 2)                  # retry -> fails again
    svc.sweep(now=time.time() + 3)                  # retry -> succeeds
    assert not svc.graveyard and not be.projects


def test_lru_eviction_releases_upstream():
    be = StubBackend()
    svc = make_service(be, max_sessions=1, oneshot_ttl=None)
    chat({"model": "m", "messages": [{"role": "user", "content": "a"}]}, svc)
    chat({"model": "m", "messages": [{"role": "user", "content": "b"}]}, svc)
    assert be.deleted_chats == ["chat-1"] and len(svc.router) == 1


def test_many_turn_conversation_never_leaves_its_chat():
    """Mixed traffic: user turns, tool rounds, cosmetic echo differences
    and a retry - all on one upstream chat."""
    be = StubBackend(turns=[
        [{"type": "answer", "text": "hi!"}],
        [{"type": "tool_call", "name": "f", "arguments": {"x": 1}}],
        [{"type": "answer", "text": "done"}],
        [{"type": "answer", "text": "bye"}]])
    svc = make_service(be)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hello"}]
    out = chat({"model": "m", "messages": msgs}, svc)
    msgs += [{"role": "assistant", "content": [{"type": "text", "text": "hi!"}]},
             {"role": "user", "content": "use f"}]
    out = chat({"model": "m", "messages": msgs}, svc)
    tc = _reply(out)["tool_calls"][0]
    msgs += [{"role": "assistant", "content": "", "tool_calls": [
        {"id": tc["id"], "type": "function",
         "function": {"name": "f", "arguments": '{ "x" : 1 }'}}]},
        {"role": "tool", "tool_call_id": tc["id"], "content": "ok"}]
    chat({"model": "m", "messages": msgs}, svc)
    again = chat({"model": "m", "messages": msgs}, svc)     # client retry
    assert _reply(again)["content"] == "done"
    msgs += [{"role": "assistant", "content": "done  "},
             {"role": "user", "content": "bye"}]
    chat({"model": "m", "messages": msgs}, svc)
    assert {c[0] for c in be.stream_calls} == {"chat-1"}
    assert len(be.stream_calls) == 4 and len(be.created) == 1
    assert be.projects_created == ["proj-1"]
