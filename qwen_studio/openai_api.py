"""OpenAI-compatible service layer over the Qwen Studio client.

This module turns the stateful, phase-machine Qwen chat API into a
stateless-looking, well-behaved OpenAI ``/v1/chat/completions`` backend.
The public HTTP wrapper lives in :mod:`qwen_studio.openai_server`; the CLI
in :mod:`qwen_studio.cli`. Everything here is transport-agnostic and unit
testable against a stub backend.

How the mapping works
---------------------
* **System prompts** - the completions endpoint ignores ``role:"system"``
  messages (live-verified, docs/capabilities.md). Real system prompts are
  server-side *project* ``custom_instruction`` state: every conversation the
  proxy opens for a request that carries a system message is created inside
  a dedicated project holding that instruction, so it is enforced by the
  backend and never visible on the wire.
* **Conversation routing (the 1-hour memory)** - the proxy keeps every
  served conversation's full OpenAI message history in memory with a TTL
  (default 3600 s). For each incoming request it compares the request's
  canonicalised ``messages`` against the stored histories:

  * exact match          -> the stored assistant reply is re-served (TTL reset);
  * exact prefix + tail  -> the request continues that conversation: only the
    of ``user``/``tool``   new user turn is sent to the *same* Qwen chat
    messages               (TTL reset);
  * anything else        -> **replay mode**: a fresh Qwen conversation is
                           created and the whole unseen history - messages,
                           tool calls, tool results and attachments - is
                           prompt-engineered into it (history document
                           uploaded as a file and/or inlined). System-prompt
                           fidelity is preserved via the project mechanism.

  Calling a conversation again always resets its TTL; expired or evicted
  sessions are deleted upstream (chats + project) best-effort.
* **Tools** - OpenAI ``tools`` function definitions are declared to Qwen as
  client-side MCP (``local_mcp``) tools. When the model invokes one, the
  proxy answers the OpenAI client with ``finish_reason:"tool_calls"``; the
  client executes and posts ``role:"tool"`` results back. Because the
  upstream ``role:"function"`` continuation is server-gated
  (docs/local-tools.md), a tool-result turn is served through replay mode -
  the history document carries the calls *and* their results, and the tools
  are re-declared so the model can call them again.
* **Files/images** - ``image_url`` (data: or https:), ``file`` and
  ``file_url`` content parts plus ``POST /v1/files`` uploads are pushed
  through :class:`qwen_studio.files.FileService`. Verified live: far more
  than 5 images per turn are accepted (the 5-image cap is web-UI only).

Everything upstream is serialised through one lock and the client's request
pacing, so a single proxy process behaves like one careful browser session.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

from . import exceptions as exc
from .chat import DEFAULT_FEATURE_CONFIG, ChatCompletion
from .client import QwenStudio
from .files import FileRef

log = logging.getLogger("qwen_studio.openai")

TOOL_SERVER_NAME = "openai_tools"   # local_mcp server bucket for OpenAI tools
HISTORY_FILENAME = "conversation-history.md"
RETRYABLE_MARKERS = ("too many messages",)


# --------------------------------------------------------------------- views
def view_message(m: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce an OpenAI message to the fields that define its identity.

    Used for history-prefix comparison: clients echo assistant messages back
    verbatim (including ``reasoning_content`` and other decorations), so the
    view keeps only role/content/tool_calls/tool_call_id/name and
    normalises structured content parts.
    """
    role = m.get("role")
    content = m.get("content")
    if isinstance(content, list):
        parts: List[Dict[str, Any]] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t == "text":
                parts.append({"type": "text", "text": p.get("text") or ""})
            elif t == "image_url":
                parts.append({"type": "image_url",
                              "url": (p.get("image_url") or {}).get("url")})
            elif t == "file":
                f = p.get("file") or {}
                parts.append({"type": "file", "file_id": f.get("file_id"),
                              "file_data": f.get("file_data"),
                              "filename": f.get("filename")})
            elif t == "file_url":
                parts.append({"type": "file_url",
                              "url": (p.get("file_url") or {}).get("url")})
            else:
                parts.append({"type": str(t)})
        cv: Any = parts
    elif isinstance(content, str):
        cv = content
    else:
        cv = "" if content is None else str(content)
    v: Dict[str, Any] = {"role": role, "content": cv}
    if m.get("tool_calls"):
        tcs = []
        for c in m["tool_calls"]:
            f = c.get("function") or {}
            tcs.append({"id": c.get("id"), "type": c.get("type") or "function",
                        "function": {"name": f.get("name"),
                                     "arguments": f.get("arguments")}})
        v["tool_calls"] = tcs
    if role == "tool":
        v["tool_call_id"] = m.get("tool_call_id")
    return v


def view_key(v: Dict[str, Any]) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def views_of(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [view_message(m) for m in messages if isinstance(m, dict)]


def keys_of(views: List[Dict[str, Any]]) -> List[str]:
    return [view_key(v) for v in views]


def text_of(v: Dict[str, Any]) -> str:
    """Plain text of a viewed message (concatenating text parts)."""
    c = v.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
    return ""


def attachments_of(v: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Attachment part descriptors from a viewed message."""
    c = v.get("content")
    if not isinstance(c, list):
        return []
    return [p for p in c if p.get("type") in ("image_url", "file", "file_url")]


# ------------------------------------------------------------------- session
@dataclass
class Session:
    """One served conversation (in-memory, TTL-controlled)."""

    id: str
    views: List[Dict[str, Any]] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)
    chat_id: Optional[str] = None
    project_id: Optional[str] = None
    chats: List[str] = field(default_factory=list)      # every upstream chat made
    model: str = ""
    last_assistant: Optional[Dict[str, Any]] = None     # full re-servable reply
    pending_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used = time.time()

    def append(self, view: Dict[str, Any]) -> None:
        self.views.append(view)
        self.keys.append(view_key(view))


@dataclass
class Decision:
    mode: str                                  # cached | native | new | replay
    session: Optional[Session] = None
    tail: List[Dict[str, Any]] = field(default_factory=list)


class SessionRouter:
    """In-memory conversation store with 1-hour (configurable) TTL routing."""

    def __init__(self, ttl: float = 3600.0, max_sessions: int = 256,
                 on_expire: Optional[Any] = None) -> None:
        self.ttl = ttl
        self.max_sessions = max_sessions
        self.on_expire = on_expire          # callback(session) for upstream cleanup
        self._sessions: Dict[str, Session] = {}
        self._order: List[str] = []         # LRU: oldest first

    # ------------------------------------------------------------------ book
    def add(self, sess: Session) -> None:
        self._sessions[sess.id] = sess
        self._order.append(sess.id)
        self._evict_over()

    def drop(self, sess: Session) -> None:
        self._sessions.pop(sess.id, None)
        if sess.id in self._order:
            self._order.remove(sess.id)

    def get(self, sid: str) -> Optional[Session]:
        return self._sessions.get(sid)

    def __len__(self) -> int:
        return len(self._sessions)

    def _touch_order(self, sid: str) -> None:
        if sid in self._order:
            self._order.remove(sid)
        self._order.append(sid)

    def _evict_over(self) -> None:
        while len(self._order) > self.max_sessions:
            oldest = self._order.pop(0)
            sess = self._sessions.pop(oldest, None)
            if sess and self.on_expire:
                try:
                    self.on_expire(sess)
                except Exception:  # noqa: BLE001 - cleanup must never raise
                    pass

    # ------------------------------------------------------------------ sweep
    def sweep(self, now: Optional[float] = None) -> List[Session]:
        """Expire sessions idle longer than the TTL; returns expired ones."""
        now = now or time.time()
        expired = [s for s in self._sessions.values()
                   if now - s.last_used > self.ttl]
        for s in expired:
            self.drop(s)
            if self.on_expire:
                try:
                    self.on_expire(s)
                except Exception:  # noqa: BLE001
                    pass
        return expired

    # ----------------------------------------------------------------- decide
    def decide(self, views: List[Dict[str, Any]]) -> Decision:
        keys = keys_of(views)
        for sid in reversed(self._order):               # most recent first
            sess = self._sessions.get(sid)
            if sess is None:
                continue
            n = len(sess.keys)
            if keys == sess.keys:
                self._touch_order(sid)
                sess.touch()
                return Decision("cached", session=sess)
            if len(keys) > n and keys[:n] == sess.keys:
                tail = views[n:]
                if all(t.get("role") in ("user", "tool") for t in tail):
                    self._touch_order(sid)
                    sess.touch()
                    return Decision("native", session=sess, tail=tail)
                break                                    # rewound/edited -> replay
            if len(keys) == n - 1 and keys == sess.keys[:len(keys)]:
                # stateless retry / "continue from this exact history": the
                # next stored turn is our own last reply -> re-serve it.
                if sess.views and sess.views[-1].get("role") == "assistant" \
                        and sess.last_assistant:
                    self._touch_order(sid)
                    sess.touch()
                    return Decision("cached", session=sess)
        non_system = [v for v in views if v.get("role") not in ("system", "developer")]
        if len(non_system) == 1 and non_system[0].get("role") == "user":
            return Decision("new")
        return Decision("replay")


# ------------------------------------------------------------------ replay
def render_history_document(views: List[Dict[str, Any]],
                            system_prompt: Optional[str] = None) -> str:
    """Render the canonical conversation history as a markdown document."""
    out = ["# Conversation continuation context", "",
           "This document is the authoritative record of an ongoing",
           "conversation between a user and an assistant (you). Treat every",
           "message, tool call and tool result below as your own memory.", ""]
    if system_prompt:
        out += ["## Active system instructions", "", system_prompt.strip(), ""]
    out += ["## Full message history", ""]
    for i, v in enumerate(views, 1):
        role = (v.get("role") or "unknown").upper()
        if role == "TOOL":
            out.append(f"### [{i}] TOOL RESULT (for call {v.get('tool_call_id')})")
        else:
            out.append(f"### [{i}] {role}")
        c = v.get("content")
        if isinstance(c, list):
            for p in c:
                t = p.get("type")
                if t == "text":
                    out.append(p.get("text") or "")
                elif t == "image_url":
                    url = p.get("url") or ""
                    label = url if url.startswith("http") else "inline image"
                    out.append(f"[image attached earlier: {label}]")
                elif t == "file":
                    out.append(f"[file attached earlier: "
                               f"{p.get('filename') or p.get('file_id')}]")
                elif t == "file_url":
                    out.append(f"[file attached earlier: {p.get('url')}]")
        else:
            out.append(str(c or ""))
        for tc in v.get("tool_calls") or []:
            f = tc.get("function") or {}
            out.append(f"- requested tool call `{tc.get('id')}`: "
                       f"**{f.get('name')}** with arguments "
                       f"`{f.get('arguments')}`")
        out.append("")
    return "\n".join(out)


def build_replay_prompt(latest_text: str, transcript: str, *,
                        inline: bool) -> str:
    """The engineered prompt that makes Qwen follow an unseen history."""
    head = (
        "You are resuming an existing conversation with a user. The complete "
        "prior conversation - every message, tool call and tool result - is "
        "provided for you. Treat that history as your own memory: keep its "
        "facts, decisions, names, style and pending tasks exactly as if you "
        "had lived through it. Do not summarise the history and do not "
        "mention the history file. Respond ONLY to the latest user message "
        "at the end, consistently with that entire history.")
    if inline:
        return (f"{head}\n\n"
                f"--- Conversation history (verbatim) ---\n{transcript}\n"
                f"--- End of conversation history ---\n\n"
                f"Latest user message:\n{latest_text}\n\n"
                f"Respond to the latest user message now.")
    return (f"{head} The full history is in the attached file "
            f"\"{HISTORY_FILENAME}\" - read it first.\n\n"
            f"Latest user message:\n{latest_text}\n\n"
            f"Respond to the latest user message now.")


# ------------------------------------------------------------------ backend
class QwenBackend:
    """All upstream Qwen operations the proxy needs, behind one object.

    Tests substitute this with a stub; :class:`OpenAICompatService` never
    touches :class:`~qwen_studio.client.QwenStudio` directly.
    """

    MODEL_CACHE_TTL = 300.0

    def __init__(self, client: QwenStudio, default_model: Optional[str] = None) -> None:
        self.q = client
        self.default_model = default_model
        self._models_at = 0.0
        self._models: List[str] = []

    # ------------------------------------------------------------------ misc
    def model_ids(self) -> List[str]:
        if time.time() - self._models_at > self.MODEL_CACHE_TTL or not self._models:
            try:
                self._models = self.q.list_model_ids() or []
            except Exception:  # noqa: BLE001 - keep stale list on failure
                if not self._models:
                    raise
            self._models_at = time.time()
        return self._models

    def resolve_model(self, requested: str, has_images: bool = False) -> str:
        requested = (requested or "").strip()
        ids = self.model_ids()
        default = self.default_model or (ids[0] if ids else requested)
        model = requested if requested in ids else default
        if has_images and not any(
                k in model.lower() for k in ("vl", "omni", "vision", "image")):
            vision = next((m for m in ids
                           if any(k in m.lower()
                                  for k in ("vl", "omni", "vision", "image"))), None)
            if vision:
                model = vision
        return model

    def download(self, url: str) -> Tuple[bytes, str]:
        from .client import as_transport_error, host_of
        try:
            r = self.q.http.get(
                url, timeout=60,
                headers={"User-Agent": self.q.headers(bearer=False)["User-Agent"]})
        except Exception as e:  # noqa: BLE001 - typed below
            te = as_transport_error(e, host_of(url))
            if te is None:
                raise
            raise te from e
        if r.status_code != 200:
            raise exc.APIError(f"attachment download failed: HTTP {r.status_code}",
                               status=r.status_code, details=url)
        ct = (r.headers.get("content-type") or "application/octet-stream")
        ct = ct.split(";")[0].strip()
        return r.content, ct

    def upload(self, data: bytes, filename: str, content_type: str) -> FileRef:
        if content_type.startswith("image/"):
            return self.q.files.upload_image(data, filename, content_type)
        return self.q.files.upload(data, filename, content_type)

    # ------------------------------------------------------------ lifecycle
    def create_chat(self, model: str, system_prompt: str) -> Tuple[str, Optional[str]]:
        """Create the upstream conversation; inside a project when a system
        prompt exists (the official server-side system-prompt mechanism)."""
        if system_prompt.strip():
            proj = self.q.projects.create(
                f"openai-proxy-{uuid.uuid4().hex[:8]}",
                description="created by the qwen-studio OpenAI-compatible proxy",
                custom_instruction=system_prompt)
            chat = self.q.chats.create(model, project_id=proj.id)
            return chat.id, proj.id
        chat = self.q.chats.create(model)
        return chat.id, None

    def cleanup(self, chat_ids: List[str], project_id: Optional[str]) -> None:
        for cid in chat_ids:
            try:
                self.q.chats.delete(cid)
            except Exception:  # noqa: BLE001
                pass
        if project_id:
            try:
                self.q.projects.delete(project_id)
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------------- stream
    def declare_tools(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """OpenAI ``tools`` -> the ``feature_config.local_mcp`` declaration."""
        bucket: Dict[str, Any] = {}
        for t in tools or []:
            if not isinstance(t, dict) or t.get("type") not in (None, "function"):
                continue
            f = t.get("function") or {}
            name = f.get("name")
            if not name:
                continue
            schema = f.get("parameters") or {"type": "object", "properties": {}}
            bucket[name] = {"description": f.get("description") or name,
                            "input_schema": schema}
        return {TOOL_SERVER_NAME: bucket} if bucket else {}

    def stream_turn(self, chat_id: str, model: str, prompt: str, *,
                    files_entries: Optional[List[Dict[str, Any]]] = None,
                    tools_decl: Optional[Dict[str, Any]] = None,
                    thinking: bool = False) -> Generator[Dict[str, Any], None, None]:
        """One user turn -> normalised upstream events:

        ``{"type":"created"|"answer"|"thinking"|"tool_call"|"done", ...}``
        """
        fc = dict(DEFAULT_FEATURE_CONFIG)
        if thinking:
            fc.update(thinking_enabled=True, auto_thinking=True,
                      thinking_mode="Enable")
        if tools_decl:
            fc["local_mcp"] = tools_decl
        msg = ChatCompletion.user_message(prompt, model, feature_config=fc,
                                          files=files_entries or [])
        body = ChatCompletion.build_body(chat_id, [msg], model)
        response_id = None
        for ev in self.q.stream_events(body, chat_id):
            if ev.type == "created":
                response_id = ev.response_id
                yield {"type": "created", "response_id": response_id}
                continue
            if ev.type != "delta":
                continue
            if ev.status == "error":
                raise exc.APIError(
                    f"upstream stream error: {ev.content or ev.phase}",
                    details=ev.raw)
            if ev.phase == "answer" and ev.content:
                yield {"type": "answer", "text": ev.content}
            elif ev.phase == "thinking_summary" and ev.content:
                yield {"type": "thinking", "text": ev.content}
            lm = ev.extra.get("local_mcp")
            if lm:
                entries: List[Any] = []
                if isinstance(lm, dict):
                    for v in lm.values():
                        entries.extend(v if isinstance(v, list) else [v])
                elif isinstance(lm, list):
                    entries = lm
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    name = e.get("tool_name")
                    if not name and isinstance(e, dict) and e:
                        name = next(iter(e))
                    yield {"type": "tool_call", "name": name,
                           "arguments": e.get("params") or {}}
        yield {"type": "done", "response_id": response_id}


class _OnceRelease:
    """Release a lock at most once, whichever teardown path runs first.

    A chat turn can finish in several ways (generator exhausted, closed by
    the HTTP layer, finalised by the GC during exception unwinding, drained
    by the non-streaming aggregator); several of them can observe the same
    turn. Guarding the release keeps the *original* exception visible -
    without it a failing upstream turn surfaced as
    ``RuntimeError: cannot release un-acquired lock`` instead of the mapped
    upstream error.
    """

    __slots__ = ("_lock", "_done")

    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._done = False

    def __call__(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._lock.release()
        except RuntimeError:      # pragma: no cover - defensive only
            pass


# ------------------------------------------------------------------- service
@dataclass
class Attachment:
    filename: str
    content_type: str
    data: bytes


@dataclass
class Plan:
    mode: str
    session: Optional[Session] = None
    cached: Optional[Dict[str, Any]] = None       # full json reply (cached mode)
    gen: Optional[Generator[Dict[str, Any], None, None]] = None


class OpenAICompatService:
    """Serves OpenAI chat-completions payloads over a :class:`QwenBackend`."""

    def __init__(self, backend: QwenBackend, *, ttl: float = 3600.0,
                 max_sessions: int = 256, replay_mode: str = "both",
                 thinking: bool = False, max_images: int = 10,
                 sweeper_interval: float = 60.0) -> None:
        assert replay_mode in ("both", "file", "inline")
        self.backend = backend
        self.replay_mode = replay_mode
        self.thinking = thinking
        self.max_images = max_images
        self.router = SessionRouter(ttl=ttl, max_sessions=max_sessions,
                                    on_expire=self._expire_session)
        self.file_store: Dict[str, Dict[str, Any]] = {}   # file-<hex> -> record
        self._lock = threading.RLock()
        self._sweeper: Optional[threading.Thread] = None
        self._sweeper_interval = sweeper_interval
        self._stop = threading.Event()

    # ------------------------------------------------------------- lifecycle
    def start_sweeper(self) -> None:
        if self._sweeper and self._sweeper.is_alive():
            return

        def run() -> None:
            while not self._stop.wait(self._sweeper_interval):
                try:
                    self.router.sweep()
                except Exception:  # noqa: BLE001
                    pass

        self._sweeper = threading.Thread(target=run, daemon=True,
                                         name="qwen-oai-ttl-sweeper")
        self._sweeper.start()

    def stop_sweeper(self) -> None:
        self._stop.set()

    def _expire_session(self, sess: Session) -> None:
        log.info("session %s expired -> cleaning %d chat(s), project=%s",
                 sess.id[:8], len(sess.chats), sess.project_id)
        try:
            self.backend.cleanup(sess.chats, sess.project_id)
        except Exception:  # noqa: BLE001
            pass

    # ----------------------------------------------------------- /v1/models
    def handle_models(self) -> Dict[str, Any]:
        now = int(time.time())
        data = [{"id": mid, "object": "model", "created": now,
                 "owned_by": "qwen"} for mid in self.backend.model_ids()]
        return {"object": "list", "data": data}

    # ------------------------------------------------------------ /v1/files
    def handle_file_upload(self, data: bytes, filename: str,
                           content_type: str) -> Dict[str, Any]:
        fid = f"file-{uuid.uuid4().hex[:24]}"
        rec = {"id": fid, "object": "file", "bytes": len(data),
               "created_at": int(time.time()), "filename": filename,
               "purpose": "assistants", "content_type": content_type,
               "data": data}
        self.file_store[fid] = rec
        return {k: v for k, v in rec.items() if k != "data"}

    def handle_file_get(self, file_id: str) -> Optional[Dict[str, Any]]:
        rec = self.file_store.get(file_id)
        if not rec:
            return None
        return {k: v for k, v in rec.items() if k != "data"}

    # ------------------------------------------------------- chat completion
    def handle_chat(self, body: Dict[str, Any], *, stream: bool = True) -> Any:
        """Serialised entry point.

        ``stream=True``  -> ``("stream", generator-of-chunk-dicts)``; the lock
        is released when the generator is fully consumed (or closed on client
        disconnect).

        ``stream=False`` -> ``("json", one chat.completion object)``; the
        generator is drained under the lock and aggregated.
        """
        self._lock.acquire()
        try:
            plan = self._plan(body)
        except Exception:
            self._lock.release()
            raise
        if plan.mode == "cached":
            try:
                return "json", plan.cached
            finally:
                self._lock.release()
        gen = plan.gen
        assert gen is not None

        # The lock must be released exactly once, by whichever of these gets
        # there first: normal exhaustion, close() from the HTTP layer, the
        # garbage collector finalising the generator while an upstream error
        # unwinds, or the non-streaming drain below. A bare ``release()`` in
        # both the generator's ``finally`` and the error path used to fire
        # twice on any failing upstream turn, so the real error was replaced
        # by ``RuntimeError: cannot release un-acquired lock``.
        unlock = _OnceRelease(self._lock)

        def guarded() -> Generator[Dict[str, Any], None, None]:
            try:
                yield from gen
            finally:
                unlock()

        if stream:
            return "stream", guarded()
        try:
            return "json", aggregate_chunks(guarded())
        except Exception:
            unlock()            # the generator may already have been finalised
            raise

    # ------------------------------------------------------------- planning
    def _plan(self, body: Dict[str, Any]) -> Plan:
        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise exc.BadRequestError("messages must be a non-empty array",
                                      code="invalid_request_error")
        views = views_of(messages)
        self.router.sweep()
        decision = self.router.decide(views)
        log.info("route -> %s (%d msgs, %d sessions in memory)",
                 decision.mode, len(views), len(self.router))
        tools_decl = self.backend.declare_tools(body.get("tools") or [])
        if decision.mode == "cached":
            return self._plan_cached(decision.session)          # type: ignore[arg-type]
        if decision.mode == "native":
            return self._plan_native(decision.session, decision.tail,  # type: ignore[arg-type]
                                     tools_decl, body)
        if decision.mode == "new":
            return self._plan_new(views, tools_decl, body)
        return self._plan_replay(views, tools_decl, body)

    # ---------------------------------------------------------------- pieces
    def _collect_attachments(self, view: Dict[str, Any]) -> List[Attachment]:
        """Materialise attachment parts of one message into bytes."""
        out: List[Attachment] = []
        for p in attachments_of(view):
            if p["type"] == "image_url":
                url = p.get("url") or ""
                if url.startswith("data:"):
                    a = _data_url_to_attachment(url, "image")
                elif url.startswith("http"):
                    data, ct = self.backend.download(url)
                    a = Attachment(_name_from_url(url, "image"), ct, data)
                else:
                    raise exc.BadRequestError(f"unsupported image url: {url[:60]}",
                                              code="invalid_request_error")
                if not a.content_type.startswith("image/"):
                    raise exc.BadRequestError(
                        "image_url part is not an image", code="invalid_request_error")
                out.append(a)
            elif p["type"] == "file_url":
                url = p.get("url") or ""
                if url.startswith("data:"):
                    a = _data_url_to_attachment(url, "file")
                elif url.startswith("http"):
                    data, ct = self.backend.download(url)
                    a = Attachment(_name_from_url(url, "file"), ct, data)
                else:
                    raise exc.BadRequestError(f"unsupported file url: {url[:60]}",
                                              code="invalid_request_error")
                out.append(a)
            elif p["type"] == "file":
                fid = p.get("file_id")
                fd = p.get("file_data")
                if fid:
                    rec = self.file_store.get(fid)
                    if not rec:
                        raise exc.BadRequestError(f"unknown file_id {fid}",
                                                  code="invalid_request_error")
                    out.append(Attachment(rec["filename"],
                                          rec["content_type"], rec["data"]))
                elif fd:
                    name = p.get("filename") or "upload.bin"
                    a = _data_url_to_attachment(fd if ":" in fd else
                                                f"application/octet-stream;name={name};base64,{fd}",
                                                "file")
                    a.filename = name if not p.get("filename") else name
                    out.append(a)
        images = [a for a in out if a.content_type.startswith("image/")]
        if len(images) > self.max_images:
            raise exc.BadRequestError(
                f"too many images in one turn ({len(images)} > {self.max_images})",
                code="invalid_request_error")
        return out

    def _upload_entries(self, atts: List[Attachment]) -> List[Dict[str, Any]]:
        entries = []
        for a in atts:
            ref = self.backend.upload(a.data, a.filename, a.content_type)
            if a.content_type.startswith("image/"):
                entries.append(ref.entry())
            else:
                entries.append(ref.entry(file_class="document",
                                         show_type="file", entry_type="file"))
        return entries

    @staticmethod
    def _system_prompt(views: List[Dict[str, Any]]) -> str:
        parts = [text_of(v) for v in views
                 if v.get("role") in ("system", "developer") and text_of(v)]
        return "\n\n".join(parts).strip()

    def _finish(self, sess: Session, assistant_view: Dict[str, Any],
                full: Dict[str, Any]) -> None:
        sess.append(assistant_view)
        sess.last_assistant = full
        sess.pending_tool_calls = full.get("tool_calls") or []
        sess.touch()

    def _drop_session(self, sess: Optional[Session]) -> None:
        """On upstream failure: forget the session so retries rebuild cleanly."""
        if sess is not None:
            self.router.drop(sess)

    # ----------------------------------------------------------- plan modes
    def _plan_cached(self, sess: Session) -> Plan:
        full = dict(sess.last_assistant or {})
        return Plan(mode="cached", session=sess, cached=full)

    def _plan_native(self, sess: Session, tail: List[Dict[str, Any]],
                     tools_decl: Dict[str, Any], body: Dict[str, Any]) -> Plan:
        last_user = next((v for v in reversed(tail) if v.get("role") == "user"), None)
        if last_user is None:
            # tool-only tail should not happen (decide routes it to replay);
            # be defensive and replay the whole thing.
            return self._plan_replay(sess.views + tail, tools_decl, body)
        try:
            atts = self._collect_attachments(last_user)
            entries = self._upload_entries(atts)
            model = self.backend.resolve_model(
                body.get("model") or sess.model,
                has_images=any(a.content_type.startswith("image/") for a in atts))
            prompt = text_of(last_user) or "(continue)"
            events = self.backend.stream_turn(
                sess.chat_id, model, prompt, files_entries=entries,
                tools_decl=tools_decl, thinking=self.thinking)
        except Exception:
            self._drop_session(sess)
            raise
        return Plan(mode="native", session=sess,
                    gen=self._drive(sess, events, model, sess.views + tail))

    def _plan_new(self, views: List[Dict[str, Any]],
                  tools_decl: Dict[str, Any], body: Dict[str, Any]) -> Plan:
        system = self._system_prompt(views)
        last = views[-1]
        sess = Session(id=uuid.uuid4().hex, views=list(views),
                       keys=keys_of(views))
        try:
            atts = self._collect_attachments(last)
        except Exception:
            self._drop_session(sess)
            raise
        # resolve the model first (needs attachment info for vision routing)
        try:
            model = self.backend.resolve_model(
                body.get("model") or "",
                has_images=any(a.content_type.startswith("image/") for a in atts))
            chat_id, project_id = self.backend.create_chat(model, system)
        except Exception:
            self._drop_session(sess)
            raise
        sess.chat_id = chat_id
        sess.project_id = project_id
        sess.chats = [chat_id]
        sess.model = model
        try:
            entries = self._upload_entries(atts)
            events = self.backend.stream_turn(
                chat_id, model, text_of(last) or "(begin)",
                files_entries=entries, tools_decl=tools_decl,
                thinking=self.thinking)
        except Exception:
            self._drop_session(sess)
            raise
        self.router.add(sess)
        return Plan(mode="new", session=sess,
                    gen=self._drive(sess, events, model, views))

    def _plan_replay(self, views: List[Dict[str, Any]],
                     tools_decl: Dict[str, Any], body: Dict[str, Any]) -> Plan:
        system = self._system_prompt(views)
        last = views[-1]
        if last.get("role") == "tool":
            # A tool-result turn: the answer continues the pending tool call.
            prompt_tail = "Provide the next assistant turn using the tool results above."
        else:
            prompt_tail = text_of(last)
        sess = Session(id=uuid.uuid4().hex, views=list(views),
                       keys=keys_of(views))
        try:
            atts = self._collect_attachments(last)
            model = self.backend.resolve_model(
                body.get("model") or "",
                has_images=any(a.content_type.startswith("image/") for a in atts))
            transcript = render_history_document(views[:-1], system)
            prompt = build_replay_prompt(prompt_tail or "(continue)",
                                         transcript,
                                         inline=self.replay_mode in ("both", "inline"))
            chat_id, project_id = self.backend.create_chat(model, system)
        except Exception:
            self._drop_session(sess)
            raise
        sess.chat_id = chat_id
        sess.project_id = project_id
        sess.chats = [chat_id]
        sess.model = model
        try:
            entries = []
            if self.replay_mode in ("both", "file"):
                doc = render_history_document(views, system).encode("utf-8")
                try:
                    ref = self.backend.upload(doc, HISTORY_FILENAME,
                                              "text/markdown")
                    entries.append(ref.entry(file_class="document",
                                             show_type="file",
                                             entry_type="file"))
                except Exception:  # noqa: BLE001 - fall back to inline text
                    prompt = build_replay_prompt(prompt_tail or "(continue)",
                                                 transcript, inline=True)
            entries += self._upload_entries(atts)
            events = self.backend.stream_turn(
                chat_id, model, prompt, files_entries=entries,
                tools_decl=tools_decl, thinking=self.thinking)
        except Exception:
            self._drop_session(sess)
            raise
        self.router.add(sess)
        return Plan(mode="replay", session=sess,
                    gen=self._drive(sess, events, model, views))

    # ------------------------------------------------------------- draining
    def _drive(self, sess: Session, events: Generator[Dict[str, Any], None, None],
               model: str, request_views: List[Dict[str, Any]]) -> Generator[
                   Dict[str, Any], None, None]:
        """Consume upstream events, emit OpenAI chunk dicts, then commit."""
        reply_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        answer: List[str] = []
        think: List[str] = []
        calls: List[Dict[str, Any]] = []
        first = True
        try:
            for ev in events:
                if ev["type"] == "answer":
                    answer.append(ev["text"])
                    yield _chunk(reply_id, created, model,
                                 {"content": ev["text"]}, first=first)
                    first = False
                elif ev["type"] == "thinking":
                    think.append(ev["text"])
                    if self.thinking:
                        yield _chunk(reply_id, created, model,
                                     {"reasoning_content": ev["text"]},
                                     first=first)
                        first = False
                elif ev["type"] == "tool_call":
                    tc = {"id": f"call_{uuid.uuid4().hex[:24]}",
                          "type": "function",
                          "function": {"name": ev["name"],
                                       "arguments": json.dumps(
                                           ev.get("arguments") or {},
                                           ensure_ascii=False)}}
                    calls.append(tc)
                    yield _chunk(reply_id, created, model,
                                 {"tool_calls": [
                                     {"index": len(calls) - 1, **tc}]},
                                 first=first)
                    first = False
            finish = "tool_calls" if calls else "stop"
            yield _chunk(reply_id, created, model, {}, finish_reason=finish)
        except Exception:
            self._drop_session(sess)
            raise
        content = "".join(answer)
        msg: Dict[str, Any] = {"role": "assistant",
                               "content": content if content else (
                                   None if calls else "")}
        if think and self.thinking:
            msg["reasoning_content"] = "".join(think)
        if calls:
            msg["tool_calls"] = calls
        full = {"id": reply_id, "object": "chat.completion",
                "created": created, "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": _null_usage()}
        # The request's history is authoritative for what this conversation
        # now is: adopt it wholesale, then append the reply we just produced.
        # Without this a *native* continuation left the new user turn out of
        # the stored views, so the very next request no longer matched the
        # stored prefix and silently fell back to the expensive replay path
        # (fresh upstream chat + history upload) for every later turn.
        sess.views = list(request_views)
        sess.keys = keys_of(sess.views)
        self._finish(sess, view_message(msg), full)

    # ------------------------------------------------------------- streaming
    def stream_cached(self, full: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """Re-serve a cached reply as OpenAI SSE chunks (streaming clients)."""
        choice = (full.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        reply_id = full.get("id") or f"chatcmpl-{uuid.uuid4().hex}"
        model = full.get("model") or ""
        created = int(time.time())
        yield _chunk(reply_id, created, model, {"role": "assistant"}, first=True)
        if msg.get("reasoning_content"):
            yield _chunk(reply_id, created, model,
                         {"reasoning_content": msg["reasoning_content"]})
        if msg.get("content"):
            yield _chunk(reply_id, created, model, {"content": msg["content"]})
        for i, tc in enumerate(msg.get("tool_calls") or []):
            yield _chunk(reply_id, created, model,
                         {"tool_calls": [{"index": i, **tc}]})
        yield _chunk(reply_id, created, model, {},
                     finish_reason=choice.get("finish_reason") or "stop")


# ------------------------------------------------------------------ helpers
def aggregate_chunks(chunks: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold OpenAI chunk dicts into one chat.completion object."""
    reply_id: Optional[str] = None
    model = ""
    created: Optional[int] = None
    content: List[str] = []
    reasoning: List[str] = []
    tool_calls: Dict[int, Dict[str, Any]] = {}
    finish = "stop"
    for ch in chunks:
        reply_id = ch.get("id") or reply_id
        model = ch.get("model") or model
        created = ch.get("created") or created
        for c in ch.get("choices") or []:
            d = c.get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                cur = tool_calls.setdefault(i, {
                    "id": tc.get("id"), "type": tc.get("type") or "function",
                    "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    cur["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    cur["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    cur["function"]["arguments"] += fn["arguments"]
            if c.get("finish_reason"):
                finish = c["finish_reason"]
    msg: Dict[str, Any] = {"role": "assistant",
                           "content": "".join(content) or (
                               None if tool_calls else "")}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return {"id": reply_id or f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created or int(time.time()), "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": _null_usage()}


def _chunk(reply_id: str, created: int, model: str, delta: Dict[str, Any], *,
           first: bool = False, finish_reason: Optional[str] = None) -> Dict[str, Any]:
    d = {"role": "assistant"} if first else {}
    d.update(delta)
    return {"id": reply_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": d, "finish_reason": finish_reason}]}


def _null_usage() -> Dict[str, Any]:
    return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def _name_from_url(url: str, kind: str) -> str:
    tail = url.split("?")[0].rstrip("/").split("/")[-1] or f"{kind}.bin"
    return tail[:80]


def _data_url_to_attachment(url: str, kind: str) -> Attachment:
    import base64 as _b64
    try:
        head, b64 = url.split(",", 1)
        meta = head[5:]
        parts = [p for p in meta.split(";") if p]
        ct = parts[0] if parts and "/" in parts[0] else "application/octet-stream"
        name = next((p.split("=", 1)[1] for p in parts if p.startswith("name=")),
                    None)
    except ValueError:
        raise exc.BadRequestError("malformed data: URL",
                                  code="invalid_request_error")
    try:
        data = _b64.b64decode(b64)
    except Exception as e:  # noqa: BLE001
        raise exc.BadRequestError(f"malformed data: URL payload: {e}",
                                  code="invalid_request_error")
    filename = name or f"{kind}-{uuid.uuid4().hex[:8]}"
    if ct.startswith("image/") and "." not in filename:
        ext = ct.split("/")[1].split("+")[0]
        filename = f"{filename}.{ext}"
    return Attachment(filename, ct, data)
