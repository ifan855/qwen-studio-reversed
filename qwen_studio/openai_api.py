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
  server-side *project* ``custom_instruction`` state: a conversation that
  carries a system message lives inside a project holding that
  instruction, so it is enforced by the backend and never on the wire.

* **Conversation memory + routing** - every served conversation's full
  OpenAI history is kept in memory (TTL, default 3600 s) together with the
  upstream *project* and *chat* it lives in. Each request's ``messages``
  are canonicalised (whitespace, ``developer``->``system``, text-part
  lists vs strings, tool-call ids and argument formatting are all
  normalised away) and matched against **every** stored conversation; the
  best match wins:

  ======================  ==================================================
  match                   action
  ======================  ==================================================
  exact / retry           re-serve the stored reply (no upstream call)
  stored + new tail       **continue in the same project + chat**: only the
                          new messages (user turns, tool results, anything
                          the client added) go upstream as one turn
  same dialog, new        update the project's instruction in place and
  system prompt           continue in the same chat
  shared prefix, then     fork: new chat in the *same project*, history up
  diverges (edit/rewind)  to the latest message replayed into it
  never seen              replay into a fresh conversation
  ======================  ==================================================

  A stored conversation whose upstream chat is gone (one-shot reaped,
  broken chat) is *revived*: a fresh chat is opened, the history replayed,
  and from then on it continues natively again.

* **One-shot hygiene** - a conversation with a single user turn that is not
  continued within ``oneshot_ttl`` seconds (default 60) has its upstream
  chat **and project deleted**; its history stays in memory, so a late
  follow-up still works (via revival). Aborted/failed first turns are
  deleted immediately. Failed upstream deletions are retried by the
  sweeper instead of leaking.

* **Tools** - OpenAI ``tools`` function definitions are declared to Qwen as
  client-side MCP (``local_mcp``) tools. When the model invokes one, the
  proxy answers with ``finish_reason:"tool_calls"``; the client posts the
  ``role:"tool"`` results back and they are delivered to the *same* chat as
  the next turn (the upstream ``role:"function"`` continuation is
  server-gated, docs/local-tools.md). If the upstream rejects that turn the
  proxy transparently falls back to a replay in a fresh chat.

* **Files/images** - ``image_url`` (data: or https:), ``file`` and
  ``file_url`` content parts plus ``POST /v1/files`` uploads are pushed
  through :class:`qwen_studio.files.FileService`.

Everything upstream is serialised through one lock and the client's request
pacing, so a single proxy process behaves like one careful browser session.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Generator, Iterable, Iterator, List,
                    Optional, Tuple)

from . import exceptions as exc
from .chat import DEFAULT_FEATURE_CONFIG, ChatCompletion
from .client import QwenStudio
from .files import FileRef

log = logging.getLogger("qwen_studio.openai")

TOOL_SERVER_NAME = "openai_tools"   # local_mcp server bucket for OpenAI tools
HISTORY_FILENAME = "conversation-history.md"
RETRYABLE_MARKERS = ("too many messages",)
SYSTEM_ROLES = ("system", "developer")

# Upstream failures that say nothing about the health of a particular chat
# (anti-bot, rate limits, quota, auth): never "repair" a conversation over
# these - surface them so the client retries against the same chat later.
TRANSIENT_ERRORS: Tuple[type, ...] = (exc.PunishedError, exc.RateLimitedError,
                                      exc.QuotaError, exc.AuthError)
# Failures that mean "this particular chat cannot take the turn" (deleted,
# rejected): the conversation is carried over into a fresh chat.
CHAT_BROKEN_ERRORS: Tuple[type, ...] = (exc.NotFoundError, exc.BadRequestError)


# --------------------------------------------------------------------- views
def view_message(m: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce an OpenAI message to the fields that define its content.

    Keeps role/content/tool_calls/tool_call_id and normalises structured
    content parts; client decorations (``reasoning_content``, ``name``,
    arbitrary extra fields) are dropped. Views are what gets stored and
    rendered; :func:`match_key` derives the (more lenient) routing identity.
    """
    role = m.get("role")
    content = m.get("content")
    if isinstance(content, list):
        parts: List[Dict[str, Any]] = []
        for p in content:
            if isinstance(p, str):
                parts.append({"type": "text", "text": p})
                continue
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t in ("text", "input_text", "output_text"):
                parts.append({"type": "text", "text": p.get("text") or ""})
            elif t == "image_url":
                iu = p.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                parts.append({"type": "image_url", "url": url})
            elif t == "file":
                f = p.get("file") or {}
                parts.append({"type": "file", "file_id": f.get("file_id"),
                              "file_data": f.get("file_data"),
                              "filename": f.get("filename")})
            elif t == "file_url":
                fu = p.get("file_url")
                url = fu.get("url") if isinstance(fu, dict) else fu
                parts.append({"type": "file_url", "url": url})
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
            if not isinstance(c, dict):
                continue
            f = c.get("function") or {}
            tcs.append({"id": c.get("id"), "type": c.get("type") or "function",
                        "function": {"name": f.get("name"),
                                     "arguments": f.get("arguments")}})
        if tcs:
            v["tool_calls"] = tcs
    if role == "tool":
        v["tool_call_id"] = m.get("tool_call_id")
    return v


def view_key(v: Dict[str, Any]) -> str:
    """Exact (byte-level) identity of a view."""
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def _norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n")).strip()


def _canon_args(a: Any) -> str:
    """Tool-call arguments compared by value, not by JSON formatting."""
    if isinstance(a, str):
        try:
            a = json.loads(a) if a.strip() else {}
        except ValueError:
            return a.strip()
    return json.dumps(a, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _digest(s: Optional[str]) -> str:
    return hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()


def match_key(v: Dict[str, Any]) -> str:
    """Lenient routing identity of a message.

    Clients echo our replies back with small, meaning-free differences;
    matching on exact bytes made continuation fragile. Normalised away:

    * ``developer`` vs ``system`` role;
    * ``"text"`` vs ``[{"type":"text","text":"text"}]`` content;
    * ``None`` vs ``""`` content, CRLF, trailing/leading whitespace;
    * tool-call ids (tool results are identified by position) and JSON
      formatting of tool-call arguments.

    Attachments are identified by a digest of their URL / id / payload.
    """
    role = v.get("role")
    if role in SYSTEM_ROLES:
        role = "system"
    c = v.get("content")
    if isinstance(c, list):
        if all(p.get("type") == "text" for p in c):
            cv: Any = _norm_text("\n".join(p.get("text") or "" for p in c))
        else:
            parts: List[Any] = []
            for p in c:
                t = p.get("type")
                if t == "text":
                    txt = _norm_text(p.get("text"))
                    if txt:
                        parts.append(["t", txt])
                elif t in ("image_url", "file_url"):
                    parts.append([t[0], _digest(p.get("url"))])
                elif t == "file":
                    parts.append(["f", p.get("file_id") or
                                  _digest(p.get("file_data")),
                                  p.get("filename")])
                else:
                    parts.append([str(t)])
            cv = parts
    else:
        cv = _norm_text(c)
    k: Dict[str, Any] = {"r": role, "c": cv}
    calls = [[(tc.get("function") or {}).get("name"),
              _canon_args((tc.get("function") or {}).get("arguments"))]
             for tc in v.get("tool_calls") or []]
    if calls:
        k["t"] = calls
    return _digest(json.dumps(k, sort_keys=True, ensure_ascii=False))


def views_of(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [view_message(m) for m in messages if isinstance(m, dict)]


def keys_of(views: List[Dict[str, Any]]) -> List[str]:
    """Routing keys of the *dialog* part (system messages excluded)."""
    return [match_key(v) for v in dialog_of(views)]


def dialog_of(views: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [v for v in views if v.get("role") not in SYSTEM_ROLES]


def system_of(views: List[Dict[str, Any]]) -> str:
    parts = [text_of(v) for v in views
             if v.get("role") in SYSTEM_ROLES and text_of(v)]
    return "\n\n".join(parts).strip()


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


def split_latest(views: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]],
                                                         List[Dict[str, Any]]]:
    """Split a history into ``(history, latest)``.

    ``latest`` is the run of dialog messages after the final assistant
    message (the user turn(s) and/or tool results the next reply must
    answer); ``history`` is everything before it, system messages included.
    """
    cut = len(views)
    while cut > 0 and views[cut - 1].get("role") not in ("assistant",) \
            and views[cut - 1].get("role") not in SYSTEM_ROLES:
        cut -= 1
    return list(views[:cut]), list(views[cut:])


# ------------------------------------------------------------------- session
@dataclass
class UpstreamProject:
    """A Qwen project (system prompt holder), ref-counted by sessions.

    A project is only ever shared inside one conversation lineage (a
    conversation and its forks) - never between unrelated conversations -
    and is deleted when its last session releases it.
    """

    id: str
    instruction: str
    refs: int = 0


@dataclass
class Session:
    """One served conversation (in-memory, TTL-controlled).

    ``views`` is the full canonical history *including* our last reply;
    ``keys`` are the routing keys of its dialog part. ``chat_id is None``
    means the conversation is *dormant*: its upstream chat was reaped
    (one-shot hygiene) or broken, and the next continuation revives it.
    """

    id: str
    views: List[Dict[str, Any]] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)
    system: str = ""
    chat_id: Optional[str] = None
    project: Optional[UpstreamProject] = None
    model: str = ""
    last_assistant: Optional[Dict[str, Any]] = None     # full re-servable reply
    upstream_turns: int = 0          # turns served in the current chat
    interrupted: bool = False        # current chat holds an undelivered reply
    failures: int = 0                # consecutive failed turns in this chat
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.views and not self.keys:
            self.keys = keys_of(self.views)
        if self.views and not self.system:
            self.system = system_of(self.views)

    # -------------------------------------------------------------- derived
    @property
    def project_id(self) -> Optional[str]:
        return self.project.id if self.project else None

    @property
    def dormant(self) -> bool:
        return self.chat_id is None

    @property
    def user_turns(self) -> int:
        return sum(1 for v in self.views if v.get("role") == "user")

    @property
    def is_oneshot(self) -> bool:
        """A conversation that never went past its first user turn."""
        return self.user_turns <= 1

    # ------------------------------------------------------------- mutation
    def touch(self) -> None:
        self.last_used = time.time()

    def set_history(self, views: List[Dict[str, Any]]) -> None:
        self.views = list(views)
        self.keys = keys_of(self.views)
        self.system = system_of(self.views)


@dataclass
class Decision:
    mode: str                    # cached | native | fork | new | replay
    session: Optional[Session] = None
    tail: List[Dict[str, Any]] = field(default_factory=list)
    system_changed: bool = False


def _common_prefix(a: List[str], b: List[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class SessionRouter:
    """In-memory conversation store + best-match routing.

    Pure bookkeeping: the service owns upstream side effects. ``on_expire``
    (optional) is called for sessions removed by :meth:`sweep` or LRU
    eviction so the owner can release their upstream resources.
    """

    def __init__(self, ttl: float = 3600.0, max_sessions: int = 256,
                 on_expire: Optional[Callable[[Session], Any]] = None) -> None:
        self.ttl = ttl
        self.max_sessions = max_sessions
        self.on_expire = on_expire
        self._sessions: Dict[str, Session] = {}
        self._order: List[str] = []         # LRU: oldest first

    # ------------------------------------------------------------------ book
    def add(self, sess: Session) -> None:
        if sess.id in self._sessions:
            self.touch(sess)
            return
        self._sessions[sess.id] = sess
        self._order.append(sess.id)
        self._evict_over()

    def drop(self, sess: Session) -> None:
        self._sessions.pop(sess.id, None)
        if sess.id in self._order:
            self._order.remove(sess.id)

    def get(self, sid: str) -> Optional[Session]:
        return self._sessions.get(sid)

    def sessions(self) -> List[Session]:
        """Oldest-first snapshot."""
        return [self._sessions[s] for s in self._order if s in self._sessions]

    def __contains__(self, sess: Session) -> bool:
        return sess.id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    def touch(self, sess: Session) -> None:
        if sess.id in self._order:
            self._order.remove(sess.id)
            self._order.append(sess.id)
        sess.touch()

    def _expire(self, sess: Session) -> None:
        if self.on_expire:
            try:
                self.on_expire(sess)
            except Exception:  # noqa: BLE001 - cleanup must never raise
                log.debug("on_expire failed for %s", sess.id, exc_info=True)

    def _evict_over(self) -> None:
        while len(self._order) > self.max_sessions:
            oldest = self._order.pop(0)
            sess = self._sessions.pop(oldest, None)
            if sess:
                self._expire(sess)

    # ------------------------------------------------------------------ sweep
    def sweep(self, now: Optional[float] = None) -> List[Session]:
        """Remove sessions idle longer than the TTL; returns them."""
        now = now or time.time()
        expired = [s for s in self.sessions() if now - s.last_used > self.ttl]
        for s in expired:
            self.drop(s)
            self._expire(s)
        return expired

    # ----------------------------------------------------------------- decide
    def decide(self, views: List[Dict[str, Any]]) -> Decision:
        """Pick the best stored conversation for an incoming history.

        Every stored session is scored (not just the most recent one that
        shares a prefix): exact/retry beats continuation, a longer stored
        history beats a shorter one, an unchanged system prompt beats a
        changed one, and recency breaks ties.
        """
        system = system_of(views)
        keys = keys_of(views)
        dialog = dialog_of(views)
        m = len(keys)
        best: Optional[Tuple[Tuple[int, int, int], Decision]] = None

        def offer(score: Tuple[int, int, int], d: Decision) -> None:
            nonlocal best
            if best is None or score > best[0]:     # strict: recency wins ties
                best = (score, d)

        for sess in reversed(self.sessions()):      # most recent first
            k = sess.keys
            n = len(k)
            same_sys = int(sess.system == system)
            lcp = _common_prefix(k, keys)
            committed = bool(sess.last_assistant) and n > 0 and \
                dialog_of(sess.views)[-1].get("role") == "assistant"
            if not committed:
                continue
            if same_sys and lcp == m == n:
                # exact resend of a history that ends with our reply
                offer((4, n, 1), Decision("cached", session=sess))
            elif same_sys and lcp == m == n - 1:
                # stateless retry: history right up to our stored reply
                offer((4, n, 1), Decision("cached", session=sess))
            elif lcp == n < m:
                offer((3, n, same_sys),
                      Decision("native", session=sess, tail=dialog[n:],
                               system_changed=not same_sys))
            elif 0 < lcp < n:
                # shares real dialog, then diverges or rewinds (edit /
                # regenerate an earlier turn): the context lives here
                offer((2, lcp, same_sys), Decision("fork", session=sess))
        if best is not None:
            return best[1]
        if len(dialog) == 1 and dialog[0].get("role") == "user":
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
        out.extend(_content_lines(v))
        for tc in v.get("tool_calls") or []:
            f = tc.get("function") or {}
            out.append(f"- requested tool call `{tc.get('id')}`: "
                       f"**{f.get('name')}** with arguments "
                       f"`{f.get('arguments')}`")
        out.append("")
    return "\n".join(out)


def _content_lines(v: Dict[str, Any], *, fresh: bool = False) -> List[str]:
    """Text lines of a message; attachments become short placeholders."""
    c = v.get("content")
    if not isinstance(c, list):
        return [str(c or "")]
    when = "" if fresh else " earlier"
    out: List[str] = []
    for p in c:
        t = p.get("type")
        if t == "text":
            out.append(p.get("text") or "")
        elif t == "image_url":
            url = p.get("url") or ""
            label = url if url.startswith("http") else "inline image"
            out.append(f"[image attached{when}: {label}]")
        elif t == "file":
            out.append(f"[file attached{when}: "
                       f"{p.get('filename') or p.get('file_id')}]")
        elif t == "file_url":
            out.append(f"[file attached{when}: {p.get('url')}]")
    return out


def render_continuation(latest: List[Dict[str, Any]],
                        context: List[Dict[str, Any]] = ()) -> str:
    """The upstream prompt for the messages that arrived since our last
    reply (one user turn, tool results, or any mix the client produced).

    ``context`` is the preceding history, used to name tool results after
    the calls that requested them.
    """
    if len(latest) == 1 and latest[0].get("role") == "user":
        return text_of(latest[0]) or "(continue)"
    if not latest:
        return "Continue the conversation from where it left off."
    names: Dict[Optional[str], str] = {}
    for v in list(context) + list(latest):
        for tc in v.get("tool_calls") or []:
            names[tc.get("id")] = (tc.get("function") or {}).get("name") or "tool"
    only_tools = all(v.get("role") == "tool" for v in latest)
    blocks: List[str] = []
    if only_tools:
        blocks.append("Here are the results of the tool call(s) you just "
                      "requested:")
    else:
        blocks.append("New messages since your last reply, in order:")
    for v in latest:
        role = v.get("role")
        body = "\n".join(_content_lines(v, fresh=True)).strip() or "(empty)"
        if role == "tool":
            cid = v.get("tool_call_id")
            blocks.append(f"### Tool result: {names.get(cid, 'tool')} "
                          f"(call {cid})\n{body}")
        elif role == "assistant":
            lines = [body] if body != "(empty)" else []
            for tc in v.get("tool_calls") or []:
                f = tc.get("function") or {}
                lines.append(f"- called tool {f.get('name')} with "
                             f"{f.get('arguments')}")
            blocks.append("### Assistant (already sent)\n" +
                          ("\n".join(lines) or "(empty)"))
        else:
            blocks.append(f"### User\n{body}")
    last = latest[-1].get("role")
    if last == "tool":
        blocks.append("Continue using these results: answer the user, or "
                      "call a tool again if you still need more.")
    elif last == "user":
        blocks.append("Respond to the latest user message now.")
    else:
        blocks.append("Continue.")
    return "\n\n".join(blocks)


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


INTERRUPTED_NOTE = ("(Note: your previous reply in this chat was cut off "
                    "and never reached the user - disregard it.)\n\n")


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
        r = self.q.http.get(url, timeout=60,
                            headers={"User-Agent": self.q.headers(bearer=False)
                                     ["User-Agent"]})
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
    # Primitive operations raise on failure: the service decides what to do
    # (retry deletions later, fall back, surface the error).
    def create_project(self, instruction: str) -> str:
        """A project whose ``custom_instruction`` is the system prompt (the
        official server-side system-prompt mechanism)."""
        proj = self.q.projects.create(
            f"openai-proxy-{uuid.uuid4().hex[:8]}",
            description="created by the qwen-studio OpenAI-compatible proxy",
            custom_instruction=instruction)
        if not proj.id:
            raise exc.APIError("project creation returned no id")
        return proj.id

    def open_chat(self, model: str, project_id: Optional[str] = None) -> str:
        chat = self.q.chats.create(model, project_id=project_id or "")
        if not chat.id:
            raise exc.APIError("chat creation returned no id")
        return chat.id

    def set_instruction(self, project_id: str, instruction: str) -> None:
        """Change a live project's system prompt in place."""
        self.q.projects.set_system_prompt(project_id, instruction)

    def delete_chat(self, chat_id: str) -> None:
        try:
            self.q.chats.delete(chat_id)
        except exc.NotFoundError:
            pass                                   # already gone == success

    def delete_project(self, project_id: str) -> None:
        try:
            self.q.projects.delete(project_id)
        except exc.NotFoundError:
            pass

    # Backwards-compatible conveniences (pre-0.4.1 interface).
    def create_chat(self, model: str, system_prompt: str) -> Tuple[str, Optional[str]]:
        pid = self.create_project(system_prompt) if system_prompt.strip() else None
        return self.open_chat(model, pid), pid

    def cleanup(self, chat_ids: List[str], project_id: Optional[str]) -> None:
        for cid in chat_ids:
            try:
                self.delete_chat(cid)
            except Exception:  # noqa: BLE001
                pass
        if project_id:
            try:
                self.delete_project(project_id)
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


def _prime(events: Iterator[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """Pull the first upstream event *now*.

    Request-level upstream failures (chat rejected, bad request, punish
    page) then surface while the request is still being planned - before a
    single byte reaches the client - so they can be recovered from (e.g. by
    reviving the conversation in a fresh chat) or returned as a proper HTTP
    error instead of a broken SSE stream.
    """
    try:
        first = next(events)
    except StopIteration:
        first = None

    def chain() -> Generator[Dict[str, Any], None, None]:
        try:
            if first is not None:
                yield first
                yield from events
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()

    return chain()


class _LockedStream:
    """Chunk iterator that owns the global upstream lock for one streamed
    reply.

    Releases the lock exactly once - on exhaustion, on error, on
    ``close()`` (client disconnect) or on garbage collection (stream never
    started) - so an aborted request can never leave the lock held or the
    service wedged.
    """

    def __init__(self, gen: Generator[Dict[str, Any], None, None],
                 release: Any) -> None:
        self._gen = gen
        self._release = release
        self._done = False

    def __iter__(self) -> "_LockedStream":
        return self

    def __next__(self) -> Dict[str, Any]:
        try:
            return next(self._gen)
        except BaseException:
            self._finish()
            raise

    def close(self) -> None:
        try:
            self._gen.close()
        finally:
            self._finish()

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            self._release()

    def __del__(self) -> None:  # pragma: no cover - GC timing dependent
        self._finish()


class OpenAICompatService:
    """Serves OpenAI chat-completions payloads over a :class:`QwenBackend`."""

    def __init__(self, backend: QwenBackend, *, ttl: float = 3600.0,
                 max_sessions: int = 256, replay_mode: str = "both",
                 thinking: bool = False, max_images: int = 10,
                 sweeper_interval: float = 15.0,
                 oneshot_ttl: Optional[float] = 60.0,
                 max_cleanup_attempts: int = 5) -> None:
        assert replay_mode in ("both", "file", "inline")
        self.backend = backend
        self.replay_mode = replay_mode
        self.thinking = thinking
        self.max_images = max_images
        self.oneshot_ttl = oneshot_ttl
        self.max_cleanup_attempts = max_cleanup_attempts
        self.router = SessionRouter(ttl=ttl, max_sessions=max_sessions,
                                    on_expire=self._release_upstream)
        self.file_store: Dict[str, Dict[str, Any]] = {}   # file-<hex> -> record
        # upstream deletions that failed and will be retried: (kind, id, tries)
        self.graveyard: List[Tuple[str, str, int]] = []
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
                    self.sweep(blocking=False)
                except Exception:  # noqa: BLE001
                    log.debug("sweep failed", exc_info=True)

        self._sweeper = threading.Thread(target=run, daemon=True,
                                         name="qwen-oai-ttl-sweeper")
        self._sweeper.start()

    def stop_sweeper(self) -> None:
        self._stop.set()

    def sweep(self, now: Optional[float] = None, *, blocking: bool = True) -> None:
        """Expire idle conversations, reap idle one-shots, retry failed
        deletions. Runs under the upstream lock (skipped when busy and
        ``blocking=False``: every request sweeps anyway)."""
        if not self._lock.acquire(blocking=blocking):
            return
        try:
            self._sweep_locked(now)
        finally:
            self._lock.release()

    def _sweep_locked(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        self.router.sweep(now)                  # -> _release_upstream
        if self.oneshot_ttl is not None:
            for s in self.router.sessions():
                if (not s.dormant and s.is_oneshot
                        and now - s.last_used > self.oneshot_ttl):
                    log.info("one-shot %s idle %.0fs -> deleting chat %s%s",
                             s.id[:8], now - s.last_used, s.chat_id,
                             f" + project {s.project_id}" if s.project
                             and s.project.refs <= 1 else "")
                    self._release_upstream(s)
        self._retry_graveyard()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop the sweeper and delete everything still held upstream."""
        self.stop_sweeper()
        got = self._lock.acquire(timeout=timeout)
        try:
            for s in self.router.sessions():
                self.router.drop(s)
                self._release_upstream(s)
            self._retry_graveyard(final=True)
        finally:
            if got:
                self._lock.release()

    # ------------------------------------------------- upstream resources
    def _acquire_project(self, instruction: str,
                         reuse: Optional[UpstreamProject] = None
                         ) -> Optional[UpstreamProject]:
        """Project for a new chat: reuse the lineage's project when its
        instruction matches, else create one (none without a prompt)."""
        if reuse is not None and reuse.refs > 0 and reuse.instruction == instruction:
            reuse.refs += 1
            return reuse
        if not instruction.strip():
            return None
        return UpstreamProject(self.backend.create_project(instruction),
                               instruction, refs=1)

    def _release_project(self, proj: Optional[UpstreamProject]) -> None:
        if proj is None:
            return
        proj.refs -= 1
        if proj.refs <= 0:
            self._delete("project", proj.id)

    def _delete(self, kind: str, rid: str) -> None:
        try:
            if kind == "chat":
                self.backend.delete_chat(rid)
            else:
                self.backend.delete_project(rid)
        except Exception as e:  # noqa: BLE001 - never raise from cleanup
            log.warning("could not delete %s %s (%s); will retry", kind, rid, e)
            self.graveyard.append((kind, rid, 1))

    def _retry_graveyard(self, final: bool = False) -> None:
        if not self.graveyard:
            return
        pending, self.graveyard = self.graveyard, []
        pending.sort(key=lambda x: x[0] != "chat")      # chats before projects
        for kind, rid, tries in pending:
            try:
                if kind == "chat":
                    self.backend.delete_chat(rid)
                else:
                    self.backend.delete_project(rid)
            except Exception as e:  # noqa: BLE001
                if final or tries + 1 >= self.max_cleanup_attempts:
                    log.error("giving up deleting %s %s after %d tries: %s",
                              kind, rid, tries + 1, e)
                else:
                    self.graveyard.append((kind, rid, tries + 1))

    def _release_upstream(self, sess: Session) -> None:
        """Delete a session's chat and drop its project reference (the
        project dies with its last user). The session becomes dormant; its
        history is untouched."""
        if sess.chat_id:
            self._delete("chat", sess.chat_id)
        self._release_project(sess.project)
        sess.chat_id = None
        sess.project = None
        sess.upstream_turns = 0
        sess.interrupted = False
        sess.failures = 0

    _expire_session = _release_upstream          # pre-0.4.1 name

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

        ``stream=True``  -> ``("stream", iterator-of-chunk-dicts)``; the lock
        is released when the stream is fully consumed, closed on client
        disconnect, or garbage collected - exactly once, never left held.

        ``stream=False`` -> ``("json", one chat.completion object)``; the
        stream is drained under the lock and aggregated.
        """
        self._lock.acquire()
        try:
            plan = self._plan(body)
        except Exception:
            self._lock.release()
            raise
        if plan.mode == "cached":
            self._lock.release()
            return "json", plan.cached
        gen = plan.gen
        assert gen is not None
        out = _LockedStream(gen, self._lock.release)
        if stream:
            return "stream", out
        return "json", aggregate_chunks(out)

    # ------------------------------------------------------------- planning
    def _plan(self, body: Dict[str, Any]) -> Plan:
        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise exc.BadRequestError("messages must be a non-empty array",
                                      code="invalid_request_error")
        views = views_of(messages)
        if not dialog_of(views):
            raise exc.BadRequestError(
                "messages must contain at least one non-system message",
                code="invalid_request_error")
        self._sweep_locked()
        decision = self.router.decide(views)
        sess = decision.session
        log.info("route -> %s%s (%d msgs, %d conversations in memory)",
                 decision.mode,
                 f" [{sess.id[:8]} chat={sess.chat_id}]" if sess else "",
                 len(views), len(self.router))
        tools_decl = self.backend.declare_tools(body.get("tools") or [])
        if decision.mode == "cached":
            assert sess is not None
            self.router.touch(sess)
            return Plan(mode="cached", session=sess,
                        cached=dict(sess.last_assistant or {}))
        if decision.mode == "native":
            assert sess is not None
            if sess.dormant or sess.failures >= 2:
                return self._plan_open(views, tools_decl, body, mode="revive",
                                       session=sess, reuse_project=sess.project)
            return self._plan_native(sess, views, decision, tools_decl, body)
        if decision.mode == "fork":
            assert sess is not None
            return self._plan_open(views, tools_decl, body, mode="fork",
                                   reuse_project=sess.project)
        return self._plan_open(views, tools_decl, body, mode=decision.mode)


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


    def _collect_many(self, views: List[Dict[str, Any]]) -> List[Attachment]:
        """Attachments of every user message in ``views`` (one turn)."""
        out: List[Attachment] = []
        for v in views:
            if v.get("role") == "user":
                out.extend(self._collect_attachments(v))
        images = [a for a in out if a.content_type.startswith("image/")]
        if len(images) > self.max_images:
            raise exc.BadRequestError(
                f"too many images in one turn ({len(images)} > {self.max_images})",
                code="invalid_request_error")
        return out

    def _retarget_instruction(self, sess: Session, system: str) -> bool:
        """The client changed the system prompt of a live conversation:
        update the project's instruction in place when the project belongs
        to this conversation alone. False -> caller opens a new chat."""
        proj = sess.project
        if proj is None or proj.refs > 1:
            return False
        try:
            self.backend.set_instruction(proj.id, system)
        except TRANSIENT_ERRORS:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("could not update project %s instruction (%s); "
                        "opening a new chat instead", proj.id, e)
            return False
        proj.instruction = system
        sess.system = system
        return True

    # ----------------------------------------------------------- plan modes
    def _plan_native(self, sess: Session, views: List[Dict[str, Any]],
                     decision: Decision, tools_decl: Dict[str, Any],
                     body: Dict[str, Any]) -> Plan:
        """Continue a stored conversation in its own project + chat: only the
        messages the client added since our last reply go upstream."""
        tail = decision.tail
        system = system_of(views)
        if decision.system_changed and not self._retarget_instruction(sess, system):
            return self._plan_open(views, tools_decl, body, mode="fork")
        atts = self._collect_many(tail)          # client errors surface as-is
        model = self.backend.resolve_model(
            body.get("model") or sess.model,
            has_images=any(a.content_type.startswith("image/") for a in atts))
        entries = self._upload_entries(atts)
        prompt = render_continuation(tail, sess.views)
        if sess.interrupted:
            prompt = INTERRUPTED_NOTE + prompt
        try:
            events = _prime(self.backend.stream_turn(
                sess.chat_id, model, prompt, files_entries=entries,
                tools_decl=tools_decl, thinking=self.thinking))
        except CHAT_BROKEN_ERRORS as e:
            # the chat itself rejected the turn (deleted, broken, gated):
            # carry the conversation over into a fresh chat, same project
            log.warning("chat %s rejected the continuation (%s); reviving "
                        "conversation %s in a fresh chat", sess.chat_id, e,
                        sess.id[:8])
            return self._plan_open(views, tools_decl, body, mode="revive",
                                   session=sess, reuse_project=sess.project)
        except TRANSIENT_ERRORS:
            raise                                # chat is fine; client retries
        except Exception:
            # unknown upstream hiccup: keep the chat, but after repeated
            # failures the next request revives the conversation elsewhere
            sess.failures += 1
            raise

        def commit(assistant_view: Dict[str, Any], full: Dict[str, Any]) -> None:
            sess.set_history(views + [assistant_view])
            sess.last_assistant = full
            sess.model = model
            sess.upstream_turns += 1
            sess.interrupted = False
            sess.failures = 0
            self.router.touch(sess)

        def abort(error: Optional[BaseException]) -> None:
            # The upstream chat now holds a partial/undelivered turn; the
            # conversation itself is intact - keep it, flag the chat.
            sess.interrupted = True
            if error is not None and not isinstance(
                    error, (GeneratorExit,) + TRANSIENT_ERRORS):
                sess.failures += 1

        return Plan(mode="native", session=sess,
                    gen=self._drive(events, model, commit, abort))

    def _plan_open(self, views: List[Dict[str, Any]],
                   tools_decl: Dict[str, Any], body: Dict[str, Any], *,
                   mode: str, session: Optional[Session] = None,
                   reuse_project: Optional[UpstreamProject] = None) -> Plan:
        """Open a fresh upstream chat for ``views`` and answer its latest turn.

        ``new``     single user message -> plain first turn.
        ``replay``  unseen history -> engineered replay prompt (+ file).
        ``fork``    diverged from a stored conversation -> replay, reusing
                    that conversation's project when the prompt matches.
        ``revive``  a stored conversation whose chat is gone/broken ->
                    replay into a new chat; the session keeps its identity.

        Nothing is registered until the reply completes; if it does not, the
        new chat (and project, if unshared) is deleted on the spot.
        """
        system = system_of(views)
        history, latest = split_latest(views)
        atts = self._collect_many(latest)
        model = self.backend.resolve_model(
            body.get("model") or (session.model if session else ""),
            has_images=any(a.content_type.startswith("image/") for a in atts))
        project: Optional[UpstreamProject] = None
        chat_id: Optional[str] = None
        try:
            project = self._acquire_project(system, reuse_project)
            chat_id = self.backend.open_chat(model, project.id if project else None)
            latest_text = render_continuation(latest, history)
            entries: List[Dict[str, Any]] = []
            if not dialog_of(history):
                prompt = latest_text if text_of(latest[-1]) or len(latest) > 1 \
                    else "(begin)"
            else:
                transcript = render_history_document(history, system)
                prompt = build_replay_prompt(
                    latest_text, transcript,
                    inline=self.replay_mode in ("both", "inline"))
                if self.replay_mode in ("both", "file"):
                    doc = transcript.encode("utf-8")
                    try:
                        ref = self.backend.upload(doc, HISTORY_FILENAME,
                                                  "text/markdown")
                        entries.append(ref.entry(file_class="document",
                                                 show_type="file",
                                                 entry_type="file"))
                    except TRANSIENT_ERRORS:
                        raise
                    except Exception:  # noqa: BLE001 - fall back to inline
                        prompt = build_replay_prompt(latest_text, transcript,
                                                     inline=True)
            entries += self._upload_entries(atts)
            events = _prime(self.backend.stream_turn(
                chat_id, model, prompt, files_entries=entries,
                tools_decl=tools_decl, thinking=self.thinking))
        except BaseException:
            if chat_id:
                self._delete("chat", chat_id)
            self._release_project(project)
            raise

        sess = session if session is not None else Session(id=uuid.uuid4().hex)

        def commit(assistant_view: Dict[str, Any], full: Dict[str, Any]) -> None:
            if sess.chat_id or sess.project:
                self._release_upstream(sess)     # the chat we moved away from
            sess.chat_id = chat_id
            sess.project = project
            sess.upstream_turns = 1
            sess.interrupted = False
            sess.failures = 0
            sess.set_history(views + [assistant_view])
            sess.last_assistant = full
            sess.model = model
            self.router.add(sess)
            self.router.touch(sess)

        def abort(error: Optional[BaseException]) -> None:
            # never delivered -> this chat is garbage; kill it right away
            if chat_id:
                self._delete("chat", chat_id)
            self._release_project(project)

        return Plan(mode=mode, session=sess,
                    gen=self._drive(events, model, commit, abort))

    # ------------------------------------------------------------- draining
    def _drive(self, events: Iterator[Dict[str, Any]], model: str,
               on_commit: Callable[[Dict[str, Any], Dict[str, Any]], None],
               on_abort: Callable[[Optional[BaseException]], None]
               ) -> Generator[Dict[str, Any], None, None]:
        """Consume upstream events, emit OpenAI chunk dicts, then commit.

        ``on_commit`` runs only once the whole reply has been handed to the
        client; any interruption (client disconnect, upstream error) calls
        ``on_abort`` instead, so memory never holds an undelivered reply.
        """
        reply_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        answer: List[str] = []
        think: List[str] = []
        calls: List[Dict[str, Any]] = []
        first = True
        finish = "stop"
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
            yield _chunk(reply_id, created, model, {}, finish_reason=finish,
                         first=first)
        except BaseException as e:
            try:
                on_abort(e)
            except Exception:  # noqa: BLE001 - cleanup must never mask
                log.debug("abort handler failed", exc_info=True)
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
        on_commit(view_message(msg), full)

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
