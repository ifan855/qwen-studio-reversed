"""Server-Sent Events parsing and the chat phase model.

Qwen Studio's chat pipeline is *not* a plain text delta stream. It is a
phase machine delivered over SSE where each ``data:`` frame is a JSON object
shaped like an OpenAI-style chunk::

    data: {"response.created": {"chat_id": "...", "response_id": "...",
                                "parent_id": "...", "response_index": "0"}}
    data: {"choices": [{"delta": {"role": "assistant", "phase": "answer",
                                  "status": "typing", "content": "Hel"}}],
           "response_id": "...", "usage": {...}, "timestamp": 1790212954}

Phases observed on the wire include ``answer``, ``thinking_summary``,
``web_search`` / ``search_result``, built-in tool phases (``tool_call``,
``bash``, ``code_interpreter``, ``fetch_page``, ``generate_image``, ...) and
the MCP phases (``local_tool`` for client-executed tools). Every delta also
carries a ``status`` (``typing``, ``finished``, ``error``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .exceptions import StreamInterruptedError

FINISHED = "finished"
TYPING = "typing"
ERROR = "error"


@dataclass
class ChatEvent:
    """One parsed SSE frame, normalised for consumption.

    Attributes:
        type: ``created`` for the response.created envelope, ``delta`` for
            choice deltas, ``raw`` for anything unrecognised.
        phase: phase name when present (``answer``, ``local_tool``, ...).
        status: delta status (``typing`` / ``finished`` / ``error``).
        content: incremental text content, when present.
        role: delta role (``assistant``, ``tool``, ...).
        extra: the delta's ``extra`` dict - carries thinking summaries,
            ``local_mcp`` tool-call descriptors, ``tool_name`` /
            ``tool_call_id`` / ``tool_result`` for built-in tools,
            ``mcp_results`` for finished local-tool turns.
        usage: token usage statistics attached to a frame, when present.
        response_id: response id, populated from the created envelope.
        raw: the full parsed JSON object.
    """

    type: str
    phase: Optional[str] = None
    status: Optional[str] = None
    content: Optional[str] = None
    role: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    usage: Optional[Dict[str, Any]] = None
    response_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_tool_phase(self) -> bool:
        return bool(self.phase) and ("tool" in self.phase.lower()
                                     or self.phase in ("web_search", "search_result",
                                                       "bash", "code_interpreter",
                                                       "fetch_page", "generate_image",
                                                       "edit_file", "read_file",
                                                       "write_file", "present_file"))

    def __str__(self) -> str:  # pragma: no cover - convenience only
        if self.type == "created":
            return f"<created response_id={self.response_id}>"
        bits = [p for p in (self.phase, self.status) if p]
        head = "/".join(bits)
        tail = self.content if self.content is not None else ""
        return f"<{head or self.type}> {tail}"


def iter_sse_frames(lines: Iterator[str]) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON objects from an SSE line iterator.

    The Qwen backend emits one JSON object per ``data:`` line (no blank-line
    separators), so each ``data:`` line is parsed immediately. If a line is
    not valid JSON on its own it is *buffered* and re-attempted joined with
    following lines, which transparently supports spec-style multi-line
    frames. Unparseable garbage is eventually discarded (bounded buffer) so
    a single poison frame cannot wedge the parser. A buffer left non-empty
    at end-of-stream means the stream ended mid-frame.
    """
    buf: List[str] = []
    for line in lines:
        if line is None:
            continue
        if line.startswith(":"):  # comment / keep-alive
            continue
        if line.startswith("data:"):
            buf.append(line[5:].lstrip())
            if len(buf) > 4:  # poison guard: drop the oldest chunk
                buf.pop(0)
            try:
                yield json.loads("\n".join(buf))
                buf = []
            except json.JSONDecodeError:
                pass  # keep buffering (partial or multi-line frame)
            continue
        # any other line (e.g. "event: x") is ignored
    if buf:
        raise StreamInterruptedError("SSE stream ended mid-frame")


def parse_event(ev: Dict[str, Any]) -> ChatEvent:
    """Normalise one raw SSE object into a :class:`ChatEvent`."""
    if "response.created" in ev:
        rc = ev["response.created"] or {}
        return ChatEvent(type="created", response_id=rc.get("response_id"),
                         raw=ev)

    choices = ev.get("choices") or []
    if choices:
        delta = (choices[0] or {}).get("delta") or {}
        return ChatEvent(
            type="delta",
            phase=delta.get("phase"),
            status=delta.get("status"),
            content=delta.get("content"),
            role=delta.get("role"),
            extra=delta.get("extra") or {},
            usage=ev.get("usage"),
            response_id=ev.get("response_id"),
            raw=ev,
        )
    return ChatEvent(type="raw", raw=ev)


@dataclass
class StreamResult:
    """Aggregated outcome of one streamed completion.

    Attributes:
        answer: concatenated ``answer`` phase text.
        phases: ordered unique ``phase/status`` transitions, first entry is
            always ``created``.
        response_id: the server's response message id for this turn.
        thinking: concatenated ``thinking_summary`` content, if any.
        tool_calls: collected ``extra.local_mcp`` descriptors (client-side
            tool invocations requested by the model).
        tool_events: every delta belonging to a tool phase, in order.
        events: all parsed events, only when ``keep_events=True``.
    """

    answer: str = ""
    phases: List[str] = field(default_factory=list)
    response_id: Optional[str] = None
    thinking: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_events: List[ChatEvent] = field(default_factory=list)
    events: Optional[List[ChatEvent]] = None

    @property
    def finished(self) -> bool:
        return bool(self.phases) and self.phases[-1].endswith(FINISHED)


def consume(lines: Iterator[str], *, keep_events: bool = False) -> StreamResult:
    """Drain an SSE stream into a :class:`StreamResult`."""
    res = StreamResult(events=[] if keep_events else None)
    last_tag: Optional[str] = None

    for ev in iter_sse_frames(lines):
        e = parse_event(ev)
        if keep_events and res.events is not None:
            res.events.append(e)
        if e.type == "created":
            res.response_id = e.response_id
            last_tag = "created"
            res.phases.append(last_tag)
            continue
        if e.type != "delta":
            continue

        tag = f"{e.phase}/{e.status}" if (e.phase or e.status) else None
        if tag and tag != last_tag:
            res.phases.append(tag)
            last_tag = tag

        if e.phase == "answer" and e.content:
            res.answer += e.content
        elif e.phase == "thinking_summary" and e.content:
            res.thinking += e.content
        elif e.extra.get("local_mcp"):
            res.tool_calls.append(e.extra["local_mcp"])

        if e.is_tool_phase or e.extra.get("local_mcp") or e.extra.get("tool_result"):
            res.tool_events.append(e)

    return res
