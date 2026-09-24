"""Custom tools for chat: wrapping Python callables into the MCP protocol.

This module implements the *client-side* MCP mechanism (``local_mcp``) - in
practice the desktop app's way of running its own tool servers - so that a
plain Python function becomes a chat-callable tool:

1. **Declaration** - tool schemas ride inline in the user message's
   ``feature_config.local_mcp``, shaped exactly like registry entries::

       {"ServerName": {"tool_name": {"description": ...,
                                     "input_schema": {...}}}}

2. **Invocation** - the model emits tool calls into the stream as deltas
   with ``phase = local_tool`` and ``extra.local_mcp`` describing the call
   (``{"ServerName": [{"tool_name": ..., "params": {...}}]}``).

3. **Execution** - the *client* runs the tool. This library executes your
   registered Python callables and serialises results exactly the way the
   official client does: ``{"ServerName": [{"tool_name": "<json string>"}]}``.

4. **Continuation** - the results go back as a follow-up completion whose
   single message has ``role: "function"`` and the stringified results as
   content; the feature config keeps its ``mcp``/``local_mcp`` keys stripped
   and the top-level ``parentId`` points at the tool-emitting response.

Live-verification status (see docs/local-tools.md): declaration and
invocation are confirmed end-to-end from an external client. The
continuation is *accepted only from recognised client sessions* - the web
edge punishes it with the anti-bot page and the desktop surface answers
``Bad_Request``. The full loop is implemented faithfully and raises
:class:`~qwen_studio.exceptions.ContinuationBlockedError` with the server's
exact reply if that gate is hit, so the code is future-proof and the
boundary is explicit rather than hidden.
"""

from __future__ import annotations

import inspect
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .client import QwenStudio
from .chat import DEFAULT_FEATURE_CONFIG
from .exceptions import (BadRequestError, ContinuationBlockedError,
                         PunishedError, StreamInterruptedError,
                         ToolExecutionError)
from .sse import StreamResult

_JSON_TYPES = {str: "string", int: "integer", float: "number",
               bool: "boolean", dict: "object", list: "array"}


def _schema_from_callable(fn: Callable[..., Any]) -> Dict[str, Any]:
    """Derive a JSON schema from a function's signature and type hints."""
    sig = inspect.signature(fn)
    props: Dict[str, Any] = {}
    required: List[str] = []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ann = p.annotation if p.annotation is not inspect._empty else str
        props[name] = {"type": _JSON_TYPES.get(ann, "string")}
        if p.default is inspect._empty:
            required.append(name)
        else:
            props[name]["description"] = f"default: {p.default!r}"
    doc = inspect.getdoc(fn) or fn.__name__.replace("_", " ")
    return {"type": "object", "properties": props,
            "required": required, "description": doc}


@dataclass
class Tool:
    """A locally executed chat tool."""

    name: str
    fn: Callable[..., Any]
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_callable(cls, fn: Callable[..., Any], name: Optional[str] = None,
                      description: Optional[str] = None,
                      input_schema: Optional[Dict[str, Any]] = None) -> "Tool":
        return cls(name=name or fn.__name__, fn=fn,
                   description=description or "",
                   input_schema=input_schema or _schema_from_callable(fn))

    def declaration(self) -> Dict[str, Any]:
        """The registry-entry shape used inside ``feature_config.local_mcp``."""
        d: Dict[str, Any] = {"description": self.description or
                             (self.input_schema.get("description") or self.name),
                             "input_schema": self.input_schema}
        return d

    def execute(self, args: Dict[str, Any]) -> str:
        try:
            return json.dumps(self.fn(**args), ensure_ascii=False,
                              default=str)
        except Exception as e:  # noqa: BLE001 - convert everything to payload
            raise ToolExecutionError(f"tool {self.name!r} failed: {e}") from e


def tool(fn: Callable[..., Any] = None, *, name: str = None,  # type: ignore[assignment]
         description: str = None, input_schema: Dict[str, Any] = None):
    """Decorator: turn a function into a :class:`Tool` (schema auto-derived)."""
    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool.from_callable(f, name=name, description=description,
                                  input_schema=input_schema)
    return wrap(fn) if fn is not None else wrap


class LocalToolSession:
    """Runs the client-side MCP loop for one chat.

    Example::

        from qwen_studio import QwenStudio

        q = QwenStudio.from_credentials(EMAIL, PASSWORD)
        chat = q.chats.create(model)

        session = q.local_tools(chat.id)
        session.register("get_weather", get_weather)   # any python callable

        turn = session.ask("What is the weather in Shanghai?")
        print(turn.text)
    """

    def __init__(self, client: QwenStudio, chat_id: str,
                 server_name: str = "LocalTools") -> None:
        self.client = client
        self.chat_id = chat_id
        self.server_name = server_name
        self._tools: Dict[str, Tool] = {}
        self._parent_id: Optional[str] = None

    # ------------------------------------------------------------ registry
    def register(self, fn_or_name: Any, fn: Optional[Callable[..., Any]] = None,
                 **kw: Any) -> Tool:
        """Register a tool from a callable or an existing :class:`Tool`."""
        if isinstance(fn_or_name, Tool):
            t = fn_or_name
        else:
            t = Tool.from_callable(fn, name=fn_or_name, **kw)
        self._tools[t.name] = t
        return t

    def tool(self, fn: Callable[..., Any] = None, **kw: Any):  # noqa: D102
        """Decorator form of :meth:`register`."""
        def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
            self.register(f, **kw)
            return f
        return wrap(fn) if fn is not None else wrap

    def declaration(self) -> Dict[str, Any]:
        """``feature_config.local_mcp`` payload for all registered tools."""
        if not self._tools:
            return {}
        return {self.server_name: {t.name: t.declaration()
                                   for t in self._tools.values()}}

    # ---------------------------------------------------------------- step1
    def send(self, prompt: str, model: str, *, thinking: bool = False,
             keep_events: bool = False) -> StreamResult:
        """Send the user turn with the local tool declaration attached."""
        fc = dict(DEFAULT_FEATURE_CONFIG)
        if thinking:
            fc.update(thinking_enabled=True, auto_thinking=True,
                      thinking_mode="Enable")
        fc["local_mcp"] = self.declaration()
        msg = {
            "id": None, "fid": str(uuid.uuid4()),
            "parentId": self._parent_id,
            "childrenIds": [str(uuid.uuid4())], "role": "user",
            "content": prompt, "user_action": "chat", "files": [],
            "timestamp": int(time.time()), "models": [model], "model": "",
            "chat_type": "t2t", "feature_config": fc,
            "extra": {"meta": {"subChatType": "t2t"}},
            "sub_chat_type": "t2t", "parent_id": self._parent_id,
        }
        body = {
            "stream": True, "version": "2.1", "incremental_output": True,
            "chatId": self.chat_id, "parentId": self._parent_id or "",
            "chat_id": self.chat_id, "chat_mode": "normal", "model": model,
            "parent_id": self._parent_id,
            "messages": [msg], "timestamp": int(time.time()),
        }
        result = self.client.stream_completion(body, self.chat_id,
                                               keep_events=keep_events)
        self._parent_id = getattr(result, "response_id", None) or self._parent_id
        return result

    # ------------------------------------------------------------- execute
    def execute_calls(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
        """Execute every requested call; returns the exact client shape.

        Shape (from the official client's serialisation):
        ``{"ServerName": [{"tool_name": "<json-stringified result>"}]}`` with
        failures recorded as the string ``"Tool call failed: ..."``.
        """
        results: Dict[str, List[Dict[str, str]]] = {self.server_name: []}
        for call in tool_calls:
            for srv, entries in call.items():
                bucket = results.setdefault(srv, [])
                if not isinstance(entries, list):
                    entries = [entries]
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    name = e.get("tool_name") or next(iter(e), "")
                    params = e.get("params") or {}
                    t = self._tools.get(name)
                    if t is None:
                        bucket.append({name: f"Tool call failed: unknown tool {name!r}"})
                        continue
                    try:
                        bucket.append({name: t.execute(params)})
                    except ToolExecutionError as te:
                        bucket.append({name: f"Tool call failed: {te}"})
        return results

    # -------------------------------------------------------------- step2
    def send_tool_results(self, model: str, response_id: str,
                          results: Dict[str, List[Dict[str, str]]],
                          *, source: Optional[str] = None) -> StreamResult:
        """Post the ``role:"function"`` continuation with executed results.

        The message replicates the official client's serialisation: a clone
        of the response turn with ``id``/``parentId``/``content_list``/
        ``user_action`` stripped, ``mcp``/``local_mcp`` removed from the
        feature config, and the results JSON string as content.
        """
        content = json.dumps(results, ensure_ascii=False)
        fc = {k: v for k, v in DEFAULT_FEATURE_CONFIG.items()
              if k not in ("mcp", "local_mcp")}
        fn_msg = {
            # This is a NEW message-tree node. Reusing the tool-emitting
            # response fid makes Qwen treat the continuation as an edit of
            # that node, which can hide the original user turn from context.
            "fid": str(uuid.uuid4()), "childrenIds": [], "role": "function",
            "content": content, "files": [], "timestamp": int(time.time()),
            "models": [model], "model": "", "chat_type": "t2t",
            "feature_config": fc,
            "extra": {"meta": {"subChatType": "t2t"}, "mcp_results": results},
            "sub_chat_type": "t2t",
            "parent_id": response_id, "parentId": response_id,
        }
        body = {
            "stream": True, "version": "2.1", "incremental_output": True,
            "chatId": self.chat_id, "parentId": response_id,
            "chat_id": self.chat_id, "chat_mode": "normal", "model": model,
            "parent_id": response_id, "messages": [fn_msg],
            "timestamp": int(time.time()),
        }
        try:
            result = self.client.stream_completion(body, self.chat_id)
            self._parent_id = getattr(result, "response_id", None) or self._parent_id
            return result
        except PunishedError as e:
            raise ContinuationBlockedError(
                "tool-result continuation hit the anti-bot risk engine "
                "(web edge). The tool call succeeded; results are attached.",
                body=e.body, results=results) from e
        except BadRequestError as e:
            raise ContinuationBlockedError(
                f"server rejected the continuation shape "
                f"({e.code}: {e.details}). The tool call succeeded; results "
                f"are attached. See docs/local-tools.md.",
                body=json.dumps(e.payload or {}), results=results) from e

    # ---------------------------------------------------------------- loop
    def ask(self, prompt: str, model: str, *, max_loops: int = 3,
            thinking: bool = False, keep_events: bool = False) -> StreamResult:
        """Full loop: send -> execute tool calls -> continue -> return answer.

        If the continuation is blocked by the server-side gate, the
        :class:`ContinuationBlockedError` propagates (the tool executions
        still happened and are attached to the exception).
        """
        res = self.send(prompt, model, thinking=thinking, keep_events=keep_events)
        loops = 0
        while res.tool_calls and loops < max_loops:
            loops += 1
            if not res.response_id:
                raise StreamInterruptedError("tool turn finished without response_id")
            results = self.execute_calls(res.tool_calls)
            res = self.send_tool_results(model, res.response_id, results)
        return res
