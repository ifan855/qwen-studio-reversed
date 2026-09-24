"""Streaming chat completions against ``POST /chat/completions``.

Builds the exact request envelope the web client sends (verified against
captures and confirmed live) and drains the SSE phase machine via
:mod:`qwen_studio.sse`.

Two tool modes are supported at this layer:

- ``mcp_enabled=True`` sets the web marker ``feature_config.local_mcp = {}``.
  The *hosted* servers that then apply are the ones activated per-account via
  :meth:`qwen_studio.tools.MCPService.enable` - the request itself carries no
  tool list, that is the central protocol finding.
- Client-executed custom tools (the ``local_mcp`` wrap) live in
  :mod:`qwen_studio.local_tools`, which uses this module for the transport.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .client import QwenStudio
from .sse import StreamResult

DEFAULT_FEATURE_CONFIG: Dict[str, Any] = {
    "thinking_enabled": False,
    "output_schema": "phase",
    "research_mode": "normal",
    "auto_thinking": False,
    "thinking_mode": "Disable",
}


@dataclass
class Turn:
    """Result of one user turn (convenience wrapper around StreamResult)."""

    chat_id: str
    response_id: Optional[str]
    result: StreamResult

    @property
    def text(self) -> str:
        return self.result.answer

    def __str__(self) -> str:  # pragma: no cover
        return self.text


class ChatCompletion:
    """High-level completion interface bound to a client."""

    def __init__(self, client: QwenStudio) -> None:
        self.client = client

    # -------------------------------------------------------------- envelope
    @staticmethod
    def user_message(content: str, model: str, *, feature_config: Dict[str, Any],
                     fid: Optional[str] = None,
                     parent_id: Optional[str] = None,
                     files: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """The user message node exactly as the web client serialises it.

        ``files`` takes the file-entry dicts produced by
        :meth:`qwen_studio.files.FileRef.entry` (verified live for images -
        see docs/capabilities.md).
        """
        fid = fid or str(uuid.uuid4())
        return {
            "id": None,
            "fid": fid,
            "parentId": parent_id,
            "childrenIds": [str(uuid.uuid4())],
            "role": "user",
            "content": content,
            "user_action": "chat",
            "files": files or [],
            "timestamp": int(time.time()),
            "models": [model],
            "model": "",
            "chat_type": "t2t",
            "feature_config": feature_config,
            "extra": {"meta": {"subChatType": "t2t"}},
            "sub_chat_type": "t2t",
            "parent_id": parent_id,
        }

    @staticmethod
    def build_body(chat_id: str, messages: List[Dict[str, Any]], model: str,
                   *, parent_id: Optional[str] = None,
                   chat_mode: str = "normal") -> Dict[str, Any]:
        """The completion request envelope (stream, version 2.1, incremental)."""
        pid = parent_id or ""
        return {
            "stream": True,
            "version": "2.1",
            "incremental_output": True,
            "chatId": chat_id,
            "parentId": pid,
            "chat_id": chat_id,
            "chat_mode": chat_mode,
            "model": model,
            "parent_id": parent_id,
            "messages": messages,
            "timestamp": int(time.time()),
        }

    # ------------------------------------------------------------------ send
    def send(self, chat_id: str, prompt: str, model: str, *,
             parent_id: Optional[str] = None,
             mcp_enabled: bool = False, thinking: bool = False,
             files: Optional[List[Dict[str, Any]]] = None,
             extra_feature_config: Optional[Dict[str, Any]] = None,
             keep_events: bool = False) -> Turn:
        """Send one user turn and stream the reply.

        Args:
            chat_id: target chat (create one via
                :meth:`qwen_studio.chats.ChatService.create`).
            prompt: user text.
            model: model id from :meth:`QwenStudio.list_model_ids` - ids
                rotate, do not hardcode. Vision inputs need a
                vision-capable model (``omni``/``vl`` in the catalogue).
            parent_id: upstream response/node id this message continues.
                Leave unset for the first turn; the server-side chat tree uses
                this linkage to append a real child turn instead of treating
                the request like an edit.
            mcp_enabled: attach the ``local_mcp: {}`` marker so the backend
                applies this account's *hosted* MCP servers (enable them
                first via :class:`~qwen_studio.tools.MCPService`).
            thinking: enable the thinking phase.
            files: attachment entries from
                :meth:`qwen_studio.files.FileRef.entry` (verified live:
                multiple images in one turn - see docs/capabilities.md).
            extra_feature_config: merged over the default feature config.
        """
        fc = dict(DEFAULT_FEATURE_CONFIG)
        if thinking:
            fc.update(thinking_enabled=True, auto_thinking=True,
                      thinking_mode="Enable")
        if mcp_enabled:
            fc["local_mcp"] = {}  # web marker: hosted MCP mode ON
        if extra_feature_config:
            fc.update(extra_feature_config)

        msg = self.user_message(prompt, model, feature_config=fc,
                                parent_id=parent_id, files=files)
        body = self.build_body(chat_id, [msg], model, parent_id=parent_id)
        result = self.client.stream_completion(body, chat_id, keep_events=keep_events)
        return Turn(chat_id=chat_id, response_id=result.response_id, result=result)

    def ask(self, chat_id: str, prompt: str, model: str, **kw: Any) -> str:
        """One-shot: send a turn, return just the answer text."""
        return self.send(chat_id, prompt, model, **kw).text
