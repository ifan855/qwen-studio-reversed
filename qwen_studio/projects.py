"""Projects: the official system-prompt mechanism (``custom_instruction``).

Findings from the capability study (docs/capabilities.md):

- The completions endpoint does NOT honour a ``role:"system"`` message in
  the request ``messages[]`` (the stream opens and produces no answer), and
  more than one message per request is rejected outright
  (``Bad_Request: Invalid input too many messages``).
- System prompts are **server-side state**: a *project* carries a
  ``custom_instruction``; chats created inside the project
  (``chats.create(model, project_id=...)``) get that instruction applied
  by the backend to every turn. Verified live: the model obeys an
  instruction it can never see in the wire traffic (token-suffix test).

Verified live: ``create`` (with custom_instruction), chat creation inside a
project, ``list_chats``. ``update``/``delete``/``add_chat`` follow the
web client's endpoints but are best-effort (exercised without deep
response inspection during the study).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .chats import Chat
from .client import QwenStudio

_log = logging.getLogger(__name__)


@dataclass
class Project:
    """A project record; ``custom_instruction`` is the system prompt."""

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    custom_instruction: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class ProjectService:
    """Bound to a :class:`~qwen_studio.client.QwenStudio` instance."""

    def __init__(self, client: QwenStudio) -> None:
        self.client = client

    # ---------------------------------------------------------------- create
    def create(self, name: str, *, description: str = "",
               custom_instruction: str = "") -> Project:
        """Create a project. ``custom_instruction`` = system prompt for all
        chats created inside it (verified live with a token-obedience test).
        """
        d = self.client.request(
            "POST", "/projects/",
            json_body={"name": name, "description": description,
                       "custom_instruction": custom_instruction})
        data = d.get("data") or {}
        return Project(id=data.get("id") or "", name=name,
                       description=description,
                       custom_instruction=custom_instruction, raw=data)

    # ----------------------------------------------------------------- read
    def get(self, project_id: str) -> Project:
        d = self.client.request("GET", f"/projects/{project_id}")
        data = d.get("data") or {}
        return Project(id=data.get("id") or project_id,
                       name=data.get("name"),
                       description=data.get("description"),
                       custom_instruction=data.get("custom_instruction"),
                       raw=data)

    # --------------------------------------------------------------- update
    def update(self, project_id: str, **fields: Any) -> Dict[str, Any]:
        """Patch project fields (name / description / custom_instruction)."""
        d = self.client.request("PUT", f"/projects/{project_id}",
                                json_body=fields)
        return d.get("data") or {}

    def set_system_prompt(self, project_id: str, instruction: str) -> Dict[str, Any]:
        """Convenience: change the project's system prompt."""
        return self.update(project_id, custom_instruction=instruction)

    def delete(self, project_id: str) -> Dict[str, Any]:
        d = self.client.request("DELETE", f"/projects/{project_id}")
        return d.get("data") or {}

    # ---------------------------------------------------------------- chats
    def add_chat(self, project_id: str, chat_id: str) -> Dict[str, Any]:
        """Attach an existing chat to the project (endpoint from the web
        client; body shape best-effort)."""
        d = self.client.request("POST", "/projects/add_chat",
                                json_body={"project_id": project_id,
                                           "chat_id": chat_id})
        return d.get("data") or {}

    def list_chats(self, project_id: str, page: int = 1) -> List[Chat]:
        """Chats belonging to a project (``GET /chats/?project_id=...``)."""
        d = self.client.request(
            "GET", "/chats/", params={"project_id": project_id, "page": page})
        data = d.get("data")
        items: List[Dict[str, Any]] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("chats") or data.get("items") or []
        return [Chat.from_api(x) for x in items if isinstance(x, dict)]

    # ------------------------------------------------------------ high level
    def system_chat(self, model: str, instruction: str, *, name: str = "",
                    auto_cleanup: bool = True) -> "SystemChatContext":
        """One-call system prompt: create a project carrying
        ``instruction``, then a chat inside it.

        Returns a :class:`SystemChatContext` that proxies attribute access
        to the underlying :class:`~qwen_studio.chats.Chat` (so ``chat.id``,
        ``chat.models`` etc. work transparently).

        When *auto_cleanup* is ``True`` (the default), the project **and**
        its chat are deleted from the web service when the context exits::

            with q.projects.system_chat("qwen-max", "Reply in haiku") as chat:
                print(q.chat.ask(chat.id, "Hello", "qwen-max"))
            # project + chat gone from chat.qwen.ai

        Set ``auto_cleanup=False`` to keep them (legacy behaviour).
        """
        name = name or f"prompt-{instruction[:24].strip()}"
        proj = self.create(name, custom_instruction=instruction)
        chat = self.client.chats.create(model, project_id=proj.id)
        chat.raw["project_id"] = proj.id  # carry for cleanup symmetry
        return SystemChatContext(
            client=self.client, chat=chat, project_id=proj.id,
            auto_cleanup=auto_cleanup)


class SystemChatContext:
    """Context manager wrapping a one-shot system-prompt chat.

    Proxies attribute access to the underlying
    :class:`~qwen_studio.chats.Chat` so callers can use ``chat.id``,
    ``chat.models``, etc. directly. On ``__exit__`` (when *auto_cleanup*
    is ``True``), deletes the chat and then the project from the service.
    """

    def __init__(self, *, client: QwenStudio, chat: Chat,
                 project_id: str, auto_cleanup: bool = True) -> None:
        self._client = client
        self._chat = chat
        self._project_id = project_id
        self._auto_cleanup = auto_cleanup

    # -- proxy Chat attributes ------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)

    @property
    def chat(self) -> Chat:
        """The underlying :class:`~qwen_studio.chats.Chat`."""
        return self._chat

    @property
    def project_id(self) -> str:
        return self._project_id

    # -- context manager ------------------------------------------------
    def __enter__(self) -> "SystemChatContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self._auto_cleanup:
            return
        # Only delete if the chat remained a true one-shot (<=1 user turn).
        # If the caller sent additional messages the chat is now a cached
        # session and must be preserved.
        try:
            msgs = self._client.chats.messages(self._chat.id)
            user_turns = sum(1 for m in msgs if m.role == "user")
        except Exception:  # noqa: BLE001
            _log.debug("cleanup: could not inspect chat %s; skipping delete",
                       self._chat.id, exc_info=True)
            return
        if user_turns > 1:
            _log.debug("cleanup: chat %s has %d user turns; keeping project %s",
                       self._chat.id, user_turns, self._project_id)
            return
        try:
            self._client.chats.delete(self._chat.id)
        except Exception:  # noqa: BLE001
            _log.debug("cleanup: failed to delete chat %s", self._chat.id,
                       exc_info=True)
        try:
            self._client.projects.delete(self._project_id)
        except Exception:  # noqa: BLE001
            _log.debug("cleanup: failed to delete project %s",
                       self._project_id, exc_info=True)
