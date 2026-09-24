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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .chats import Chat
from .client import QwenStudio


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
    def system_chat(self, model: str, instruction: str, *, name: str = "") -> Any:
        """One-call system prompt: create a project carrying
        ``instruction``, then a chat inside it. Returns the chat.

        Every turn sent to the returned chat (``q.chat.send``) runs under
        the instruction - invisible on the wire, enforced server-side.
        Clean up with ``q.chats.delete`` + ``q.projects.delete``.
        """
        name = name or f"prompt-{instruction[:24].strip()}"
        proj = self.create(name, custom_instruction=instruction)
        chat = self.client.chats.create(model, project_id=proj.id)
        chat.raw["project_id"] = proj.id  # carry for cleanup symmetry
        return chat
