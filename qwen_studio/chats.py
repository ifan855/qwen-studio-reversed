"""Chat lifecycle: create, list, read, delete, batch-delete, archive, pin.

Verified live against the service (see docs/chats.md):

- ``POST /chats/new`` creates the chat record; the server assigns the id
  (the ``chatId`` field in the request body is ignored / left empty).
- ``DELETE /chats/{id}`` -> ``{"success": true, "data": {"status": true}}``;
  a subsequent ``GET /chats/{id}`` returns ``Not_Found``.
- ``POST /chats/batch_delete`` with ``{"ids": [...]}`` deletes many at once.
- ``GET /chats/?page=N&exclude_project=true`` lists history pages.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .client import BASE, QwenStudio

T2T = "t2t"  # text-to-text; the client also uses t2i, t2v, i2i, ...


@dataclass
class Chat:
    """A chat record as returned by the server."""

    id: str
    title: Optional[str] = None
    models: List[str] = field(default_factory=list)
    chat_type: Optional[str] = None
    updated_at: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "Chat":
        keep = ("id", "title", "models", "chat_type", "updated_at")
        return cls(**{k: d.get(k) for k in keep if k in d},  # type: ignore[arg-type]
                   raw=d)


@dataclass
class ChatMessage:
    """One message node from the stored conversation tree."""

    role: str
    content: str = ""
    id: Optional[str] = None
    fid: Optional[str] = None
    parent_id: Optional[str] = None
    phase: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class ChatService:
    """Bound to a :class:`~qwen_studio.client.QwenStudio` instance."""

    def __init__(self, client: QwenStudio) -> None:
        self.client = client

    # ---------------------------------------------------------------- create
    def create(self, model: str, *, chat_type: str = T2T,
               chat_mode: str = "normal", project_id: str = "") -> Chat:
        """Create a new chat and return it with its server-assigned id.

        The body mirrors the web client: millisecond timestamp, empty
        ``chatId`` (the server assigns the real uuid), the target model, the
        chat type (``t2t`` for text chat) and the mode.
        """
        d = self.client.request(
            "POST", "/chats/new",
            json_body={"chatId": "", "models": [model], "project_id": project_id,
                       "timestamp": int(time.time() * 1000),
                       "chat_type": chat_type, "chat_mode": chat_mode})
        data = d.get("data") or {}
        chat_id = data.get("id") or ""
        return Chat(id=chat_id, models=[model], chat_type=chat_type, raw=data)

    # ----------------------------------------------------------------- read
    def list(self, page: int = 1, *, exclude_project: bool = True) -> List[Chat]:
        """List history chats, newest page first."""
        d = self.client.request(
            "GET", "/chats/",
            params={"page": page, "exclude_project": str(exclude_project).lower()})
        data = d.get("data")
        items: List[Dict[str, Any]] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("chats") or data.get("items") or []
        return [Chat.from_api(x) for x in items if isinstance(x, dict)]

    def get(self, chat_id: str) -> Dict[str, Any]:
        """Full chat record including the message tree."""
        d = self.client.request("GET", f"/chats/{chat_id}")
        data = d.get("data") or {}
        return data.get("chat") if isinstance(data.get("chat"), dict) else data

    def messages(self, chat_id: str) -> List[ChatMessage]:
        """Convenience: the chat's message list as :class:`ChatMessage`."""
        chat = self.get(chat_id)
        out: List[ChatMessage] = []
        for m in chat.get("messages") or []:
            fc = m.get("feature_config") or {}
            out.append(ChatMessage(
                role=m.get("role") or "", content=m.get("content") or "",
                id=m.get("id"), fid=m.get("fid"), parent_id=m.get("parent_id"),
                phase=fc.get("phase_selected"), raw=m))
        return out

    # --------------------------------------------------------------- delete
    def delete(self, chat_id: str) -> bool:
        """Delete one chat. Verified: returns True, subsequent GET 404s."""
        d = self.client.request("DELETE", f"/chats/{chat_id}")
        data = d.get("data") or {}
        return bool(data.get("status", True))

    def delete_many(self, chat_ids: List[str]) -> bool:
        """Batch delete via ``POST /chats/batch_delete``."""
        d = self.client.request("POST", "/chats/batch_delete",
                                json_body={"ids": list(chat_ids)})
        return bool((d.get("data") or {}).get("status", True))

    # -------------------------------------------------------------- extras
    def rename(self, chat_id: str, title: str) -> Dict[str, Any]:
        return self.client.request("POST", f"/chats/{chat_id}/title",
                                   json_body={"title": title}).get("data") or {}

    def pin(self, chat_id: str) -> Dict[str, Any]:
        return self.client.request("POST", f"/chats/{chat_id}/pin").get("data") or {}

    def archive(self, chat_id: str) -> Dict[str, Any]:
        return self.client.request("POST", f"/chats/{chat_id}/archive").get("data") or {}

    def pinned(self) -> List[Chat]:
        d = self.client.request("GET", "/chats/pinned")
        data = d.get("data")
        items = data if isinstance(data, list) else (data or {}).get("chats", [])
        return [Chat.from_api(x) for x in items if isinstance(x, dict)]

    # --------------------------------------------------------------- import
    @staticmethod
    def build_import_record(chat_id: str, title: str, messages: List[Dict[str, Any]],
                            user_id: str, *, model: str = "",
                            created_at: Optional[int] = None) -> Dict[str, Any]:
        """Build one import record in the schema the server validates
        (reverse-engineered via its ValidationError messages):

        ``[{id, user_id, title, chat: {id, title, models, chat_type,
        messages: [...], history: {messages: {fid: node}, currentId}},
        created_at, updated_at, archived, pinned, folder_id, ...}]``

        Message nodes need real ``id``/``fid`` values wired into a
        parent/children chain (the last node's fid is ``history.currentId``).
        See :meth:`import_history` for the live-verified boundary.
        """
        import json as _json

        def _clone(m: Dict[str, Any]) -> Dict[str, Any]:
            # strip non-serialisable / oversized material defensively
            return _json.loads(_json.dumps(m, default=str))

        msgs = [_clone(m) for m in messages]
        by_fid = {m["fid"]: m for m in msgs if m.get("fid")}
        chat_obj = {
            "id": chat_id,
            "title": title,
            "models": [model] if model else [],
            "chat_type": "t2t",
            "messages": msgs,
            "history": {"messages": by_fid,
                        "currentId": msgs[-1]["fid"] if msgs else None,
                        "currentResponseIds": [], "pagination": {}},
        }
        ts = created_at or int(time.time())
        return {"id": chat_id, "user_id": user_id, "title": title,
                "chat": chat_obj, "created_at": ts, "updated_at": ts,
                "archived": False, "pinned": False, "folder_id": None,
                "group_ids": [], "tags": []}

    def import_history(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """``POST /chats/import`` with fabricated/prepared history records.

        Live-verified boundary (docs/capabilities.md): the endpoint *accepts*
        fully fabricated, never-happened histories (``data.success`` lists
        server-assigned chat ids, and ``GET /chats/{id}`` showed every
        fabricated message persisted for a short window) - but the server
        then flags the imported chat as deleted almost immediately
        (``CHAT_NOT_FOUND: This chat has been deleted``), so imports cannot
        currently serve as a stable base for continuations. Kept for
        completeness and backup-restore use.
        """
        import json as _json

        blob = _json.dumps(records).encode()
        boundary = f"----qwenstudio{uuid.uuid4().hex}"
        filename = "import.json"
        body = ((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="{filename}"\r\nContent-Type: application/json\r\n\r\n'
                 ).encode() + blob + f"\r\n--{boundary}--\r\n".encode())
        self.client.ensure_access_token()
        self.client._paced_sleep()
        r = self.client.http.post(
            f"{BASE}/chats/import", data=body,
            headers=self.client.headers(
                **{"Content-Type": f"multipart/form-data; boundary={boundary}"}),
            cookies=self.client._cookies(), timeout=self.client.timeout)
        self.client._check_punish(self.client._read_body(r)[:2000])
        d = self.client._json(r)
        self.client._check_app_error(r, d)
        return d.get("data") or {"success": [], "failed": []}

    # -------------------------------------------------------- helper: uuids
    @staticmethod
    def _uuid() -> str:
        return str(uuid.uuid4())
