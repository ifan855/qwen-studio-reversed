"""File uploads: STS token exchange + direct Aliyun OSS PUT.

Reverse-engineered from the web client's upload pipeline and verified live
(docs/capabilities.md):

1. ``POST /files/getstsToken`` with ``{filename, filesize (string), filetype}``
   returns STS credentials plus the pre-registered object coordinates:
   ``access_key_id``, ``access_key_secret``, ``security_token``,
   ``bucketname``, ``endpoint``, ``file_id``, ``file_path``, ``file_url``.
2. The client PUTs the bytes straight to
   ``https://{bucketname}.{endpoint}/{file_path}`` with OSS signature V1
   (``Authorization: OSS <key>:<sig>``, ``x-oss-security-token: ...``).
   Implemented here with stdlib HMAC only - no OSS SDK dependency.
3. Messages reference the upload via a *file entry* dict (``files=[...]``
   on :meth:`qwen_studio.chat.ChatCompletion.send`). Verified live: a
   single-image turn and a **seven-image single turn** both worked against
   a vision-capable model - the well-known "5 images" limit is a *client
   UI* cap, not an API one.

Client-side caps from the web bundle (server can override via config):
images 5 x 20 MB, documents 5 x 20 MB, video 1 x 500 MB, audio 1 x 100 MB.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .client import QwenStudio


@dataclass
class FileRef:
    """An uploaded file as the API knows it (``file_id`` + CDN url)."""

    id: str
    url: Optional[str] = None
    name: Optional[str] = None
    content_type: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def entry(self, *, file_class: str = "vision", show_type: str = "image",
              entry_type: str = "image") -> Dict[str, Any]:
        """The message ``files`` entry shape the web client serialises."""
        return {"type": entry_type, "name": self.name or self.id,
                "file_type": self.content_type or "application/octet-stream",
                "showType": show_type, "status": "uploaded",
                "file_class": file_class, "id": self.id, "url": self.url}


class FileService:
    """Bound to a :class:`~qwen_studio.client.QwenStudio` instance."""

    def __init__(self, client: QwenStudio) -> None:
        self.client = client

    # ------------------------------------------------------------- STS token
    def _sts(self, filename: str, size: int, content_type: str) -> Dict[str, Any]:
        d = self.client.request(
            "POST", "/files/getstsToken",
            json_body={"filename": filename, "filesize": str(size),
                       "filetype": content_type},
            **{"Accept-Language": "en-US,en;q=0.9"})
        data = d.get("data") or {}
        missing = [k for k in ("access_key_id", "access_key_secret",
                               "security_token", "bucketname", "endpoint",
                               "file_id", "file_path") if not data.get(k)]
        if missing:
            from . import exceptions as exc
            raise exc.APIError(f"STS response missing fields: {missing}",
                               details=data)
        return data

    # ------------------------------------------------------------- OSS PUT
    @staticmethod
    def _oss_string_to_sign(method: str, content_type: str, date: str,
                            token: str, bucket: str, key: str) -> str:
        """OSS signature V1 StringToSign (canonicalised headers/resource)."""
        return (f"{method}\n\n{content_type}\n{date}\n"
                f"x-oss-security-token:{token}\n/{bucket}/{key}")

    def _oss_put(self, sts: Dict[str, Any], blob: bytes,
                 content_type: str) -> str:
        date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        token = sts["security_token"]
        bucket, endpoint = sts["bucketname"], sts["endpoint"]
        host = f"{bucket}.{endpoint.replace('https://', '').replace('http://', '')}"
        key = sts["file_path"]
        s2s = self._oss_string_to_sign("PUT", content_type, date, token, bucket, key)
        sig = base64.b64encode(hmac.new(sts["access_key_secret"].encode(),
                                        s2s.encode(), hashlib.sha1).digest()).decode()
        from .client import as_transport_error
        from . import exceptions as exc
        try:
            r = self.client.http.put(
                f"https://{host}/{key}", data=blob,
                headers={"Date": date, "Content-Type": content_type,
                         "x-oss-security-token": token,
                         "Authorization": f"OSS {sts['access_key_id']}:{sig}"},
                timeout=self.client.timeout)
        except Exception as e:  # noqa: BLE001 - typed below
            te = as_transport_error(e, host)
            if te is None:
                raise
            raise te from e
        if r.status_code not in (200, 201):
            from . import exceptions as exc
            raise exc.APIError(f"OSS upload failed: HTTP {r.status_code}",
                               status=r.status_code,
                               details=self.client._read_body(r)[:300])
        return sts.get("file_url") or f"https://{host}/{key}"

    # ---------------------------------------------------------------- upload
    def upload(self, data: bytes, filename: Optional[str] = None,
               content_type: str = "application/octet-stream") -> FileRef:
        """Upload bytes: STS exchange then direct OSS PUT. Returns a
        :class:`FileRef` whose ``entry()`` plugs into ``q.chat.send(files=)``.
        """
        filename = filename or f"upload-{uuid.uuid4().hex[:8]}"
        sts = self._sts(filename, len(data), content_type)
        url = self._oss_put(sts, data, content_type)
        return FileRef(id=sts["file_id"], url=url, name=filename,
                       content_type=content_type, raw=sts)

    def upload_image(self, data: bytes, filename: str = "image.png",
                     content_type: str = "image/png") -> FileRef:
        """Image upload (the vision / ``file_class: "vision"`` path)."""
        ref = self.upload(data, filename, content_type)
        ref.raw["file_class"] = "vision"
        return ref
