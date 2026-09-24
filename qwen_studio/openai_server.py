"""A well-behaved OpenAI-compatible HTTP server over the Qwen proxy layer.

Pure standard library (``http.server``) - no extra dependencies. Endpoints:

- ``POST /v1/chat/completions`` (and ``/chat/completions``) - streaming
  (SSE) and non-streaming, system prompts, ``tools`` (client-side MCP
  wrap), image/file content parts.
- ``GET  /v1/models`` (and ``/models``) - the live Qwen catalogue.
- ``POST /v1/files`` (multipart field ``file``) - in-memory file store;
  reference uploads from later chat turns via
  ``{"type":"file","file":{"file_id": ...}}``.
- ``GET  /v1/files`` / ``GET /v1/files/{id}`` - list / inspect uploads.
- ``GET  /health`` - liveness.

Errors are returned as OpenAI error objects; upstream punish/quota states
map to 503/429 so standard OpenAI SDK retry logic behaves.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from . import exceptions as exc
from .openai_api import OpenAICompatService


def error_object(message: str, type_: str = "api_error",
                 code: Optional[str] = None) -> Dict[str, Any]:
    return {"error": {"message": message, "type": type_, "code": code,
                      "param": None}}


def map_exception(e: Exception) -> Tuple[int, Dict[str, Any]]:
    """Library exceptions -> (http status, OpenAI error object)."""
    if isinstance(e, exc.PunishedError):
        return 503, error_object(
            "upstream anti-bot risk engine intercepted the request; wait a "
            "few minutes and retry", "upstream_unavailable", "punished")
    if isinstance(e, exc.RateLimitedError):
        return 429, error_object(str(e), "rate_limit_error", "rate_limited")
    if isinstance(e, exc.QuotaError):
        return 429, error_object(str(e), "quota_error", "quota_exhausted")
    if isinstance(e, exc.BadRequestError):
        return 400, error_object(str(e), "invalid_request_error",
                                 getattr(e, "code", None))
    if isinstance(e, exc.AuthError):
        return 500, error_object(f"proxy authentication problem: {e}",
                                 "configuration_error", "auth")
    if isinstance(e, exc.QwenStudioError):
        return 502, error_object(str(e), "upstream_error",
                                 getattr(e, "code", None))
    return 500, error_object(f"internal proxy error: {e}", "internal_error")


class _Multipart:
    """Minimal multipart/form-data reader (returns the ``file`` field)."""

    @staticmethod
    def parse(body: bytes, content_type: str) -> Optional[Tuple[str, str, bytes]]:
        marker = "boundary="
        if marker not in content_type:
            return None
        boundary = content_type.split(marker, 1)[1].split(";")[0].strip('"')
        delim = b"--" + boundary.encode()
        for part in body.split(delim):
            part = part.strip(b"\r\n")
            if not part or part == b"--":
                continue
            if b"\r\n\r\n" not in part:
                continue
            head, _, data = part.partition(b"\r\n\r\n")
            headers = head.decode("utf-8", "ignore")
            if 'name="file"' not in headers and "name=file" not in headers:
                continue
            filename, ct = "upload.bin", "application/octet-stream"
            for line in headers.splitlines():
                low = line.lower()
                if low.startswith("content-disposition") and "filename=" in low:
                    seg = line.split("filename=", 1)[1].split(";")[0].strip().strip('"')
                    filename = seg or filename
                elif low.startswith("content-type:"):
                    ct = line.split(":", 1)[1].strip() or ct
            return filename, ct, data.rstrip(b"\r\n")
        return None


class OpenAIHandler(BaseHTTPRequestHandler):
    """One instance per connection; state lives on the server object."""

    protocol_version = "HTTP/1.1"
    server_version = "qwen-openai-proxy"

    # ------------------------------------------------------------ properties
    @property
    def service(self) -> OpenAICompatService:
        return self.server.service  # type: ignore[attr-defined]

    @property
    def api_key(self) -> Optional[str]:
        return getattr(self.server, "api_key", None)

    # ------------------------------------------------------------- plumbing
    def log_message(self, fmt: str, *args: Any) -> None:
        import sys
        sys.stderr.write("[proxy] %s %s\n" % (self.address_string(), fmt % args))

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Authorization, Content-Type, x-api-key")

    def _json_send(self, status: int, payload: Dict[str, Any]) -> None:
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self._cors()
        self.end_headers()
        self.wfile.write(blob)

    def _authorized(self) -> bool:
        if not self.api_key:
            return True
        auth = self.headers.get("Authorization", "") or ""
        key = self.headers.get("x-api-key", "") or ""
        return auth == f"Bearer {self.api_key}" or key == self.api_key

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            d = json.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise exc.BadRequestError(f"invalid JSON body: {e}",
                                      code="invalid_request_error")
        if not isinstance(d, dict):
            raise exc.BadRequestError("body must be a JSON object",
                                      code="invalid_request_error")
        return d

    # --------------------------------------------------------------- routes
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/health":
            return self._json_send(200, {"ok": True,
                                         "service": "qwen-openai-proxy"})
        if not self._authorized():
            return self._json_send(401, error_object("invalid api key",
                                                     "auth_error"))
        try:
            if path in ("/v1/models", "/models"):
                return self._json_send(200, self.service.handle_models())
            if path in ("/v1/files", "/files"):
                data = [self.service.handle_file_get(fid)
                        for fid in list(self.service.file_store)]
                return self._json_send(200, {"object": "list",
                                             "data": [d for d in data if d]})
            if path.startswith(("/v1/files/", "/files/")):
                fid = path.rsplit("/", 1)[-1]
                meta = self.service.handle_file_get(fid)
                if meta is None:
                    return self._json_send(404, error_object(f"no file {fid}",
                                                             "not_found"))
                return self._json_send(200, meta)
        except Exception as e:  # noqa: BLE001
            status, payload = map_exception(e)
            return self._json_send(status, payload)
        return self._json_send(404, error_object(f"unknown path {path}",
                                                 "not_found"))

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"
        if not self._authorized():
            return self._json_send(401, error_object("invalid api key",
                                                     "auth_error"))
        try:
            if path in ("/v1/chat/completions", "/chat/completions"):
                return self._chat_completions()
            if path in ("/v1/files", "/files"):
                return self._upload_file()
        except BrokenPipeError:
            self.close_connection = True
            return
        except Exception as e:  # noqa: BLE001
            status, payload = map_exception(e)
            return self._json_send(status, payload)
        return self._json_send(404, error_object(f"unknown path {path}",
                                                 "not_found"))

    # ------------------------------------------------------------ handlers
    def _upload_file(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        parsed = _Multipart.parse(body, self.headers.get("Content-Type", ""))
        if parsed is None:
            return self._json_send(400, error_object(
                "expected multipart/form-data with a 'file' field",
                "invalid_request_error"))
        filename, ct, data = parsed
        return self._json_send(200,
                               self.service.handle_file_upload(data, filename, ct))

    def _chat_completions(self) -> None:
        body = self._read_json()
        want_stream = bool(body.get("stream"))
        kind, payload = self.service.handle_chat(body, stream=want_stream)
        if kind == "json":
            if want_stream:      # cached reply re-served as SSE
                return self._write_sse(self.service.stream_cached(payload))
            return self._json_send(200, payload)
        return self._write_sse(payload)

    def _write_sse(self, chunks) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        self.close_connection = True
        try:
            for ch in chunks:
                self.wfile.write(
                    f"data: {json.dumps(ch, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
        except BrokenPipeError:
            self.close_connection = True
            return
        except Exception as e:  # noqa: BLE001 - mid-stream failure
            try:
                _, err = map_exception(e)
                self.wfile.write(
                    f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode())
            except Exception:  # noqa: BLE001
                return
        finally:
            # Close the chunk generator explicitly: it owns the service lock,
            # so relying on refcount-driven finalisation risks holding the
            # lock when a client disconnects mid-stream (every later request
            # would then block until the GC happened to run).
            close = getattr(chunks, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001 - teardown must not mask
                    pass
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:  # noqa: BLE001
            pass


class OpenAIProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: Tuple[str, int], service: OpenAICompatService,
                 api_key: Optional[str] = None) -> None:
        self.service = service
        self.api_key = api_key
        super().__init__(addr, OpenAIHandler)


def serve(client, *, host: str = "127.0.0.1", port: int = 8080,
          api_key: Optional[str] = None, ttl: float = 3600.0,
          max_sessions: int = 256, replay_mode: str = "both",
          thinking: bool = False, default_model: Optional[str] = None
          ) -> OpenAIProxyServer:
    """Build backend + service + HTTP server (call ``serve_forever`` after)."""
    from .openai_api import QwenBackend
    backend = QwenBackend(client, default_model=default_model)
    service = OpenAICompatService(backend, ttl=ttl, max_sessions=max_sessions,
                                  replay_mode=replay_mode, thinking=thinking)
    service.start_sweeper()
    return OpenAIProxyServer((host, port), service, api_key=api_key)
