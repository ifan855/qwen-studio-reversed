"""HTTP-transport behaviour of the OpenAI-compatible proxy.

Covers what clients actually see on the wire: unknown/WebSocket paths, JSON
(vs the stdlib's HTML) error pages, and the fact that a client that simply
goes away produces one log line - not a traceback dump that looks like a
server crash. Everything runs in-process against a stub backend.
"""

import json
import socket
import threading
import time

import pytest

from qwen_studio import exceptions as qe
from qwen_studio.client import QwenStudio, as_transport_error, host_of
from qwen_studio.files import FileRef
from qwen_studio.openai_api import OpenAICompatService
from qwen_studio.openai_server import OpenAIProxyServer, map_exception


# --------------------------------------------------------------------- stub
class MinimalBackend:
    """Just enough backend for the HTTP layer to answer a chat request."""

    def model_ids(self):
        return ["qwen3-max"]

    def resolve_model(self, requested, has_images=False):
        return "qwen3-max"

    def declare_tools(self, tools):
        return {}

    def create_chat(self, model, system_prompt):
        return "chat-1", None

    def upload(self, data, filename, content_type):
        return FileRef(id="f1", url="https://x/y", name=filename)

    def cleanup(self, chat_ids, project_id):
        pass

    def stream_turn(self, chat_id, model, prompt, *, files_entries=None,
                    tools_decl=None, thinking=False):
        yield {"type": "answer", "text": "ok"}
        yield {"type": "done", "response_id": "r1"}


# ------------------------------------------------------------------ harness
@pytest.fixture
def server():
    srv = OpenAIProxyServer(("127.0.0.1", 0),
                            OpenAICompatService(MinimalBackend(), ttl=60.0))
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, kwargs={
        "poll_interval": 0.05}, daemon=True)
    thread.start()
    time.sleep(0.05)
    yield srv, port
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=2)


def raw(port, request: bytes, *, wait: float = 0.15) -> bytes:
    """Send raw bytes, return the raw response (or b'' if the peer vanished)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        s.sendall(request)
        time.sleep(wait)
        s.settimeout(2)
        out = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            out += chunk
    except (socket.timeout, ConnectionResetError):
        pass
    finally:
        s.close()
    return out


def json_body(response: bytes):
    """JSON payload of a response.

    A request the stdlib cannot parse a protocol version from is answered
    HTTP/0.9 style - body only, no status line or headers - so fall back to
    parsing the whole thing.
    """
    payload = (response.split(b"\r\n\r\n", 1)[1]
               if b"\r\n\r\n" in response else response)
    try:
        return json.loads(payload)
    except ValueError:
        return None


def status_line(response: bytes) -> bytes:
    return response.split(b"\r\n", 1)[0]


WS_UPGRADE = (b"GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
              b"Connection: Upgrade\r\n"
              b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
              b"Sec-WebSocket-Version: 13\r\n\r\n")


# ------------------------------------------------------- websocket-ish paths
def test_websocket_upgrade_gets_a_clear_json_answer(server):
    """A client opening /ws is told what this proxy is, in JSON - not
    silently reset, and not left looking like a server crash."""
    srv, port = server
    resp = raw(port, WS_UPGRADE)
    assert b"501" in status_line(resp)
    body = json_body(resp)
    assert body is not None, "error body must be JSON"
    err = body["error"]
    assert err["code"] == "websocket_unsupported"
    assert "WebSocket" in err["message"]
    assert "/v1/chat/completions" in err["message"]   # tells them what works


@pytest.mark.parametrize("path", ["/ws", "/socket.io", "/websocket"])
def test_websocket_paths_never_look_like_a_success(server, path):
    srv, port = server
    resp = raw(port, f"GET {path} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    assert b"501" in status_line(resp)
    assert json_body(resp)["error"]["code"] == "websocket_unsupported"


def test_ws_probe_then_abort_logs_one_line_and_no_traceback(server, capsys):
    """The reported symptom: something opens /ws, drops the connection, and
    socketserver dumps a ConnectionResetError traceback."""
    srv, port = server
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(WS_UPGRADE)
    s.close()                                    # abort without reading
    deadline = time.time() + 3
    while time.time() < deadline:
        if "went away" in capsys.readouterr().err:
            break
        time.sleep(0.05)
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "Exception occurred" not in err


def test_genuine_handler_errors_still_report_fully(server, capsys):
    """Only disconnects are quietened - real bugs must stay loud."""
    srv, _ = server
    try:
        raise ValueError("real bug")
    except ValueError:
        srv.handle_error(None, ("127.0.0.1", 1234))
    assert "Traceback" in capsys.readouterr().err


def test_disconnect_is_reported_in_one_line(server, capsys):
    srv, _ = server
    for exc in (ConnectionResetError(104, "Connection reset by peer"),
                BrokenPipeError(32, "Broken pipe"),
                ConnectionAbortedError(103, "Connection aborted"),
                TimeoutError("timed out")):
        try:
            raise exc
        except type(exc):
            srv.handle_error(None, ("127.0.0.1", 4321))
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert err.count("connection closed") == 4


# ----------------------------------------------------------- other odd paths
def test_unknown_path_lists_the_real_endpoints(server):
    srv, port = server
    resp = raw(port, b"GET /nope HTTP/1.1\r\nHost: x\r\n\r\n")
    assert b"404" in status_line(resp)
    err = json_body(resp)["error"]
    assert err["code"] == "unknown_path"
    assert "/v1/chat/completions" in err["message"]


def test_root_serves_a_browsable_index(server):
    srv, port = server
    for path in ("/", "/v1"):
        resp = raw(port, f"GET {path} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        assert b"200" in status_line(resp)
        body = json_body(resp)
        assert body["service"]
        assert any("POST /v1/chat/completions" == e for e in body["endpoints"])


def test_protocol_errors_are_json_not_html(server):
    """stdlib sends an HTML error page; OpenAI clients parse JSON."""
    srv, port = server
    resp = raw(port, b"this is not http\r\n\r\n")
    assert b"400" in status_line(resp)
    assert b"<!DOCTYPE" not in resp and b"<html" not in resp
    assert json_body(resp)["error"]["code"] == "http_400"


def test_unsupported_method_is_json(server):
    srv, port = server
    resp = raw(port, b"PUT /v1/models HTTP/1.1\r\nHost: x\r\n"
                     b"Content-Length: 0\r\n\r\n")
    assert b"501" in status_line(resp)
    assert b"<html" not in resp
    assert json_body(resp)["error"]["code"] == "http_501"


def test_normal_api_is_unaffected(server):
    srv, port = server
    assert b"200" in status_line(raw(port, b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n"))
    body = json.dumps({"model": "m", "messages": [
        {"role": "user", "content": "hi"}]}).encode()
    resp = raw(port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                     b"Content-Type: application/json\r\nContent-Length: "
                     + str(len(body)).encode() + b"\r\n\r\n" + body)
    assert b"200" in status_line(resp)
    assert json_body(resp)["choices"][0]["message"]["content"] == "ok"


# ----------------------------------------------------- transport error typing
def test_transport_error_maps_to_502_upstream_unavailable():
    status, obj = map_exception(qe.TransportError(
        "cannot reach chat.qwen.ai: Connection closed abruptly",
        target="chat.qwen.ai"))
    assert status == 502
    assert obj["error"]["type"] == "upstream_unavailable"
    assert obj["error"]["code"] == "upstream_unreachable"
    assert "chat.qwen.ai" in obj["error"]["message"]


class _ForeignCurlError(Exception):
    """Stands in for curl_cffi's RequestsError (a foreign type)."""


_ForeignCurlError.__module__ = "curl_cffi.requests.errors"


@pytest.mark.parametrize("exc", [
    _ForeignCurlError("Failed to perform, curl: (35) ... Connection closed abroad"),
    OSError(101, "Network is unreachable"),
    TimeoutError("timed out"),
])
def test_transport_failures_are_rewritten_with_the_host(exc):
    out = as_transport_error(exc, "chat.qwen.ai")
    assert isinstance(out, qe.TransportError)
    assert out.target == "chat.qwen.ai"
    assert "cannot reach chat.qwen.ai" in str(out)
    assert out.details                       # original text kept


def test_non_transport_errors_are_left_alone():
    assert as_transport_error(qe.APIError("app error"), "h") is None
    assert as_transport_error(ValueError("bug"), "h") is None


def test_host_of():
    assert host_of("https://chat.qwen.ai/api/v2") == "chat.qwen.ai"
    assert host_of("https://b.oss.example.com/p/1") == "b.oss.example.com"


class _DeadSession:
    """Session whose every call fails like a blocked network path."""

    def request(self, *a, **k):
        raise _ForeignCurlError("curl: (35) Connection closed abruptly")

    def get(self, *a, **k):
        raise _ForeignCurlError("curl: (35) Connection closed abruptly")

    def post(self, *a, **k):
        raise _ForeignCurlError("curl: (35) Connection closed abruptly")


def test_client_wraps_dead_transport_for_api_calls():
    q = QwenStudio(session_token="tok", session=_DeadSession())
    with pytest.raises(qe.TransportError) as e:
        q.request("GET", "/models/", bearer=False)
    assert e.value.target == "chat.qwen.ai"

    q2 = QwenStudio(access_token="AT", auto_refresh=False,
                    session=_DeadSession())
    with pytest.raises(qe.TransportError) as e2:
        next(q2.open_stream({"chatId": "c"}, "c"))
    assert e2.value.target == "chat.qwen.ai"


def test_client_wraps_dead_transport_for_token_refresh():
    q = QwenStudio(session_token="tok", session=_DeadSession())
    with pytest.raises(qe.TransportError) as e:
        q.refresh()
    assert e.value.target == "auth.qwen.ai"


def test_serve_startup_survives_a_dead_transport(tmp_path, capsys,
                                                 monkeypatch):
    """`serve` must start (and explain itself) when the network is blocked."""
    from qwen_studio import cli
    import qwen_studio.openai_server as server_mod

    class DummyServer:
        def __init__(self, addr, service, api_key=None):
            self.addr = addr

        def serve_forever(self, poll_interval=0.5):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    class DeadClient:
        cookie_source = "test:dead"

        def _cookies(self):
            return {}

        def list_model_ids(self):
            raise qe.TransportError(
                "cannot reach chat.qwen.ai: Connection closed abruptly",
                target="chat.qwen.ai")

    monkeypatch.setattr(server_mod, "OpenAIProxyServer", DummyServer)
    monkeypatch.setattr(cli, "_build_client", lambda args: DeadClient())
    assert cli.cmd_serve(cli.build_parser().parse_args(["serve"])) == 0
    captured = capsys.readouterr()
    assert "cannot reach chat.qwen.ai" in captured.err     # actionable
    assert "OpenAI-compatible API on" in captured.out      # still serving
