"""Socket-level tests for the OpenAI-compatible HTTP server (no network).

Pins the connection hygiene: websocket probes at ``/ws`` are answered with a
clean reply and a closed connection (no keep-alive read that dies with
``ConnectionResetError``), routine disconnects never print socketserver
tracebacks, and an interrupted first-message stream followed by a resend
serves a real reply instead of a blank one.
"""

import json
import socket
import sys
import threading
import time

import pytest

from qwen_studio.files import FileRef
from qwen_studio.openai_api import OpenAICompatService
from qwen_studio.openai_server import OpenAIProxyServer


class StubBackend:
    def __init__(self):
        self.turns = [[{"type": "answer", "text": "hello there"}],
                      [{"type": "answer", "text": "hello again"}]]
        self.stream_calls = []
        self.created = []
        self.chat_seq = 0

    def model_ids(self):
        return ["qwen3.7-plus"]

    def resolve_model(self, requested, has_images=False):
        return "qwen3.7-plus"

    def declare_tools(self, tools):
        return {}

    def create_project(self, instruction):
        return f"proj-{len(self.created)}"

    def open_chat(self, model, project_id=None):
        self.created.append((model, project_id))
        self.chat_seq += 1
        return f"chat-{self.chat_seq}"

    def set_instruction(self, project_id, instruction):
        pass

    def delete_chat(self, chat_id):
        pass

    def delete_project(self, project_id):
        pass

    def upload(self, data, filename, content_type):
        return FileRef(id="f", url="u", name=filename, content_type=content_type)

    def stream_turn(self, chat_id, model, prompt, **kw):
        self.stream_calls.append(prompt)
        events = self.turns.pop(0) if self.turns else [
            {"type": "answer", "text": "x"}]
        yield from events
        yield {"type": "done", "response_id": "r"}


@pytest.fixture()
def server():
    be = StubBackend()
    svc = OpenAICompatService(be)
    srv = OpenAIProxyServer(("127.0.0.1", 0), svc)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, be
    srv.shutdown()
    srv.server_close()


def _raw(sock, payload: bytes) -> bytes:
    sock.sendall(payload)
    out = b""
    sock.settimeout(2)
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            out += chunk
    except (socket.timeout, ConnectionResetError, BrokenPipeError):
        pass
    return out


def _post_chat(port: int, messages, *, stream=False, abort_after=None):
    body = json.dumps({"model": "m", "stream": stream,
                       "messages": messages}).encode()
    req = (f"POST /v1/chat/completions HTTP/1.1\r\n"
           f"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
           ).encode() + body
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        s.sendall(req)
        if abort_after is not None:     # read a little then hard-drop (RST)
            s.settimeout(5)
            s.recv(abort_after)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                         b"\x01\x00\x00\x00\x00\x00\x00\x00")
            return b""
        out = b""
        s.settimeout(10)
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            out += chunk
        return out
    finally:
        s.close()


def test_ws_probe_is_clean_and_closes(server, capsys):
    srv, _ = server
    port = srv.server_address[1]
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        resp = _raw(s, (b"GET /ws HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                        b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                        b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                        b"Sec-WebSocket-Version: 13\r\n\r\n"))
    finally:
        s.close()
    assert b" 404 " in resp.split(b"\r\n", 1)[0] or b"404" in resp.split(b"\r\n", 1)[0]
    assert b"websocket_not_supported" in resp
    assert b"Connection: close" in resp          # never keep-alive -> no reset
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "Exception occurred during processing" not in err


def test_ws_probe_plain_get(server):
    srv, _ = server
    port = srv.server_address[1]
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        resp = _raw(s, b"GET /ws HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    finally:
        s.close()
    assert b"404" in resp
    assert b"Connection: close" in resp


def test_connection_reset_prints_no_traceback(server, capsys):
    """RST right after a request must not produce socketserver tracebacks."""
    srv, _ = server
    port = srv.server_address[1]
    for _ in range(3):
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(b"GET /ws HTTP/1.1\r\nHost: x\r\n"
                  b"Upgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                     b"\x01\x00\x00\x00\x00\x00\x00\x00")
        s.close()                               # abrupt reset
    time.sleep(0.3)
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "ConnectionResetError" not in err
    assert "Exception occurred during processing" not in err


def test_first_message_interrupted_then_resend_serves_reply(server):
    srv, be = server
    port = srv.server_address[1]
    msgs = [{"role": "system", "content": "S"},
            {"role": "user", "content": "hi"}]
    # first stream: client reads a few bytes then resets the connection
    _post_chat(port, msgs, stream=True, abort_after=16)
    time.sleep(0.2)
    # resend of the exact same first message -> a real reply, never blank
    resp = _post_chat(port, msgs, stream=True)
    head, _, tail = resp.partition(b"\r\n\r\n")
    assert b"200" in head.split(b"\r\n", 1)[0]
    assert b"hello again" in tail or b"hello there" in tail
    assert be.stream_calls                      # the turn actually ran


def test_chat_completions_non_stream(server):
    srv, _ = server
    port = srv.server_address[1]
    resp = _post_chat(port, [{"role": "user", "content": "hi"}], stream=False)
    head, _, body = resp.partition(b"\r\n\r\n")
    assert b"200" in head.split(b"\r\n", 1)[0]
    data = json.loads(body)
    assert data["choices"][0]["message"]["content"] == "hello there"
