"""Tests for the browser-mimicry layer (status beacons + sidebar refresh).

Captured from the live SPA via agent-browser: the browser fires
``POST /api/v2/users/status`` before AND after every ``/chat/completions``
plus periodic sidebar refreshes. These tests pin the wiring without
hitting the network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from qwen_studio import QwenStudio  # noqa: E402
from qwen_studio.client import SEC_CH_UA, SEC_CH_UA_PLATFORM, BX_V  # noqa: E402


# --------------------------------------------------------------- headers
def test_headers_include_client_hints():
    """Real Chrome sends sec-ch-ua* on every request; the Baxia risk
    engine scores their absence."""
    q = QwenStudio.from_access_token("t")
    h = q.headers()
    assert h["sec-ch-ua"] == SEC_CH_UA
    assert h["sec-ch-ua-mobile"] == "?0"
    assert h["sec-ch-ua-platform"] == SEC_CH_UA_PLATFORM
    assert h["Accept-Language"] == "en-US,en;q=0.9"
    assert h["Sec-Fetch-Dest"] == "empty"
    assert h["Sec-Fetch-Mode"] == "cors"
    assert h["Sec-Fetch-Site"] == "same-origin"


def test_headers_include_baxia_version():
    """The SPA injects bx-v on every XHR via the Baxia SDK; missing it
    is a strong bot signal."""
    q = QwenStudio.from_access_token("t")
    h = q.headers()
    assert h["bx-v"] == BX_V
    assert BX_V == "2.5.37"


# -------------------------------------------------------- status beacon
def test_send_status_beacon_default_on(monkeypatch):
    """mimic_browser=True (default) fires the beacon on every call."""
    q = QwenStudio.from_access_token("t")
    captured = []
    monkeypatch.setattr(
        q.http, "post",
        lambda url, **kw: captured.append((url, kw)) or _FakeResp(200),
    )
    q._send_status_beacon(page_id="//chat.qwen.ai/c/abc")
    assert len(captured) == 1
    url, kw = captured[0]
    assert "/users/status" in url
    payload = kw["json"]["typarms"]
    assert payload["typarm1"] == "web"
    assert payload["typarm3"] == "prod"
    assert payload["typarm4"] == "qwen_chat"
    assert payload["page_id"] == "//chat.qwen.ai/c/abc"
    assert payload["orgid"] == "tongyi"
    assert "spmId" in payload


def test_send_status_beacon_kind_beacon_payload(monkeypatch):
    """kind='beacon' produces the sendBeacon form with a logId + timestamp."""
    q = QwenStudio.from_access_token("t")
    captured = []
    monkeypatch.setattr(
        q.http, "post",
        lambda url, **kw: captured.append(kw) or _FakeResp(200),
    )
    q._send_status_beacon(kind="beacon")
    payload = captured[0]["json"]["typarms"]
    assert "logId" in payload and len(payload["logId"]) >= 16
    assert "timestamp" in payload
    assert payload["serviceName"] == "tongyiLogService"
    assert payload["requestType"] == "sendBeacon"


def test_send_status_beacon_disabled_when_mimic_off(monkeypatch):
    """mimic_browser=False skips the beacon entirely."""
    q = QwenStudio.from_access_token("t", mimic_browser=False)
    called = []
    monkeypatch.setattr(q.http, "post",
                        lambda *a, **kw: called.append(1) or _FakeResp(200))
    q._send_status_beacon()
    assert called == []


def test_send_status_beacon_swallows_errors(monkeypatch):
    """The beacon is decorative; failures must not propagate."""
    q = QwenStudio.from_access_token("t")

    def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(q.http, "post", boom)
    # must not raise
    q._send_status_beacon()


def test_send_status_beacon_includes_user_uuid(monkeypatch):
    """After refresh(), the user UUID from the JWT is used as typarm2."""
    q = QwenStudio.from_access_token("t")
    q._user_uuid = "user-uuid-1234"
    captured = []
    monkeypatch.setattr(
        q.http, "post",
        lambda url, **kw: captured.append(kw) or _FakeResp(200),
    )
    q._send_status_beacon()
    payload = captured[0]["json"]["typarms"]
    assert payload["typarm2"] == "user-uuid-1234"


# ------------------------------------------------------ sidebar refresh
def test_refresh_sidebar_calls_all_endpoints(monkeypatch):
    """The sidebar refresh hits the same set of endpoints the SPA polls."""
    q = QwenStudio.from_access_token("t")
    urls = []
    # both GET and POST go through the same mock
    def fake_req(method, url, **kw):
        urls.append((method, url))
        return _FakeResp(200)
    monkeypatch.setattr(q.http, "get", lambda url, **kw: (urls.append(("GET", url)), _FakeResp(200))[1])
    monkeypatch.setattr(q.http, "post", lambda url, **kw: (urls.append(("POST", url)), _FakeResp(200))[1])
    q._refresh_sidebar(force=True)
    # 10 GET endpoints + 1 POST (customer-service/entry)
    gets = [u for m, u in urls if m == "GET"]
    posts = [u for m, u in urls if m == "POST"]
    assert len(gets) == 10
    assert any("chats/pinned" in u for u in gets)
    assert any("chats/?page=1" in u for u in gets)
    assert any("library/list" in u for u in gets)
    assert any("projects/" in u for u in gets)
    assert any("users/user/settings" in u for u in gets)
    assert any("credits/pricing" in u for u in gets)
    assert any("configs" in u for u in gets)
    assert any("tts/config" in u for u in gets)
    assert any("folders/" in u for u in gets)
    assert len(posts) == 1
    assert "customer-service/entry" in posts[0]


def test_refresh_sidebar_throttled(monkeypatch):
    """Within SIDEBAR_REFRESH_INTERVAL, subsequent calls are no-ops."""
    q = QwenStudio.from_access_token("t")
    calls = []
    monkeypatch.setattr(
        q.http, "get",
        lambda url, **kw: calls.append(url) or _FakeResp(200),
    )
    q._refresh_sidebar(force=True)
    n1 = len(calls)
    q._refresh_sidebar()  # should be throttled
    assert len(calls) == n1


def test_refresh_sidebar_disabled_when_mimic_off(monkeypatch):
    q = QwenStudio.from_access_token("t", mimic_browser=False)
    called = []
    monkeypatch.setattr(q.http, "get",
                        lambda *a, **kw: called.append(1) or _FakeResp(200))
    q._refresh_sidebar(force=True)
    assert called == []


def test_refresh_sidebar_swallows_errors(monkeypatch):
    q = QwenStudio.from_access_token("t")

    def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(q.http, "get", boom)
    q._refresh_sidebar(force=True)  # must not raise


# ------------------------------------------------------- open_stream wiring
def test_open_stream_fires_beacon_before_and_after(monkeypatch):
    """open_stream() should call _send_status_beacon twice (before + after)
    AND _refresh_sidebar(force=True) once after the stream completes."""
    q = QwenStudio.from_access_token("t")
    beacons = []
    refreshes = []
    aplus = []
    monkeypatch.setattr(q, "_send_status_beacon",
                        lambda **kw: beacons.append(kw))
    monkeypatch.setattr(q, "_refresh_sidebar",
                        lambda **kw: refreshes.append(kw))
    monkeypatch.setattr(q, "_send_aplus_beacon",
                        lambda: alus.append(1) if False else aplus.append(1))

    # mock the HTTP POST to return a fake SSE stream
    class _FakeStreamResp:
        def __init__(self):
            self.headers = {"content-type": "text/event-stream"}
            self._lines = [b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n",
                           b"data: [DONE]\n"]

        def iter_lines(self):
            for ln in self._lines:
                yield ln

    monkeypatch.setattr(q.http, "post",
                        lambda *a, **kw: _FakeStreamResp())
    # patch ensure_access_token to avoid the network
    monkeypatch.setattr(q, "ensure_access_token", lambda: "tok")
    # patch _paced_sleep to avoid the wait
    monkeypatch.setattr(q, "_paced_sleep", lambda: None)

    list(q.open_stream({"chatId": "c1"}, "c1"))
    assert len(beacons) == 2, f"expected 2 beacons (before + after), got {len(beacons)}"
    # before-chat beacon is the session kind
    assert beacons[0].get("kind", "session") == "session"
    # after-chat beacon is the sendBeacon kind
    assert beacons[1].get("kind") == "beacon"
    assert len(refreshes) == 1
    assert refreshes[0].get("force") is True
    assert len(aplus) == 1, "aplus beacon should fire once after each chat"


# ------------------------------------------------------ aplus beacon
def test_send_aplus_beacon_default_on(monkeypatch):
    """mimic_browser=True fires the aplus beacon."""
    q = QwenStudio.from_access_token("t")
    captured = []
    monkeypatch.setattr(
        q.http, "post",
        lambda url, **kw: captured.append((url, kw)) or _FakeResp(200),
    )
    q._send_aplus_beacon()
    assert len(captured) == 1
    url, kw = captured[0]
    assert "aplus.qwen.ai" in url
    assert "aes.1.1" in url
    # payload includes the cna cookie value if set
    q.extra_cookies["cna"] = "TEST_CNA_VALUE"
    captured.clear()
    q._send_aplus_beacon()
    payload = captured[0][1]["json"]
    assert "TEST_CNA_VALUE" in payload["gokey"]


def test_send_aplus_beacon_disabled_when_mimic_off(monkeypatch):
    """mimic_browser=False skips the aplus beacon."""
    q = QwenStudio.from_access_token("t", mimic_browser=False)
    called = []
    monkeypatch.setattr(q.http, "post",
                        lambda *a, **kw: called.append(1) or _FakeResp(200))
    q._send_aplus_beacon()
    assert called == []


def test_send_aplus_beacon_swallows_errors(monkeypatch):
    """The aplus beacon is decorative; failures must not propagate."""
    q = QwenStudio.from_access_token("t")

    def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(q.http, "post", boom)
    q._send_aplus_beacon()  # must not raise


# --------------------------------------------------------- JWT extraction
def test_user_uuid_extracted_from_jwt_after_refresh(monkeypatch):
    """refresh() should extract the user id from the access_token JWT."""
    import base64 as _b64
    import json as _json

    q = QwenStudio.from_session_token("session-token-xyz", auto_refresh=False)

    # craft a fake JWT with an id field
    payload = _json.dumps({"id": "user-abc-123", "type": "access_token"})
    payload_b64 = _b64.urlsafe_b64encode(payload.encode()).rstrip(b"=").decode()
    fake_jwt = f"header.{payload_b64}.signature"

    fake_resp = _FakeResp(200, json_body={
        "success": True,
        "data": {"access_token": fake_jwt, "refresh_token": "r"},
    })
    monkeypatch.setattr(q.http, "get", lambda *a, **kw: fake_resp)
    monkeypatch.setattr(q, "_absorb_response_cookies", lambda r: None)

    q.refresh()
    assert q._user_uuid == "user-abc-123"


# --------------------------------------------------------------- helpers
class _FakeResp:
    def __init__(self, status, *, json_body=None, headers=None):
        self.status_code = status
        self._json = json_body or {"success": True, "data": {}}
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        return self._json

    @property
    def text(self):
        import json as _json
        return _json.dumps(self._json)

    def iter_content(self, n=0):
        return iter([])

    def iter_lines(self):
        return iter([])
