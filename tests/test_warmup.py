"""Tests for the headless-browser warmup path and env-var fallback.

These tests are network-free; the headless-browser backends (Playwright /
agent-browser) are stubbed out so we can pin the wiring without running a
real browser. Live behaviour of the warmup itself was verified manually
during the fix (see docs/anti-bot.md for the experiment record).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from qwen_studio import QwenStudio, warmup  # noqa: E402
from qwen_studio.exceptions import AuthError  # noqa: E402
from qwen_studio.warmup import WarmupError  # noqa: E402


# ----------------------------------------------------------------- backends
def test_has_playwright_returns_bool():
    assert isinstance(warmup.has_playwright(), bool)


def test_has_agent_browser_returns_bool():
    assert isinstance(warmup.has_agent_browser(), bool)


def test_warmup_no_token_raises():
    with pytest.raises(WarmupError, match="session token"):
        warmup.warmup_cookie_jar("")


def test_warmup_unknown_backend_raises():
    with pytest.raises(WarmupError, match="unknown warmup backend"):
        warmup.warmup_cookie_jar("tok", backend="selenium")


def test_warmup_no_backend_available(monkeypatch):
    monkeypatch.setattr(warmup, "has_playwright", lambda: False)
    monkeypatch.setattr(warmup, "has_agent_browser", lambda: False)
    with pytest.raises(WarmupError, match="no warmup backend available"):
        warmup.warmup_cookie_jar("tok")


# ------------------------------------------------------ playwright path mocked
class _FakeResp:
    def __init__(self, set_cookie_lines=None):
        self.headers = _FakeHeaders(set_cookie_lines or [])


class _FakeHeaders:
    def __init__(self, lines):
        self._lines = lines

    def get_list(self, name):
        if name.lower() == "set-cookie":
            return self._lines
        return []

    def get(self, name, default=""):
        if name.lower() == "set-cookie":
            return self._lines[0] if self._lines else default
        return default


def test_absorb_response_cookies_extracts_set_cookie():
    q = QwenStudio.from_access_token("fake-token-for-test")
    q.extra_cookies.clear()
    r = _FakeResp([
        "acw_tc=ACW1; Path=/; Domain=chat.qwen.ai; HttpOnly",
        "x-ap=ap-southeast-1; Path=/; SameSite=lax",
        "token=should-not-be-absorbed; Path=/; Domain=.qwen.ai",
        "cna=CNA1; Path=/; Domain=.qwen.ai",
    ])
    q._absorb_response_cookies(r)
    # token must NOT appear in extra_cookies (it's tracked separately)
    assert "token" not in q.extra_cookies
    assert q.extra_cookies["acw_tc"] == "ACW1"
    assert q.extra_cookies["x-ap"] == "ap-southeast-1"
    assert q.extra_cookies["cna"] == "CNA1"


def test_absorb_response_cookies_handles_no_headers():
    q = QwenStudio.from_access_token("fake-token-for-test")
    q.extra_cookies.clear()
    # response with no Set-Cookie at all should be a no-op
    r = _FakeResp([])
    q._absorb_response_cookies(r)
    assert q.extra_cookies == {}


def test_absorb_response_cookies_malformed_lines_skipped():
    q = QwenStudio.from_access_token("fake-token-for-test")
    q.extra_cookies.clear()
    r = _FakeResp([
        "no-equals-sign-here",
        "=empty-name",
        "good=val; Path=/",
    ])
    q._absorb_response_cookies(r)
    assert q.extra_cookies == {"good": "val"}


# -------------------------------------------------------- warmup wiring
def test_do_warmup_merges_jarr_and_skips_token(monkeypatch):
    """do_warmup() must merge the captured jar into extra_cookies."""
    q = QwenStudio.from_access_token("fake-token-for-test")
    q.session_token = "session-xyz"
    captured = {
        "token": "session-xyz",   # same token: should NOT overwrite
        "cna": "CNA-FROM-WARMUP",
        "tfstk": "TFK-FROM-WARMUP",
        "isg": "ISG-FROM-WARMUP",
        "ssxmod_itna": "SSX1",
        "ssxmod_itna2": "SSX2",
    }
    monkeypatch.setattr(
        "qwen_studio.warmup.warmup_cookie_jar",
        lambda *a, **k: captured,
    )
    out = q.do_warmup()
    assert out["cna"] == "CNA-FROM-WARMUP"
    assert q.extra_cookies["tfstk"] == "TFK-FROM-WARMUP"
    assert q.extra_cookies["isg"] == "ISG-FROM-WARMUP"
    assert q.extra_cookies["ssxmod_itna"] == "SSX1"
    assert q.extra_cookies["ssxmod_itna2"] == "SSX2"
    # token stays the same (it was unchanged by warmup)
    assert q.session_token == "session-xyz"
    assert q._warmed_up is True


def test_do_warmup_respects_rotation_of_session_token(monkeypatch):
    """If the SPA's load rotates the session token, do_warmup must honour it."""
    q = QwenStudio.from_access_token("fake-token-for-test")
    q.session_token = "old-token"
    captured = {
        "token": "rotated-new-token",
        "cna": "CNA",
    }
    monkeypatch.setattr(
        "qwen_studio.warmup.warmup_cookie_jar",
        lambda *a, **k: captured,
    )
    q.do_warmup()
    assert q.session_token == "rotated-new-token"
    assert q.extra_cookies["cna"] == "CNA"


def test_do_warmup_idempotent_unless_force(monkeypatch):
    """A second do_warmup() call must be a no-op unless force=True."""
    q = QwenStudio.from_access_token("fake-token-for-test")
    q.session_token = "tok"
    calls = []

    def fake_warmup(token, **kw):
        calls.append(token)
        return {"cna": f"CNA-{len(calls)}"}

    monkeypatch.setattr(
        "qwen_studio.warmup.warmup_cookie_jar",
        fake_warmup,
    )
    q.do_warmup()
    q.do_warmup()
    assert len(calls) == 1, "second do_warmup() must be a no-op"
    q.do_warmup(force=True)
    assert len(calls) == 2


def test_do_warmup_without_session_token_raises():
    q = QwenStudio.from_access_token("fake-token-for-test")
    q.session_token = None
    with pytest.raises(AuthError, match="session token"):
        q.do_warmup()


# ---------------------------------------------------------- from_credentials
def test_from_credentials_invokes_warmup(monkeypatch):
    """from_credentials(warmup=True) must call signin + do_warmup."""
    calls = {"signin": 0, "warmup": 0}

    def fake_signin(self):
        calls["signin"] += 1
        self.session_token = "signed-in-token"
        return {}

    def fake_warmup(self, **kw):
        calls["warmup"] += 1
        self._warmed_up = True
        self.extra_cookies["cna"] = "CNA-WARMUP"
        return dict(self.extra_cookies)

    monkeypatch.setattr(QwenStudio, "signin", fake_signin)
    monkeypatch.setattr(QwenStudio, "do_warmup", fake_warmup)
    q = QwenStudio.from_credentials("e@x.com", "pw", warmup=True)
    assert calls == {"signin": 1, "warmup": 1}
    assert q.session_token == "signed-in-token"
    assert q.extra_cookies["cna"] == "CNA-WARMUP"


def test_from_credentials_without_warmup_does_not_call_warmup(monkeypatch):
    calls = {"signin": 0, "warmup": 0}

    def fake_signin(self):
        calls["signin"] += 1
        self.session_token = "tok"
        return {}

    def fake_warmup(self, **kw):
        calls["warmup"] += 1
        return {}

    monkeypatch.setattr(QwenStudio, "signin", fake_signin)
    monkeypatch.setattr(QwenStudio, "do_warmup", fake_warmup)
    QwenStudio.from_credentials("e@x.com", "pw", warmup=False)
    assert calls == {"signin": 1, "warmup": 0}


# ------------------------------------------------------------- env fallback
def test_from_browser_falls_back_to_env_vars(monkeypatch, fake_home):
    """from_browser() with no profile + env vars set must use credentials + warmup."""
    from qwen_studio import browser_cookies as bc

    # ensure no browser profiles exist (fake_home is empty)
    monkeypatch.setenv("QWEN_EMAIL", "env@user.com")
    monkeypatch.setenv("QWEN_PASSWORD", "env-pw-123")

    # capture the credentials + warmup flag that from_browser passes on
    captured = {}

    def fake_from_credentials(cls, email, password, *, warmup=False,
                              warmup_backend="auto", **kw):
        captured["email"] = email
        captured["password"] = password
        captured["warmup"] = warmup
        # return a real client so the constructor path completes
        c = cls(access_token="env-fallback-token",
                warmup=warmup, warmup_backend=warmup_backend, **kw)
        c.email = email
        c._password = password
        c.cookie_source = f"env-credentials (warmup={warmup})"
        return c

    monkeypatch.setattr(QwenStudio, "from_credentials",
                        classmethod(fake_from_credentials))
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        q = QwenStudio.from_browser(auto_refresh=False)
    assert captured["email"] == "env@user.com"
    assert captured["password"] == "env-pw-123"
    assert captured["warmup"] is True, (
        "env-var fallback must force warmup=True (otherwise the resulting "
        "session would have a thin anti-bot jar and get punished)")


def test_from_browser_env_fallback_disabled_raises(monkeypatch, fake_home):
    """fallback_to_env=False must propagate the original BrowserCookieError."""
    monkeypatch.setenv("QWEN_EMAIL", "env@user.com")
    monkeypatch.setenv("QWEN_PASSWORD", "env-pw-123")
    with pytest.raises(Exception, match="no supported browser"):
        QwenStudio.from_browser(fallback_to_env=False, auto_refresh=False)


def test_from_browser_without_env_vars_raises(monkeypatch, fake_home):
    """No browser profile + no env vars = same BrowserCookieError as before."""
    # don't set QWEN_EMAIL / QWEN_PASSWORD
    monkeypatch.delenv("QWEN_EMAIL", raising=False)
    monkeypatch.delenv("QWEN_PASSWORD", raising=False)
    with pytest.raises(Exception, match="no supported browser"):
        QwenStudio.from_browser(auto_refresh=False)


# ------------------------------------------------------------------ CLI
def test_cli_login_supports_env_vars(monkeypatch, fake_home, tmp_path, capsys):
    """qwen-studio login must accept QWEN_EMAIL/QWEN_PASSWORD env vars."""
    monkeypatch.setenv("QWEN_EMAIL", "env@user.com")
    monkeypatch.setenv("QWEN_PASSWORD", "env-pw-123")

    captured = {}

    def fake_from_credentials(cls, email, password, *, warmup=False,
                              warmup_backend="auto", **kw):
        captured["email"] = email
        captured["password"] = password
        captured["warmup"] = warmup
        c = cls(access_token="t", warmup=warmup,
                warmup_backend=warmup_backend, **kw)
        c.email = email
        c._password = password
        c.session_token = "session-from-env-signin"
        c.extra_cookies["cna"] = "CNA-WARMUP"
        c.extra_cookies["tfstk"] = "TFK-WARMUP"
        c.cookie_source = "env-credentials"
        return c

    monkeypatch.setattr(QwenStudio, "from_credentials",
                        classmethod(fake_from_credentials))

    from qwen_studio.cli import cmd_login
    import argparse
    args = argparse.Namespace(
        browser=None, profile=None, domain="qwen.ai",
        email=None, password=None, save_password=False,
        file=str(tmp_path / "auth.json"), min_interval=4.0,
        warmup=False, warmup_backend="auto",
    )
    rc = cmd_login(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "auth file written" in out
    assert "env@user.com" == captured["email"]
    assert captured["warmup"] is True, (
        "no-browser-profile path must auto-enable warmup")
    auth = json.loads((tmp_path / "auth.json").read_text())
    assert auth["session_token"] == "session-from-env-signin"
    assert auth["extra_cookies"]["cna"] == "CNA-WARMUP"
    assert auth["warmup"] is True


def test_cli_login_warmup_flag_passes_through(monkeypatch, fake_home, tmp_path):
    """--warmup on the CLI must be honoured even when a browser profile exists."""
    monkeypatch.delenv("QWEN_EMAIL", raising=False)
    monkeypatch.delenv("QWEN_PASSWORD", raising=False)

    captured = {}

    def fake_from_credentials(cls, email, password, *, warmup=False,
                              warmup_backend="auto", **kw):
        captured["warmup"] = warmup
        c = cls(access_token="t", warmup=warmup,
                warmup_backend=warmup_backend, **kw)
        c.email, c._password = email, password
        c.session_token = "tok"
        c.cookie_source = "credentials"
        return c

    monkeypatch.setattr(QwenStudio, "from_credentials",
                        classmethod(fake_from_credentials))

    from qwen_studio.cli import cmd_login
    import argparse
    args = argparse.Namespace(
        browser=None, profile=None, domain="qwen.ai",
        email="cli@user.com", password="cli-pw", save_password=False,
        file=str(tmp_path / "auth.json"), min_interval=4.0,
        warmup=True, warmup_backend="playwright",
    )
    rc = cmd_login(args)
    assert rc == 0
    assert captured["warmup"] is True


# --------------------------------------------------------------- fixtures
@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Isolated HOME with no browser profiles."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path
