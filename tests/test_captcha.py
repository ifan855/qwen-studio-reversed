"""Tests for the Baxia slider auto-solver.

Network-free tests: Playwright is stubbed out, the slider drag is
mocked, the slide endpoint response is faked. Live verification was
done manually during development (see docs/anti-bot.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from qwen_studio import QwenStudio, captcha  # noqa: E402
from qwen_studio.exceptions import PunishedError  # noqa: E402


# --------------------------------------------------------- URL extraction
def test_extract_punish_url_from_json_body():
    """The punish response contains a JSON body with a url field."""
    body = ('{"ret":["FAIL_SYS_USER_VALIDATE","RGV587_ERROR::SM::..."],'
            '"data":{"url":"https://chat.qwen.ai:443//api/v2/chat/'
            'completions/_____tmd_____/punish?x5secdata=xg123"}}')
    url = captcha.extract_punish_url(body)
    assert url is not None
    assert "_____tmd_____/punish" in url
    assert "x5secdata=xg123" in url
    # the :443 port should be stripped (causes double-slash issues)
    assert ":443" not in url


def test_extract_punish_url_with_escaped_slashes():
    """Some punish bodies escape forward slashes as \\/."""
    body = ('{"data":{"url":"https:\\/\\/chat.qwen.ai:443\\/api\\/v2\\/punish"}}')
    url = captcha.extract_punish_url(body)
    assert url == "https://chat.qwen.ai/api/v2/punish"


def test_extract_punish_url_returns_none_for_empty():
    assert captcha.extract_punish_url("") is None
    assert captcha.extract_punish_url(None) is None


def test_extract_punish_url_returns_none_for_non_punish():
    body = '{"success": true, "data": {}}'
    assert captcha.extract_punish_url(body) is None


# --------------------------------------------------------- _check_punish
def test_check_punish_extracts_url_into_exception():
    """_check_punish should populate punish_url on the PunishedError."""
    body = ('{"ret":["FAIL_SYS_USER_VALIDATE"],"data":{"url":'
            '"https://chat.qwen.ai:443//api/v2/_____tmd_____/punish?x5secdata=xg123"}}')
    with pytest.raises(PunishedError) as exc_info:
        QwenStudio._check_punish(body)
    assert exc_info.value.punish_url is not None
    assert "_____tmd_____/punish" in exc_info.value.punish_url
    assert ":443" not in exc_info.value.punish_url


def test_check_punish_html_form_no_url():
    """The HTML punish form has no JSON url field; punish_url should be None."""
    body = '<script>window.location.replace("..._____tmd_____/punish?x5secdata=xg...")</script>'
    with pytest.raises(PunishedError) as exc_info:
        QwenStudio._check_punish(body)
    # the HTML form doesn't have a JSON "url" field, so punish_url is None
    assert exc_info.value.punish_url is None


# --------------------------------------------------------- has_solver
def test_has_solver_returns_bool():
    assert isinstance(captcha.has_solver(), bool)


# --------------------------------------------------------- solver wiring
def test_solve_punish_raises_without_playwright(monkeypatch):
    """When Playwright is not installed, solve_punish raises CaptchaSolverError."""
    monkeypatch.setattr(captcha, "has_solver", lambda: False)
    with pytest.raises(captcha.CaptchaSolverError, match="Playwright is not installed"):
        captcha.solve_punish("https://example.com/punish", {})


def test_solve_punish_with_mocked_playwright(monkeypatch):
    """solve_punish should drive Playwright to drag the slider and return cookies."""
    # fake Playwright module
    class FakeMouse:
        def __init__(self):
            self.moves = []
        def move(self, x, y):
            self.moves.append((x, y))
        def down(self): pass
        def up(self): pass

    class FakeSlider:
        def bounding_box(self):
            return {"x": 100, "y": 200, "width": 30, "height": 30}

    class FakePage:
        def __init__(self):
            self.mouse = FakeMouse()
            self._response_handler = None
            self.url = "https://chat.qwen.ai/punish"
            self._responses_sent = False
        def on(self, event, handler):
            if event == "response":
                self._response_handler = handler
        def goto(self, url, **kw):
            self.url = url
            # simulate the slide response firing after navigation
            # (in real life the /slide response fires after the drag completes)
        def query_selector(self, sel):
            return FakeSlider()
        def close(self): pass

    class FakeContext:
        def __init__(self):
            self.page = FakePage()
            self._init_scripts = []
            self._cookies = []
        def add_init_script(self, script):
            self._init_scripts.append(script)
        def add_cookies(self, cookies):
            self._cookies.extend(cookies)
        def new_page(self):
            return self.page
        def cookies(self, url):
            # simulate the x5sec cookie being set after the solve
            return [
                {"name": "x5sec", "value": "test-x5sec-value",
                 "domain": ".qwen.ai", "path": "/"},
                {"name": "token", "value": "session-token",
                 "domain": ".qwen.ai", "path": "/"},
            ]
        def close(self): pass

    class FakeBrowser:
        def __init__(self):
            self._ctx = FakeContext()
        def new_context(self, **kw):
            return self._ctx
        def close(self): pass

    class FakeBrowserType:
        def launch(self, **kw):
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeBrowserType()

    # patch the imports
    fake_pw_module = type(sys)("playwright")
    fake_pw_sync = type(sys)("playwright.sync_api")
    fake_pw_sync.sync_playwright = lambda: _FakeCtxManager(FakePlaywright())
    fake_pw_module.sync_api = fake_pw_sync
    monkeypatch.setitem(sys.modules, "playwright", fake_pw_module)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_pw_sync)
    monkeypatch.setattr(captcha, "has_solver", lambda: True)

    # mock time.sleep to skip the delays AND fire the fake slide response
    original_sleep = captcha.time.sleep
    def fast_sleep(seconds):
        # don't actually sleep, but fire the slide response on first call
        # after navigation
        pass
    monkeypatch.setattr(captcha.time, "sleep", fast_sleep)

    # simulate the /slide endpoint returning code=0 (success) by
    # firing the response handler after the page loads
    class FakeSlideResponse:
        def __init__(self):
            self.url = "https://chat.qwen.ai/_____tmd_____/slide?slidedata=..."
        def json(self):
            return {"result": {"code": 0, "sig": "from bx"}}

    # patch page.goto to fire the slide response after navigation
    original_goto = FakePage.goto
    def goto_with_slide(self, url, **kw):
        original_goto(self, url, **kw)
        # fire the slide response after a short delay
        if self._response_handler:
            self._response_handler(FakeSlideResponse())
    FakePage.goto = goto_with_slide

    # call solve_punish
    result = captcha.solve_punish(
        "https://chat.qwen.ai/punish?x5secdata=test",
        {"token": "session-token"},
        max_retries=1,
    )

    # the solver should return the cookies including x5sec
    assert "x5sec" in result
    assert result["x5sec"] == "test-x5sec-value"
    assert result["token"] == "session-token"


class _FakeCtxManager:
    def __init__(self, pw):
        self._pw = pw
    def __enter__(self):
        return self._pw
    def __exit__(self, *a):
        return False


# --------------------------------------------------------- open_stream auto-solve
def test_open_stream_auto_solves_on_punish(monkeypatch):
    """open_stream should catch PunishedError, call solve_punish, and retry."""
    q = QwenStudio.from_access_token("t")
    # disable mimicry beacons to keep the test focused
    q._mimic_browser = False

    call_count = {"first": 0, "second": 0}

    class _FakePunishResp:
        def __init__(self):
            self.headers = {"content-type": "application/json"}
            self._body = ('{"ret":["FAIL_SYS_USER_VALIDATE"],"data":{"url":'
                          '"https://chat.qwen.ai/punish?x5secdata=xg123"}}')
        @property
        def text(self):
            return self._body
        def iter_content(self, n=0):
            return iter([])

    class _FakeStreamResp:
        def __init__(self):
            self.headers = {"content-type": "text/event-stream"}
            self._lines = [b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
                           b'data: [DONE]\n']
        def iter_lines(self):
            for ln in self._lines:
                yield ln

    def fake_post(url, **kw):
        call_count["first"] += 1
        if call_count["first"] == 1:
            # first call: punish
            return _FakePunishResp()
        # second call (after solve): success stream
        call_count["second"] += 1
        return _FakeStreamResp()

    monkeypatch.setattr(q.http, "post", fake_post)
    monkeypatch.setattr(q, "ensure_access_token", lambda: "tok")
    monkeypatch.setattr(q, "_paced_sleep", lambda: None)

    # mock the captcha solver
    solved = {"called": False}
    def fake_solve(punish_url, cookies, **kw):
        solved["called"] = True
        solved["punish_url"] = punish_url
        return {"x5sec": "solved-x5sec-value"}
    monkeypatch.setattr("qwen_studio.captcha.solve_punish", fake_solve)
    monkeypatch.setattr("qwen_studio.captcha.has_solver", lambda: True)

    # drain the stream
    lines = list(q.open_stream({"chatId": "c1"}, "c1"))
    assert len(lines) >= 1
    assert solved["called"] is True
    assert "x5sec" in q.extra_cookies
    assert q.extra_cookies["x5sec"] == "solved-x5sec-value"
    # the second call should have succeeded
    assert call_count["second"] == 1


def test_open_stream_no_solve_when_disabled(monkeypatch):
    """auto_solve_captcha=False should re-raise PunishedError without solving."""
    q = QwenStudio.from_access_token("t", auto_solve_captcha=False)
    q._mimic_browser = False

    class _FakePunishResp:
        def __init__(self):
            self.headers = {"content-type": "application/json"}
            self._body = ('{"ret":["FAIL_SYS_USER_VALIDATE"],"data":{"url":'
                          '"https://chat.qwen.ai/punish?x5secdata=xg123"}}')
        @property
        def text(self):
            return self._body
        def iter_content(self, n=0):
            return iter([])

    monkeypatch.setattr(q.http, "post", lambda *a, **kw: _FakePunishResp())
    monkeypatch.setattr(q, "ensure_access_token", lambda: "tok")
    monkeypatch.setattr(q, "_paced_sleep", lambda: None)

    with pytest.raises(PunishedError):
        list(q.open_stream({"chatId": "c1"}, "c1"))


def test_open_stream_no_solve_when_already_attempted(monkeypatch):
    """If a retry also punishes, don't solve again - re-raise immediately."""
    q = QwenStudio.from_access_token("t")
    q._mimic_browser = False

    class _FakePunishResp:
        def __init__(self):
            self.headers = {"content-type": "application/json"}
            self._body = ('{"ret":["FAIL_SYS_USER_VALIDATE"],"data":{"url":'
                          '"https://chat.qwen.ai/punish?x5secdata=xg123"}}')
        @property
        def text(self):
            return self._body
        def iter_content(self, n=0):
            return iter([])

    monkeypatch.setattr(q.http, "post", lambda *a, **kw: _FakePunishResp())
    monkeypatch.setattr(q, "ensure_access_token", lambda: "tok")
    monkeypatch.setattr(q, "_paced_sleep", lambda: None)

    solve_calls = []
    def fake_solve(*a, **kw):
        solve_calls.append(1)
        return {"x5sec": "value"}
    monkeypatch.setattr("qwen_studio.captcha.solve_punish", fake_solve)
    monkeypatch.setattr("qwen_studio.captcha.has_solver", lambda: True)

    with pytest.raises(PunishedError):
        list(q.open_stream({"chatId": "c1"}, "c1"))
    # solve should have been called exactly once (for the first punish)
    assert len(solve_calls) == 1


def test_open_stream_no_solve_when_no_punish_url(monkeypatch):
    """If the punish body has no URL (HTML form), don't attempt to solve."""
    q = QwenStudio.from_access_token("t")
    q._mimic_browser = False

    class _FakePunishResp:
        def __init__(self):
            self.headers = {"content-type": "text/html"}
            # HTML punish form - no JSON url field
            self._body = '<script>window.location.replace("..._____tmd_____...")</script>'
        @property
        def text(self):
            return self._body
        def iter_content(self, n=0):
            return iter([])

    monkeypatch.setattr(q.http, "post", lambda *a, **kw: _FakePunishResp())
    monkeypatch.setattr(q, "ensure_access_token", lambda: "tok")
    monkeypatch.setattr(q, "_paced_sleep", lambda: None)

    solve_calls = []
    monkeypatch.setattr("qwen_studio.captcha.solve_punish",
                        lambda *a, **kw: solve_calls.append(1) or {})
    monkeypatch.setattr("qwen_studio.captcha.has_solver", lambda: True)

    with pytest.raises(PunishedError):
        list(q.open_stream({"chatId": "c1"}, "c1"))
    assert len(solve_calls) == 0
