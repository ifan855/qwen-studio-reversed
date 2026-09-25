"""HTTP layer: hosts, header contract, token lifecycle, request plumbing.

Reverse-engineered from the chat.qwen.ai web client (v0.3.11) and verified
against the live service. See docs/authentication.md for the narrative.

Hosts
-----
- ``chat.qwen.ai``  - the SPA + the entire ``/api/v2`` application interface.
- ``auth.qwen.ai``  - dedicated auth service owning the refresh exchange.

Auth chain
----------
``POST /api/v2/auths/signin`` with the SHA-256 hex digest of the password
returns a 30-day *session token*. ``GET https://auth.qwen.ai/api/v2/auths/refresh``
with cookie ``token=<session>`` returns a short-lived (900 s) ``access_token``
plus a rotated 30-day ``refresh_token``. All application endpoints take
``Authorization: Bearer <access_token>``; one known exception is
``GET /mcp/list`` which authenticates via the session *cookie* only.

Header contract
---------------
The client sends ``source`` (web/h5/desktop), ``version``, a JS-style
``timezone`` string and a UUID ``x-request-id`` with every call. Live probing
showed none of these are validated for ordinary authenticated requests, but
they are reproduced here for fidelity (and because the edge risk engine may
score requests that look nothing like the official client).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
import warnings
from typing import Any, Dict, Iterator, Optional

try:  # optional fallback transport; curl_cffi is the supported default
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:  # Chrome TLS/HTTP2 fingerprint transport (the supported default)
    from curl_cffi import requests as crequests
except ImportError:  # pragma: no cover
    crequests = None

from . import exceptions as exc
from .sse import consume, StreamResult

WEB_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

#: Chrome Client-Hints that real Chrome sends on every request. curl_cffi's
#: ``impersonate="chrome"`` handles the TLS/HTTP2 fingerprint but does NOT
#: auto-emit these headers — the Baxia risk engine scores their absence.
SEC_CH_UA = ('"Chromium";v="131", "Not_A Brand";v="24", '
             '"Google Chrome";v="131"')
SEC_CH_UA_MOBILE = "?0"
SEC_CH_UA_PLATFORM = '"Windows"'

#: Baxia SDK version header — Alibaba's anti-bot SDK on the SPA injects
#: this on every XHR. Missing it is a strong bot signal. The version
#: captured from the live SPA at the time of writing was 2.5.37.
BX_V = "2.5.37"

BASE = "https://chat.qwen.ai/api/v2"
AUTH_BASE = "https://auth.qwen.ai/api/v2"

PUNISH_MARKERS = ("RGV587", "_____tmd_____", "FAIL_SYS_USER_VALIDATE",
                  "x5secdata", "aliyun_waf_aa", "sessionStorage.x5referer")


def make_http_session(impersonate: Optional[str] = "chrome",
                      min_interval: float = 4.0, jitter: float = 1.5):
    """Build the HTTP session used for all calls.

    When ``impersonate`` is set (default ``"chrome"``) the session presents
    a real Chrome TLS/HTTP2 fingerprint - so script traffic is born with the
    same transport signature as the browser instead of acquiring one after
    being punished. This is the default and the supported configuration:
    curl_cffi is a hard dependency. ``impersonate=None`` opts out (plain
    ``requests`` if installed) and emits a ``RuntimeWarning`` - un-
    fingerprinted traffic is exactly what the risk engine punishes.

    ``min_interval``/``jitter`` shape request cadence: every call through
    :meth:`QwenStudio._paced_sleep` waits so that consecutive requests are
    spaced out like a human, which is the behaviour dimension of the same
    risk score.
    """
    if impersonate:
        if crequests is None:
            raise ImportError(
                f"impersonate={impersonate!r} requires curl_cffi - install "
                "it with `pip install curl_cffi`. Pass impersonate=None to "
                "opt out, but un-fingerprinted traffic is what the anti-bot "
                "engine punishes on /chat/completions.")
        sess = crequests.Session(impersonate=impersonate)
        sess._is_curl_cffi = True
    elif requests is not None:
        warnings.warn(
            "impersonate=None sends plain python-requests traffic (no "
            "browser fingerprint); this is the profile the risk engine "
            "punishes on /chat/completions", RuntimeWarning, stacklevel=2)
        sess = requests.Session()
        sess._is_curl_cffi = False
    elif crequests is not None:
        sess = crequests.Session()
        sess._is_curl_cffi = True
    else:
        raise ImportError("no HTTP transport: install curl_cffi (or requests)")
    sess._min_interval = min_interval
    sess._jitter = jitter
    sess._last_request_ts = 0.0
    return sess


def iter_lines_compat(resp) -> Iterator[str]:
    """Decode ``iter_lines`` output for requests *and* curl_cffi."""
    for ln in resp.iter_lines():
        if isinstance(ln, bytes):
            yield ln.decode("utf-8", "ignore")
        elif ln is not None:
            yield ln


def js_timezone() -> str:
    """The JS ``Date().toString()`` style timezone header the client sends."""
    return time.strftime("%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)",
                         time.gmtime())


def _looks_punished(text: str) -> bool:
    return any(m in text for m in PUNISH_MARKERS)


class QwenStudio:
    """Authenticated client for the Qwen Studio web API.

    Construct with one of:

    - :meth:`from_browser` - **zero-setup**: pulls the complete cookie jar
      (session token + anti-bot set) straight from a local Firefox /
      Chrome / Chromium / Brave / Edge profile on Linux (recommended).
    - :meth:`from_credentials` - email + password (signs in, full lifecycle).
    - :meth:`from_session_token` - an existing 30-day session token (cookie).
    - :meth:`from_access_token` - a live 15-minute access token (bearer-only,
      no renewal possible).

    Sub-services: :attr:`chats` (create/list/delete/history), :attr:`tools`
    (hosted MCP registry + per-user activation), :attr:`chat` (streaming
    completions incl. MCP-enabled turns).
    """

    def __init__(self, *, session_token: Optional[str] = None,
                 access_token: Optional[str] = None,
                 refresh_token: Optional[str] = None,
                 email: Optional[str] = None, password: Optional[str] = None,
                 source: str = "web", version: str = "0.3.11",
                 timeout: float = 180.0, session: Optional[Any] = None,
                 auto_refresh: bool = True,
                 impersonate: Optional[str] = "chrome",
                 min_interval: float = 4.0, jitter: float = 1.5,
                 extra_cookies: Optional[Dict[str, str]] = None,
                 warmup: bool = False,
                 warmup_backend: str = "auto",
                 mimic_browser: bool = True,
                 auto_solve_captcha: bool = True) -> None:
        self.cookie_source = "explicit"
        if not (session_token or access_token or (email and password)):
            raise exc.AuthError(
                "provide credentials, a session token, or an access token")
        self.email = email
        self._password = password
        self.session_token = session_token
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.source = source
        self.version = version
        self.timeout = timeout
        self.auto_refresh = auto_refresh
        self.impersonate = impersonate
        self.extra_cookies = dict(extra_cookies or {})
        self.http = session or make_http_session(
            impersonate, min_interval=min_interval, jitter=jitter)
        self._access_expires_at = 0.0
        self._bootstrapped = False
        # warmup configuration - remembered so ensure_access_token() can
        # re-run it after an automatic re-sign-in (see _do_warmup).
        self._warmup_enabled = bool(warmup)
        self._warmup_backend = warmup_backend
        self._warmed_up = False
        # browser-mimicry configuration: when on (default), the client fires
        # the same /api/v2/users/status telemetry beacons and sidebar
        # refreshes a real browser makes around each /chat/completions
        # call. The Baxia risk engine scores request PATTERN, not just
        # cookies - a clean "just /chats/new + /chat/completions" sequence
        # is the strongest bot signal after cookie absence. See
        # docs/anti-bot.md for the experiment that proved this.
        self._mimic_browser = bool(mimic_browser)
        self._last_sidebar_refresh = 0.0
        self._user_uuid = None  # populated from /auths/ refresh response
        # captcha auto-solve: when on (default), a PunishedError raised
        # from open_stream() triggers qwen_studio.captcha.solve_punish
        # which launches a headless browser to drag the Baxia slider
        # and obtain the x5sec cookie. The cookie is merged into
        # extra_cookies and the stream is retried once.
        self._auto_solve_captcha = bool(auto_solve_captcha)
        self._captcha_solve_attempted = False  # guard against retry loops
        self._wire_services()

    def _paced_sleep(self) -> None:
        """Space out consecutive requests (behavioural risk dimension)."""
        last = getattr(self.http, "_last_request_ts", 0.0)
        min_iv = getattr(self.http, "_min_interval", 0.0)
        jitter = getattr(self.http, "_jitter", 0.0)
        wait = min_iv + (uuid.uuid4().int % 1000) / 1000.0 * jitter - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        self.http._last_request_ts = time.time()

    def _wire_services(self) -> None:
        """Attach the sub-services (lazy imports to avoid cycles)."""
        from .chats import ChatService
        from .chat import ChatCompletion
        from .tools import MCPService
        from .local_tools import LocalToolSession
        from .projects import ProjectService
        from .files import FileService

        self.chats = ChatService(self)
        self.chat = ChatCompletion(self)
        self.tools = MCPService(self)
        self.projects = ProjectService(self)
        self.files = FileService(self)

        def local_tools(chat_id: str, server_name: str = "LocalTools") -> LocalToolSession:
            """Open a client-side tool session for an existing chat id."""
            return LocalToolSession(self, chat_id, server_name=server_name)

        self.local_tools = local_tools

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_credentials(cls, email: str, password: str, *,
                         warmup: bool = True,
                         warmup_backend: str = "auto",
                         **kw: Any) -> "QwenStudio":
        """Sign in with email + plaintext password (hashed in transit).

        ``warmup=True`` (default since v0.5.0) launches a headless
        browser (Playwright if installed, otherwise the ``agent-browser``
        CLI) to let the SPA mint the complete anti-bot cookie jar
        (``cna``, ``tfstk``, ``isg``, ``ssxmod_itna*``, ...). Without
        the warmup, the session carries only the four cookies
        ``/auths/signin`` itself returns (``token``, ``acw_tc``,
        ``x-ap``, ``refresh_token``); those are enough for low-volume
        use but the risk engine eventually punishes bare-jar traffic on
        ``/chat/completions`` - see docs/anti-bot.md.

        The warmup is now default-on because:
        1. Playwright is a soft dependency (``pip install
           qwen-studio[warmup]``); if it's not installed, the constructor
           falls back to the thin-jar path with a ``RuntimeWarning``.
        2. The browser-mimicry layer (``mimic_browser=True``) extends
           the session's survival, but the warmup is still the right
           default - it gives every session a browser-equivalent cookie
           jar from the first request.
        3. The auto-captcha-solver (``auto_solve_captcha=True``) catches
           any punish that slips through and lifts it via the Baxia
           slider solver, so even a thin-jar session can recover.
        """
        c = cls(email=email, password=password,
                warmup=warmup, warmup_backend=warmup_backend, **kw)
        c.signin()
        if warmup:
            try:
                c.do_warmup(backend=warmup_backend)
            except Exception as e:  # noqa: BLE001 - warn, don't crash
                import warnings as _w
                _w.warn(
                    f"warmup failed ({e}); the session is usable but "
                    f"the anti-bot cookie jar may be thin - install "
                    f"Playwright with `pip install qwen-studio[warmup] "
                    f"&& playwright install chromium` for the full jar",
                    RuntimeWarning, stacklevel=2)
        return c

    @classmethod
    def from_session_token(cls, token: str, *,
                          warmup: bool = True,
                          warmup_backend: str = "auto",
                          **kw: Any) -> "QwenStudio":
        """Reuse a 30-day session token (e.g. exported from a browser cookie).

        When ``warmup=True`` (default), runs a headless browser to mint
        the full anti-bot jar against this token - useful when the
        token was obtained from a non-browser source (a previous
        ``from_credentials`` sign-in, a stored auth file, an
        env-var-supplied token, ...).
        """
        c = cls(session_token=token,
                warmup=warmup, warmup_backend=warmup_backend, **kw)
        if kw.get("auto_refresh", True):
            try:
                c.refresh()
            except Exception:  # noqa: BLE001 - stay cookie-capable
                pass
        if warmup:
            try:
                c.do_warmup(backend=warmup_backend)
            except Exception as e:  # noqa: BLE001 - warn, don't crash
                import warnings as _w
                _w.warn(
                    f"warmup failed ({e}); the session is usable but may be "
                    f"punished on /chat/completions; see docs/anti-bot.md",
                    RuntimeWarning, stacklevel=2)
        return c

    @classmethod
    def from_access_token(cls, token: str, **kw: Any) -> "QwenStudio":
        """Use a live access token; no renewal material, it will simply expire."""
        return cls(access_token=token, auto_refresh=False, **kw)

    @classmethod
    def from_browser(cls, browser: Optional[str] = None,
                     profile: Optional[str] = None,
                     domain: str = "qwen.ai",
                     *,
                     fallback_to_env: bool = True,
                     warmup: bool = False,
                     warmup_backend: str = "auto",
                     **kw: Any) -> "QwenStudio":
        """Build the client from a local browser profile's cookies (Linux).

        Reads the *complete* cookie jar - the session ``token`` plus the
        anti-bot set (``acw_tc``, ``tfstk``, ``isg``, ``ssxmod_itna*``,
        ``cna``, ...) - straight from Firefox (``cookies.sqlite``) or a
        Chrome-family profile (decrypted with the browser's own keys), so
        the session presents exactly the material the real browser earned.
        Combined with the default Chrome-impersonated transport this makes
        script traffic browser-equivalent from the first request::

            q = QwenStudio.from_browser()              # auto-detect
            q = QwenStudio.from_browser("firefox")     # or chrome/chromium

        The browser must be logged in to chat.qwen.ai. When several
        profiles qualify, the most recently used one holding a ``token``
        cookie wins (see :mod:`qwen_studio.browser_cookies`).
        ``auto_refresh`` defaults on; a failed initial refresh still leaves
        the client cookie-capable.

        When no local browser profile is found (headless servers,
        containers, CI) and ``fallback_to_env=True`` (default), the
        constructor falls back to :meth:`from_credentials` using the
        ``QWEN_EMAIL`` / ``QWEN_PASSWORD`` environment variables, with
        ``warmup=True`` so the missing anti-bot jar is minted via a
        headless browser. Without the env vars set, raises
        :class:`BrowserCookieError` as before.
        """
        from . import browser_cookies as bc
        try:
            prof, cookies = bc.find_qwen_jar(browser, profile, domain=domain)
        except bc.BrowserCookieError:
            if not fallback_to_env:
                raise
            email = os.environ.get("QWEN_EMAIL")
            password = os.environ.get("QWEN_PASSWORD")
            if not (email and password):
                raise
            # log the fallback so the user knows why they're not seeing
            # a browser-profile cookie source
            import warnings as _w
            _w.warn(
                "no local browser profile found; falling back to "
                "QWEN_EMAIL/QWEN_PASSWORD env vars with warmup=True - "
                "the headless browser will mint the anti-bot cookie jar "
                "that from_browser() would normally read from disk",
                RuntimeWarning, stacklevel=2)
            # warmup is mandatory on the env fallback path - without it the
            # session would carry only the four signin cookies and get
            # punished on /chat/completions (see docs/anti-bot.md)
            return cls.from_credentials(
                email, password,
                warmup=True, warmup_backend=warmup_backend, **kw)

        jar = {c.name: c.value for c in cookies}
        token = jar.pop("token", None)
        if not token:
            raise exc.AuthError(
                f"no session token for {domain!r} in {prof.browser}:"
                f"{prof.name}; log in to chat.qwen.ai in that browser first")
        c = cls(session_token=token, extra_cookies=jar,
                warmup=warmup, warmup_backend=warmup_backend, **kw)
        c.cookie_source = f"{prof.browser}:{prof.name} ({prof.cookie_db})"
        if kw.get("auto_refresh", True):
            try:
                c.refresh()
            except Exception:  # noqa: BLE001 - stay cookie-capable
                pass
        return c

    @staticmethod
    def cookies_from_browser_state(path: str, *, domain: str = "qwen") -> Dict[str, str]:
        """Extract a flat cookie dict from a Playwright storage-state JSON.

        The anti-bot cookie set the browser earned (acw_tc, tfstk, isg,
        ssxmod_itna*, ...) is what the edge risk engine expects alongside
        the auth token - see docs/anti-bot.md. Prefer :meth:`from_browser`,
        which reads the same jar directly from a local browser profile
        without any export step; use this when you already hold a
        Playwright state file.
        """
        state = json.load(open(path))
        return {c["name"]: c["value"] for c in state.get("cookies", [])
                if domain in c.get("domain", "")}

    # ------------------------------------------------------------------ headers
    def headers(self, *, bearer: bool = True, **extra: Any) -> Dict[str, str]:
        h = {
            "User-Agent": WEB_UA,
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": "https://chat.qwen.ai",
            "Referer": "https://chat.qwen.ai/",
            # Chrome Client-Hints — real Chrome sends these on every
            # request; curl_cffi's chrome impersonation handles TLS/HTTP2
            # but does NOT auto-emit Client-Hints, and the Baxia risk
            # engine scores their absence.
            "sec-ch-ua": SEC_CH_UA,
            "sec-ch-ua-mobile": SEC_CH_UA_MOBILE,
            "sec-ch-ua-platform": SEC_CH_UA_PLATFORM,
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            # Baxia SDK version — the SPA injects this header on every XHR
            # via its tracker SDK. Missing it is a strong "this is not the
            # browser" signal to the risk engine.
            "bx-v": BX_V,
            "source": self.source,
            "version": self.version,
            "timezone": js_timezone(),
            "x-request-id": str(uuid.uuid4()),
        }
        if bearer and self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        h.update(extra)
        return h

    def _cookies(self) -> Dict[str, str]:
        jar: Dict[str, str] = {}
        if self.session_token:
            jar["token"] = self.session_token
        # the anti-bot cookie set (acw_tc, tfstk, isg, ssxmod_itna*, ...) is
        # the decisive session material on /chat/completions - see
        # docs/anti-bot.md. Merge user-supplied jar last so it can also
        # provide/override the session token cookie.
        jar.update(self.extra_cookies)
        return jar

    # -------------------------------------------------------------- token cycle
    def _absorb_response_cookies(self, r: Any) -> None:
        """Pull Set-Cookie values from a response into ``extra_cookies``.

        ``curl_cffi`` stores them in the session jar, but the jar is not
        serialised when the auth file is written - so callers that reload
        from disk (e.g. ``qwen-studio serve``) would lose them. Merging
        into ``extra_cookies`` makes them persistent across restarts.
        """
        try:
            # curl_cffi exposes a headers.get_list / .get_all
            if hasattr(r.headers, "get_list"):
                lines = r.headers.get_list("Set-Cookie")
            elif hasattr(r.headers, "get_all"):
                lines = r.headers.get_all("Set-Cookie") or []
            else:
                raw = r.headers.get("set-cookie", "")
                lines = [raw] if raw else []
        except Exception:  # noqa: BLE001
            lines = []
        for ln in lines:
            # parse 'name=value; Path=/; Domain=.qwen.ai; ...'
            head = ln.split(";", 1)[0].strip()
            if "=" not in head:
                continue
            k, v = head.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and k != "token":  # token is tracked separately
                self.extra_cookies[k] = v

    def signin(self) -> Dict[str, Any]:
        """Password sign-in -> 30-day session token.

        Also absorbs the ``Set-Cookie`` headers from the response into
        :attr:`extra_cookies` (``acw_tc``, ``x-ap``, ``cna`` if the edge
        mints one here, ...). These plus the session token form the
        *minimal* cookie jar; :meth:`do_warmup` extends it to the full
        risk-engine set when a headless browser is available.
        """
        if not (self.email and self._password):
            raise exc.AuthError("signin requires email and password")
        pw = hashlib.sha256(self._password.encode()).hexdigest()
        r = self.http.post(f"{BASE}/auths/signin",
                           json={"email": self.email, "password": pw},
                           headers=self.headers(bearer=False),
                           timeout=self.timeout)
        d = self._json(r)
        if not d.get("success"):
            raise exc.AuthError("signin rejected", status=r.status_code, payload=d)
        data = d.get("data") or {}
        self.session_token = data.get("token") or self.session_token
        self._absorb_response_cookies(r)
        self._bootstrapped = True
        return data

    def do_warmup(self, *, backend: str = "auto",
                  timeout_ms: int = 30_000,
                  force: bool = False) -> Dict[str, str]:
        """Run a headless-browser warmup and merge the jar into the client.

        Idempotent unless ``force=True``: re-runs only matter after a fresh
        sign-in (which ``ensure_access_token`` triggers when the 30-day
        session expires). Returns the captured cookie dict.

        Raises :class:`qwen_studio.warmup.WarmupError` when no backend is
        available or the SPA fails to boot. The caller should treat this
        as a soft-failure (the session is still usable for low-volume work).
        """
        if self._warmed_up and not force:
            return dict(self.extra_cookies)
        if not self.session_token:
            raise exc.AuthError(
                "warmup requires a session token; call signin() first")
        from .warmup import warmup_cookie_jar
        jar = warmup_cookie_jar(self.session_token,
                                backend=backend, timeout_ms=timeout_ms)
        # the SPA may rotate the token (e.g. refresh-on-load); honour that
        new_token = jar.pop("token", None)
        if new_token and new_token != self.session_token:
            self.session_token = new_token
        # merge the full anti-bot jar; the curl_cffi session jar will
        # pick these up via the explicit cookies= on each request
        self.extra_cookies.update(jar)
        self._warmed_up = True
        return dict(self.extra_cookies)

    def refresh(self, *, force: bool = False) -> str:
        """Exchange the session cookie for a fresh 900 s access token.

        The refresh lives on the dedicated auth host and is the one call that
        goes cross-host with cookie credentials (plus ``x-request-origin``).
        The 30-day refresh token rotates on use; the response payload carries
        both tokens.
        """
        if not self.session_token:
            raise exc.TokenExpiredError("no session token; cannot refresh")
        r = self.http.get(
            f"{AUTH_BASE}/auths/refresh",
            cookies=self._cookies(),
            headers=self.headers(bearer=False,
                                 **{"x-request-origin": "https://chat.qwen.ai"}),
            timeout=self.timeout)
        d = self._json(r)
        if not d.get("success"):
            data = (d.get("data") or {}) if isinstance(d.get("data"), dict) else {}
            raise exc.AuthError(
                f"refresh failed: {data.get('code', r.status_code)}",
                status=r.status_code, payload=d)
        data = d["data"]
        self.access_token = data["access_token"]
        self.refresh_token = data.get("refresh_token") or self.refresh_token
        self._access_expires_at = time.time() + 900 - 30  # 15 min minus margin
        self._bootstrapped = True
        # capture the user UUID from the JWT for the status-beacon payload
        # (the SPA includes it as typarm2 on every /users/status call)
        try:
            import base64 as _b64
            payload_b64 = self.access_token.split(".")[1]
            # JWT uses base64url without padding
            payload_b64 += "=" * (-len(payload_b64) % 4)
            jwt_payload = json.loads(_b64.urlsafe_b64decode(payload_b64))
            self._user_uuid = jwt_payload.get("id") or self._user_uuid
        except Exception:  # noqa: BLE001 - best-effort
            pass
        # refresh happens cross-host (auth.qwen.ai) - capture any Set-Cookie
        # it returns so the persisted jar stays complete across restarts
        self._absorb_response_cookies(r)
        return self.access_token

    def ensure_access_token(self) -> str:
        """Return a valid access token, refreshing/re-signing-in as needed."""
        if self.access_token and time.time() < self._access_expires_at:
            return self.access_token
        if self.auto_refresh:
            if self.session_token:
                self.refresh()
                return self.access_token  # type: ignore[return-value]
            if self.email and self._password:
                self.signin()
                # if warmup was configured, the new session token needs a
                # fresh anti-bot jar - the cookies from the old sign-in
                # are no longer valid against the new session
                if self._warmup_enabled:
                    try:
                        self.do_warmup(backend=self._warmup_backend, force=True)
                    except Exception:  # noqa: BLE001 - warn, don't crash
                        import warnings as _w
                        _w.warn(
                            "post-resignin warmup failed; the session is "
                            "usable but the risk engine may punish it",
                            RuntimeWarning, stacklevel=2)
                self.refresh()
                return self.access_token  # type: ignore[return-value]
        if self.access_token:
            return self.access_token
        raise exc.TokenExpiredError("access token expired and no renewal path")

    # ------------------------------------------------------------- request core
    @staticmethod
    def _read_body(r: Any) -> str:
        """Read a (possibly streamed) response body as text, both stacks.

        curl_cffi returns an empty ``.text`` for responses opened with
        ``stream=True``; iterate the content instead.
        """
        try:
            t = r.text
            if t:
                return t
        except Exception:  # noqa: BLE001
            pass
        chunks = []
        try:
            for c in r.iter_content(8192):
                chunks.append(c.decode("utf-8", "ignore")
                              if isinstance(c, bytes) else c)
        except Exception:  # noqa: BLE001
            pass
        return "".join(chunks)

    @staticmethod
    def _json(r: Any) -> Dict[str, Any]:
        try:
            return r.json()
        except ValueError:
            return {}

    @staticmethod
    def _check_punish(text: str) -> None:
        if _looks_punished(text):
            # try to extract the punish URL so the caller (or the auto-solve
            # path in open_stream) can pass it to qwen_studio.captcha.solve_punish
            punish_url = None
            try:
                import re as _re
                m = _re.search(r'"url"\s*:\s*"([^"]+)"', text)
                if m:
                    punish_url = m.group(1).replace("\\/", "/").replace(":443", "")
            except Exception:  # noqa: BLE001
                pass
            raise exc.PunishedError(
                "request intercepted by the anti-bot risk engine (RGV587); "
                "stop and wait several minutes before any further call",
                body=text[:500],
                punish_url=punish_url)

    def _check_app_error(self, r: Any, d: Dict[str, Any]) -> None:
        if d.get("success") is True:
            return
        data = d.get("data") if isinstance(d.get("data"), dict) else {}
        code = (data or {}).get("code")
        details = (data or {}).get("details")
        status = r.status_code
        if code in ("Not_Found", "CHAT_NOT_FOUND"):
            raise exc.NotFoundError(details or "not found", code=code,
                                    details=details, status=status, payload=d)
        if code in ("Bad_Request", "PARENT_NOT_FOUND", "RequestValidationError"):
            raise exc.BadRequestError(details or "bad request", code=code,
                                      details=details, status=status, payload=d)
        if exc.QuotaError.matches(code) or exc.RateLimitedError.matches(code):
            cls = exc.RateLimitedError if exc.RateLimitedError.matches(code) else exc.QuotaError
            raise cls(f"{code}: {details or ''}".strip(), code=code,
                      details=details, status=status, payload=d)
        raise exc.APIError(f"{code or status}: {details or 'request failed'}",
                           code=code, details=details, status=status, payload=d)

    def request(self, method: str, path: str, *, params: Optional[dict] = None,
                json_body: Any = None, bearer: bool = True,
                cookie_auth: bool = False, **hdr: Any) -> Dict[str, Any]:
        """One JSON API call with auth renewal, punish detection and error mapping."""
        if bearer:
            self.ensure_access_token()
        self._paced_sleep()
        r = self.http.request(
            method, f"{BASE}{path}", params=params, json=json_body,
            headers=self.headers(bearer=bearer and bool(self.access_token), **hdr),
            cookies=self._cookies() if cookie_auth else None,
            timeout=self.timeout)
        self._check_punish(r.text[:2000])
        d = self._json(r)
        self._check_app_error(r, d)
        return d

    # ----------------------------------------------------------------- high lvl
    def models(self) -> Dict[str, Any]:
        """Model catalogue (``GET /models/``). Ids rotate; never hardcode."""
        return self.request("GET", "/models/", bearer=True).get("data") or {}

    def list_model_ids(self) -> list:
        data = self.models()
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [m.get("id") or m.get("model") for m in v
                            if isinstance(m, dict)]
        return []

    # ------------------------------------------------------ browser mimicry
    #: how often (seconds) to refresh the sidebar/listing endpoints when
    #: :attr:`_mimic_browser` is on. The real SPA polls these ~30s when
    #: idle and immediately after each chat completes.
    SIDEBAR_REFRESH_INTERVAL = 30.0

    def _send_status_beacon(self, *, page_id: str = "//chat.qwen.ai/",
                            kind: str = "session") -> None:
        """Mimic the SPA's ``POST /api/v2/users/status`` telemetry beacon.

        The browser fires this BEFORE and AFTER every ``/chat/completions``
        call (and on every SPA route change). The payload shape captured
        from the live SPA:

            {"typarms": {
                "typarm1": "web",                # source
                "typarm2": "<user-uuid>",        # account id (post-login)
                "typarm3": "prod",                # env
                "typarm4": "qwen_chat",           # product
                "typarm5": "product",             # channel
                "typarm6": "",
                "orgid": "tongyi",
                "share_id": "", "project_id": "",
                "channel_type": "", "community_type": "", "from_id": "",
                "cdn_version": "0.3.11",
                "page_id": "//chat.qwen.ai/c/<chat_id>",   # current route
                "spmId": "a2ty_o01.29997180"     # marketing id
            }}

        ``kind=session`` sends the navigation/session beacon above;
        ``kind=beacon`` sends the alternative ``sendBeacon`` form with a
        random logId + timestamp (which the SPA fires as
        ``navigator.sendBeacon`` onunload).

        Failures are swallowed: the beacon is decorative for the risk
        engine, not authoritative for the application.
        """
        if not self._mimic_browser:
            return
        try:
            if kind == "beacon":
                payload = {
                    "typarms": {
                        "logId": uuid.uuid4().hex,
                        "timestamp": int(time.time() * 1000),
                        "domain": "chat.qwen.ai",
                        "testTag": "compareLogService",
                        "testVersion": "5.0.0",
                        "serviceName": "tongyiLogService",
                        "requestType": "sendBeacon",
                    }
                }
            else:
                payload = {
                    "typarms": {
                        "typarm1": self.source,
                        "typarm2": self._user_uuid or "",
                        "typarm3": "prod",
                        "typarm4": "qwen_chat",
                        "typarm5": "product",
                        "typarm6": "",
                        "orgid": "tongyi",
                        "share_id": "",
                        "project_id": "",
                        "channel_type": "",
                        "community_type": "",
                        "from_id": "",
                        "cdn_version": self.version,
                        "page_id": page_id,
                        "spmId": "a2ty_o01.29997180",
                    }
                }
            # fire-and-forget - short timeout, no error mapping
            self.http.post(
                f"{BASE}/users/status", json=payload,
                headers=self.headers(bearer=bool(self.access_token)),
                cookies=self._cookies(),
                timeout=10.0,
            )
        except Exception:  # noqa: BLE001 - decorative beacon
            pass

    def _refresh_sidebar(self, *, force: bool = False) -> None:
        """Refresh the sidebar/listing endpoints the SPA polls.

        The real SPA calls these immediately after a chat completes and
        every ~30s while idle. The risk engine scores the absence of this
        background noise: a session that ONLY hits ``/chat/completions``
        looks like a bot even with a perfect cookie jar.

        Captured from the live SPA around each chat:
        - ``GET /configs/``, ``/configs/setting-config``, ``/tts/config``
          (SPA re-reads feature flags)
        - ``GET /chats/pinned``, ``/chats/?page=1&...``, ``/library/list``,
          ``/folders/``, ``/projects/``, ``/users/user/settings``,
          ``/credits/pricing`` (sidebar refresh)
        - ``POST /files/customer-service/entry`` (customer service beacon;
          the SPA pings this on every chat to track user activity)

        Throttled to ``SIDEBAR_REFRESH_INTERVAL``; pass ``force=True`` to
        bypass the throttle (used by :meth:`open_stream` after a chat
        completes).
        """
        if not self._mimic_browser:
            return
        now = time.time()
        if not force and (now - self._last_sidebar_refresh
                          < self.SIDEBAR_REFRESH_INTERVAL):
            return
        self._last_sidebar_refresh = now
        # GET endpoints - the SPA polls these on every chat
        get_paths = (
            "/configs/",
            "/configs/setting-config",
            "/tts/config?omni_speakers=v1&audio_tts_speakers=v1&"
            "omni_language=v1&audio_tts_language=v1",
            "/chats/pinned",
            "/chats/?page=1&exclude_project=true",
            "/library/list?type=all",
            "/folders/?exclude_project=true",
            "/projects/",
            "/users/user/settings",
            "/credits/pricing",
        )
        for path in get_paths:
            try:
                self.http.get(
                    f"{BASE}{path}",
                    headers=self.headers(bearer=bool(self.access_token)),
                    cookies=self._cookies(),
                    timeout=10.0,
                )
            except Exception:  # noqa: BLE001 - decorative poll
                pass
        # POST /files/customer-service/entry - customer service beacon
        try:
            self.http.post(
                f"{BASE}/files/customer-service/entry",
                json={},
                headers=self.headers(bearer=bool(self.access_token)),
                cookies=self._cookies(),
                timeout=10.0,
            )
        except Exception:  # noqa: BLE001 - decorative beacon
            pass

    def _send_aplus_beacon(self) -> None:
        """Fire a tracking beacon at ``aplus.qwen.ai``.

        The SPA fires these continuously (the Aliyun aplus tracker SDK).
        They go to a *different* host (``aplus.qwen.ai``) but the risk
        engine cross-references them: a session that hits qwen.ai APIs
        without firing any aplus beacons looks like a bot. We send one
        beacon per chat to close that gap.

        Failures are swallowed: the beacon is decorative for the risk
        engine, not authoritative for the application.
        """
        if not self._mimic_browser:
            return
        try:
            # the simplest aplus beacon: a POST to /aes.1.1 with a tiny
            # JSON payload. The real SPA sends a much richer payload
            # (spmId chain, page info, etc.) but the risk engine only
            # checks for the *presence* of aplus traffic, not its shape.
            self.http.post(
                "https://aplus.qwen.ai/aes.1.1",
                json={"_p_url": "https://chat.qwen.ai/",
                      "logtype": 2,
                      "gmkey": "OTHER",
                      "gokey": f"pid=chat_qwen_ai&_p_url=https%3A%2F%2Fchat.qwen.ai%2F"
                               f"&cache=2f7c9a2&jsver=aplus.js&lver=1.13.26"
                               f"&platformType=pc&device_model=Linux&os=Linux"
                               f"&language=en-US&b=chrome131"
                               f"&cna={self.extra_cookies.get('cna', '')}"
                               f"&_t={int(time.time() * 1000)}"},
                headers={"Content-Type": "application/json",
                         "User-Agent": WEB_UA,
                         "Referer": "https://chat.qwen.ai/",
                         "Origin": "https://chat.qwen.ai"},
                timeout=5.0,
            )
        except Exception:  # noqa: BLE001 - decorative beacon
            pass

    # ---------------------------------------------------------------- streaming
    def open_stream(self, body: Dict[str, Any], chat_id: str) -> Iterator[str]:
        """POST /chat/completions and return the *raw* SSE line iterator.

        When ``mimic_browser=True`` (default), this also fires the same
        ``POST /api/v2/users/status`` telemetry beacon the SPA fires
        before AND after every chat (see :meth:`_send_status_beacon`),
        plus a sidebar refresh (see :meth:`_refresh_sidebar`). The risk
        engine scores request PATTERN, not just cookies - a clean
        ``/chats/new + /chat/completions`` sequence is the strongest bot
        signal after cookie absence.

        When ``auto_solve_captcha=True`` (default), a
        :class:`PunishedError` from this call triggers
        :func:`qwen_studio.captcha.solve_punish` which launches a
        headless browser to drag the Baxia slider, obtains the
        ``x5sec`` cookie, merges it into ``extra_cookies``, and retries
        the stream exactly once. If the retry also punishes, the
        ``PunishedError`` is re-raised.

        Note the extra request headers the web client uses for streams:
        ``Accept: application/json`` and ``x-accel-buffering: no`` (the latter
        disables proxy buffering so frames arrive live). Raises the same
        errors as :meth:`stream_completion` for non-SSE replies.
        """
        try:
            yield from self._open_stream_once(body, chat_id)
        except exc.PunishedError as e:
            if not self._auto_solve_captcha or self._captcha_solve_attempted:
                raise
            if not e.punish_url:
                # no URL to solve - re-raise as-is
                raise
            # attempt the slider solve
            self._captcha_solve_attempted = True
            try:
                from .captcha import solve_punish, has_solver
                if not has_solver():
                    import warnings as _w
                    _w.warn(
                        "punished but Playwright is not installed; install "
                        "it with `pip install playwright && playwright "
                        "install chromium` to enable the automatic slider "
                        "solver", RuntimeWarning, stacklevel=2)
                    raise
                # solve the slider and merge the resulting cookies
                new_cookies = solve_punish(e.punish_url, self._cookies())
                self.extra_cookies.update(new_cookies)
            except Exception as solve_err:
                # solver failed - re-raise the original PunishedError
                # with the solver error as context
                import warnings as _w
                _w.warn(
                    f"captcha auto-solve failed: {solve_err}; the original "
                    f"PunishedError is re-raised", RuntimeWarning,
                    stacklevel=2)
                raise e from solve_err
            # retry the stream once with the new x5sec cookie
            yield from self._open_stream_once(body, chat_id)
            # reset the guard so future punish events can also be solved
            self._captcha_solve_attempted = False

    def _open_stream_once(self, body: Dict[str, Any],
                          chat_id: str) -> Iterator[str]:
        """Single attempt at POST /chat/completions. See :meth:`open_stream`."""
        self.ensure_access_token()
        # browser fires users/status beacon before every chat
        self._send_status_beacon(page_id=f"//chat.qwen.ai/c/{chat_id}")
        self._paced_sleep()
        r = self.http.post(
            f"{BASE}/chat/completions", params={"chat_id": chat_id},
            json=body,
            headers=self.headers(Accept="application/json",
                                 **{"x-accel-buffering": "no"}),
            cookies=self._cookies(),
            stream=True, timeout=self.timeout)
        ct = r.headers.get("content-type", "")
        if "event-stream" not in ct:
            raw = self._read_body(r)[:2000]
            self._check_punish(raw)
            # surface as app error or generic API error
            d = self._json(r)
            if d:
                self._check_app_error(r, d)
            raise exc.StreamInterruptedError(
                f"expected SSE, got content-type={ct!r} status={r.status_code}")
        # stream the SSE response
        for line in iter_lines_compat(r):
            yield line
        # browser fires users/status beacon + sidebar refresh AFTER the chat
        self._send_status_beacon(page_id=f"//chat.qwen.ai/c/{chat_id}",
                                 kind="beacon")
        self._refresh_sidebar(force=True)
        # fire the aplus.qwen.ai tracking beacon the SPA fires after each chat
        self._send_aplus_beacon()

    def stream_events(self, body: Dict[str, Any], chat_id: str) -> Iterator["ChatEvent"]:
        """Incremental variant of :meth:`stream_completion`: yields parsed
        :class:`~qwen_studio.sse.ChatEvent` objects as they arrive instead of
        draining into an aggregate. Powers live proxies (the OpenAI-compatible
        server uses exactly this).
        """
        from .sse import iter_sse_frames, parse_event
        for ev in iter_sse_frames(self.open_stream(body, chat_id)):
            yield parse_event(ev)

    def stream_completion(self, body: Dict[str, Any], chat_id: str, *,
                          keep_events: bool = False) -> StreamResult:
        """POST /chat/completions and drain the SSE stream."""
        return consume(self.open_stream(body, chat_id), keep_events=keep_events)
