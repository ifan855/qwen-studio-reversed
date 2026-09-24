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
import time
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional

if TYPE_CHECKING:  # annotation-only import (keeps the sse module optional)
    from .sse import ChatEvent

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


def host_of(url: str) -> str:
    """``https://chat.qwen.ai/api/v2`` -> ``chat.qwen.ai`` (for error text)."""
    return url.split("//", 1)[-1].split("/", 1)[0] or url


# First component of the modules transport failures come from, e.g.
# ``curl_cffi.requests.errors.RequestsError`` -> ``curl_cffi``.
TRANSPORT_MODULES = ("curl_cffi", "requests", "urllib3", "socket", "ssl")


def as_transport_error(e: BaseException, target: str) -> Optional[exc.TransportError]:
    """Rewrite a low-level HTTP/socket failure as a typed short error.

    ``curl_cffi``, ``requests`` and socket failures (DNS, TLS reset, blocked
    egress) arrive as foreign exception types carrying a wall of transport
    detail - e.g. ``curl: (35) BoringSSL SSL_connect: Connection closed
    abruptly`` - which tells the caller neither which host failed nor what to
    do. The rewritten error names the host and keeps the original text in
    ``details``; the OpenAI-compatible proxy maps it to a 502
    ``upstream_unavailable`` instead of a generic 500.

    Returns ``None`` when *e* is not a transport failure, so callers can
    re-raise it unchanged.
    """
    if isinstance(e, exc.QwenStudioError):
        return None
    module = type(e).__module__.split(".", 1)[0]
    if not (isinstance(e, OSError) or module in TRANSPORT_MODULES):
        return None
    text = " ".join(str(e).split())
    return exc.TransportError(f"cannot reach {target}: {text[:200]}",
                              target=target, details=text)


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
                 extra_cookies: Optional[Dict[str, str]] = None) -> None:
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
    def from_credentials(cls, email: str, password: str, **kw: Any) -> "QwenStudio":
        """Sign in with email + plaintext password (hashed in transit)."""
        c = cls(email=email, password=password, **kw)
        c.signin()
        return c

    @classmethod
    def from_session_token(cls, token: str, **kw: Any) -> "QwenStudio":
        """Reuse a 30-day session token (e.g. exported from a browser cookie)."""
        c = cls(session_token=token, **kw)
        if kw.get("auto_refresh", True):
            try:
                c.refresh()
            except Exception:  # noqa: BLE001 - stay cookie-capable
                pass  # bearer calls will raise on demand
        return c

    @classmethod
    def from_access_token(cls, token: str, **kw: Any) -> "QwenStudio":
        """Use a live access token; no renewal material, it will simply expire."""
        return cls(access_token=token, auto_refresh=False, **kw)

    @classmethod
    def from_browser(cls, browser: Optional[str] = None,
                     profile: Optional[str] = None,
                     domain: str = "qwen.ai", **kw: Any) -> "QwenStudio":
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
        """
        from . import browser_cookies as bc
        prof, cookies = bc.find_qwen_jar(browser, profile, domain=domain)
        jar = {c.name: c.value for c in cookies}
        token = jar.pop("token", None)
        if not token:
            raise exc.AuthError(
                f"no session token for {domain!r} in {prof.browser}:"
                f"{prof.name}; log in to chat.qwen.ai in that browser first")
        c = cls(session_token=token, extra_cookies=jar, **kw)
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
    def _call(self, fn: Any, target: str) -> Any:
        """Run one raw HTTP call, typing transport failures (see
        :func:`as_transport_error`) so callers never see a bare curl error."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - re-raised below
            te = as_transport_error(e, target)
            if te is None:
                raise
            raise te from e

    def signin(self) -> Dict[str, Any]:
        """Password sign-in -> 30-day session token."""
        if not (self.email and self._password):
            raise exc.AuthError("signin requires email and password")
        pw = hashlib.sha256(self._password.encode()).hexdigest()
        r = self._call(lambda: self.http.post(
            f"{BASE}/auths/signin", json={"email": self.email, "password": pw},
            headers=self.headers(bearer=False), timeout=self.timeout),
            host_of(BASE))
        d = self._json(r)
        if not d.get("success"):
            raise exc.AuthError("signin rejected", status=r.status_code, payload=d)
        data = d.get("data") or {}
        self.session_token = data.get("token") or self.session_token
        self._bootstrapped = True
        return data

    def refresh(self, *, force: bool = False) -> str:
        """Exchange the session cookie for a fresh 900 s access token.

        The refresh lives on the dedicated auth host and is the one call that
        goes cross-host with cookie credentials (plus ``x-request-origin``).
        The 30-day refresh token rotates on use; the response payload carries
        both tokens.
        """
        if not self.session_token:
            raise exc.TokenExpiredError("no session token; cannot refresh")
        r = self._call(lambda: self.http.get(
            f"{AUTH_BASE}/auths/refresh",
            cookies={"token": self.session_token},
            headers=self.headers(bearer=False,
                                 **{"x-request-origin": "https://chat.qwen.ai"}),
            timeout=self.timeout),
            host_of(AUTH_BASE))
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
            raise exc.PunishedError(
                "request intercepted by the anti-bot risk engine (RGV587); "
                "stop and wait several minutes before any further call",
                body=text[:500])

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
        r = self._call(lambda: self.http.request(
            method, f"{BASE}{path}", params=params, json=json_body,
            headers=self.headers(bearer=bearer and bool(self.access_token), **hdr),
            cookies=self._cookies() if cookie_auth else None,
            timeout=self.timeout), host_of(BASE))
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

    # ---------------------------------------------------------------- streaming
    def open_stream(self, body: Dict[str, Any], chat_id: str) -> Iterator[str]:
        """POST /chat/completions and return the *raw* SSE line iterator.

        Note the extra request headers the web client uses for streams:
        ``Accept: application/json`` and ``x-accel-buffering: no`` (the latter
        disables proxy buffering so frames arrive live). Raises the same
        errors as :meth:`stream_completion` for non-SSE replies.
        """
        self.ensure_access_token()
        self._paced_sleep()
        r = self._call(lambda: self.http.post(
            f"{BASE}/chat/completions", params={"chat_id": chat_id},
            json=body,
            headers=self.headers(Accept="application/json",
                                 **{"x-accel-buffering": "no"}),
            cookies=self._cookies(),
            stream=True, timeout=self.timeout), host_of(BASE))
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
        return iter_lines_compat(r)

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
