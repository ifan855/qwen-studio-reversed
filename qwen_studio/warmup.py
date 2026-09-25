"""Headless-browser warmup: mint the complete anti-bot cookie jar.

The Qwen Studio risk engine (Alibaba's RGV587 / x5sec stack) discriminates
on the *complete* ``.qwen.ai`` cookie jar - see docs/anti-bot.md. Of that
jar, the session ``token`` and the WAF cookie ``acw_tc`` come straight from
``POST /api/v2/auths/signin``. The rest of the risk-engine cookies
(``cna``, ``tfstk``, ``isg``, ``ssxmod_itna``, ``ssxmod_itna2``,
``cnaui``, ``aui``, ``atpsida``, ``sca``, ``xlly_s``, ...) are minted by
Aliyun's edge / tracking infrastructure via JS beacons that fire when the
SPA boots. A pure-HTTP client (even one with a perfect Chrome TLS
fingerprint) cannot synthesise them - they require a real DOM that
actually runs the SPA's JavaScript.

This module launches a *headless* Chromium (Playwright preferred, the
``agent-browser`` CLI as a subprocess fallback) for a few seconds, injects
the session token we already obtained via password sign-in, lets the SPA
mint the full jar, and returns it as a flat ``{name: value}`` dict.

The result is merged into :attr:`QwenStudio.extra_cookies`, so every
subsequent request through the curl_cffi transport presents exactly the
same cookie material the official browser does - the decisive variable
in the A/B/A experiment in docs/anti-bot.md.

This is a *warmup* layer, not a token minter. It does not log in, solve
challenges, or perform any interaction with the anti-bot infrastructure
beyond letting the SPA's own first-party JS run. It is the missing piece
that makes :meth:`QwenStudio.from_credentials` reach the same risk-score
footing as :meth:`QwenStudio.from_browser` on machines that have no
logged-in browser profile.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional

from .exceptions import QwenStudioError

__all__ = [
    "WarmupError",
    "warmup_cookie_jar",
    "has_playwright",
    "has_agent_browser",
    "WARMUP_REQUIRED_COOKIES",
    "WARMUP_TARGET_URL",
]

#: cookies we expect a healthy warmup to surface (the risk-engine set).
#: Used by callers to log how many are missing; not enforced.
WARMUP_REQUIRED_COOKIES = (
    "cna", "tfstk", "isg", "ssxmod_itna", "ssxmod_itna2",
    "cnaui", "aui", "atpsida", "sca", "xlly_s",
)

WARMUP_TARGET_URL = "https://chat.qwen.ai/"

#: the same Chrome UA the curl_cffi transport presents
WEB_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


class WarmupError(QwenStudioError):
    """The headless-browser warmup could not be completed.

    Common causes: Playwright is not installed, the Chromium binary is
    missing, the SPA failed to reach a network-idle state within the
    timeout, or no cookies were captured at all. The caller should treat
    this as a soft-failure: the session remains usable but the
    anti-bot risk engine may eventually punish it - fall back to
    :meth:`QwenStudio.from_browser` if a local browser profile is
    available, or to manual credential refreshes at low call rates.
    """


# ---------------------------------------------------------- backend detection
def has_playwright() -> bool:
    """True iff the Playwright sync API and a Chromium install are importable."""
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except ImportError:
        return False


def has_agent_browser() -> bool:
    """True iff the ``agent-browser`` CLI is on PATH (used as a fallback)."""
    return shutil.which("agent-browser") is not None


# -------------------------------------------------------------- playwright
def _warmup_with_playwright(session_token: str, *,
                            timeout_ms: int = 30_000,
                            settle_ms: int = 3_000,
                            headless: bool = True) -> Dict[str, str]:
    """Use Playwright's sync API to load the SPA and harvest the cookie jar."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            context = browser.new_context(
                user_agent=WEB_UA,
                viewport={"width": 1280, "height": 720},
                locale="en-US",
            )
            # pre-inject the session token so the SPA boots authenticated -
            # this avoids the second hop through the login form, which the
            # risk engine scores separately and which we already did via
            # /auths/signin
            context.add_cookies([{
                "name": "token",
                "value": session_token,
                "domain": ".qwen.ai",
                "path": "/",
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            }])
            page = context.new_page()
            # networkidle lets Aliyun's tracking beacons (acrm.aliyun.com,
            # log.aliyun.com) finish minting cna/tfstk/isg/ssxmod_itna*
            try:
                page.goto(WARMUP_TARGET_URL, wait_until="networkidle",
                          timeout=timeout_ms)
            except Exception:
                # networkidle may not fire if a beacon keeps a long poll
                # open; the cookie jar is what we care about, not nav state
                pass
            time.sleep(max(0, settle_ms) / 1000.0)
            # one reload catches the deferred beacons that fire after
            # the SPA mounts and starts polling /api/chat/list, etc.
            try:
                page.reload(wait_until="networkidle", timeout=timeout_ms)
            except Exception:
                pass
            time.sleep(max(0, settle_ms) / 1000.0)
            cookies = context.cookies(WARMUP_TARGET_URL)
        finally:
            browser.close()
    if not cookies:
        raise WarmupError(
            "Playwright warmup returned no cookies; the SPA likely did not "
            "boot - check that the session token is valid and that Chromium "
            "is installed (`playwright install chromium`).")
    return {c["name"]: c["value"] for c in cookies}


# -------------------------------------------------------------- agent-browser
def _warmup_with_agent_browser(session_token: str, *,
                               timeout_s: int = 30) -> Dict[str, str]:
    """Shell out to the ``agent-browser`` CLI to mint the cookie jar.

    Used when Playwright is not available. The CLI uses an isolated
    Chromium profile and exposes ``cookies`` / ``state save`` commands
    we can drive from a temp directory. We inject the session token via
    the CLI's own cookie-set command, navigate, then dump the jar.
    """
    session_id = f"qwen-warmup-{os.getpid()}-{int(time.time())}"
    ab = ["agent-browser", "--session", session_id]

    def _run(args: list, **kw: Any) -> subprocess.CompletedProcess:
        return subprocess.run(ab + args, capture_output=True, text=True,
                              timeout=timeout_s, **kw)

    state_path = tempfile.mktemp(prefix="qwen-warmup-state-", suffix=".json")
    try:
        # 1. open the SPA (unauthenticated) - this sets the basic edge
        #    cookies (acw_tc, x-ap) on the agent-browser profile
        _run(["open", WARMUP_TARGET_URL])

        # 2. inject our session token cookie so the reload is authenticated
        _run(["cookies", "set", "token", session_token])

        # 3. reload - the SPA now boots with the session token in place
        #    and the JS beacons mint the risk-engine cookies
        _run(["reload"])
        # let the beacons settle
        time.sleep(3)
        try:
            _run(["wait", "--load", "networkidle"])
        except subprocess.TimeoutExpired:
            pass

        # 4. save the full state (cookies + localStorage) to a JSON file
        _run(["state", "save", state_path])

        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except subprocess.TimeoutExpired as e:
        raise WarmupError(f"agent-browser timed out after {timeout_s}s: "
                          f"{e.cmd}") from e
    except FileNotFoundError as e:
        raise WarmupError(
            "agent-browser CLI not found on PATH; install it with "
            "`npm install -g agent-browser && agent-browser install`, "
            "or use Playwright (`pip install playwright && "
            "playwright install chromium`)") from e
    finally:
        # tear down the isolated session so we don't leak a browser
        try:
            _run(["close"])
        except Exception:
            pass
        try:
            os.unlink(state_path)
        except OSError:
            pass

    cookies = state.get("cookies", []) if isinstance(state, dict) else []
    if not cookies:
        raise WarmupError(
            "agent-browser warmup returned no cookies; the SPA likely did "
            "not boot - check the session token and the agent-browser "
            "install (`agent-browser install --with-deps`).")
    # cookies here have a broader domain set; keep only qwen.ai hosts
    out: Dict[str, str] = {}
    for c in cookies:
        domain = (c.get("domain") or "").lstrip(".")
        if "qwen.ai" not in domain:
            continue
        out[c["name"]] = c.get("value", "")
    return out


# -------------------------------------------------------------- top-level
def warmup_cookie_jar(session_token: str, *,
                      backend: str = "auto",
                      timeout_ms: int = 30_000,
                      headless: bool = True) -> Dict[str, str]:
    """Run a headless-browser warmup and return the full ``.qwen.ai`` jar.

    Parameters
    ----------
    session_token
        The 30-day session cookie value obtained from
        :meth:`QwenStudio.signin` (or the browser's ``token`` cookie).
        It is injected into the headless browser before navigation so the
        SPA boots authenticated and all first-party beacons fire.
    backend
        ``"auto"`` (default) tries Playwright first, then ``agent-browser``;
        ``"playwright"`` / ``"agent-browser"`` force one.
    timeout_ms
        Per-navigation timeout. The full warmup can take roughly
        ``3 * timeout_ms / 1000`` seconds in the worst case (initial load
        + reload + settle).
    headless
        Forwarded to Playwright. ``False`` is useful for debugging on a
        desktop with a display.

    Returns
    -------
    dict
        ``{name: value}`` for every ``qwen.ai`` cookie the browser ended
        up with. The session ``token`` is included so the caller can
        update :attr:`QwenStudio.session_token` if the server rotated it.

    Raises
    ------
    WarmupError
        No backend available, the SPA did not boot, or no cookies were
        captured. The caller should treat this as a soft-failure (the
        session remains usable but risk-prone).
    """
    if not session_token:
        raise WarmupError("warmup requires a session token; sign in first")

    backends: list
    if backend == "auto":
        backends = []
        if has_playwright():
            backends.append("playwright")
        if has_agent_browser():
            backends.append("agent-browser")
    elif backend in ("playwright", "agent-browser"):
        backends = [backend]
    else:
        raise WarmupError(f"unknown warmup backend {backend!r}; "
                          "use 'auto', 'playwright' or 'agent-browser'")

    if not backends:
        raise WarmupError(
            "no warmup backend available: install Playwright "
            "(`pip install playwright && playwright install chromium`) "
            "or the agent-browser CLI (`npm install -g agent-browser && "
            "agent-browser install`)")

    last_err: Optional[Exception] = None
    for b in backends:
        try:
            if b == "playwright":
                return _warmup_with_playwright(
                    session_token,
                    timeout_ms=timeout_ms,
                    headless=headless,
                )
            return _warmup_with_agent_browser(
                session_token,
                timeout_s=max(10, timeout_ms // 1000),
            )
        except Exception as e:  # noqa: BLE001 - try the next backend
            last_err = e
            print(f"  warmup backend {b!r} failed: {e}", file=sys.stderr)
    raise WarmupError(f"all warmup backends failed; last error: {last_err}")


if __name__ == "__main__":  # pragma: no cover
    # quick CLI debug: python -m qwen_studio.warmup <session_token>
    if len(sys.argv) != 2:
        print("usage: python -m qwen_studio.warmup <session_token>",
              file=sys.stderr)
        sys.exit(2)
    jar = warmup_cookie_jar(sys.argv[1])
    print(f"captured {len(jar)} cookies:")
    for k in sorted(jar):
        print(f"  {k:<22} {jar[k][:24]}")
    missing = set(WARMUP_REQUIRED_COOKIES) - set(jar)
    if missing:
        print(f"\nmissing required cookies: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)
