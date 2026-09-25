"""Baxia slider auto-solver.

When the anti-bot risk engine punishes a session (RGV587), the punish
response contains a URL pointing to a Baxia punish page. The page loads
Alibaba's AWSC (Anti-bot Web Security Center) captcha JS which renders
a "slide to unlock" slider. The slider must be dragged to the end of
the track; the AWSC JS computes a fingerprint (Fireye token) from the
browser environment and submits it with the slide data to the
``/_____tmd_____/slide`` endpoint. If the fingerprint is valid, the
response carries an ``bx-x5sec`` header containing the ``x5sec`` cookie
that lifts the punish for ~30 minutes.

The fingerprinting library (Fireye) checks WebGL renderer strings,
canvas hash, audio context, and navigator properties. A headless
Chromium with software WebGL (swiftshader) reports "SwiftShader" as
the renderer, which is an immediate bot signal. This module spoofs
the WebGL renderer to look like a real Intel GPU before the AWSC JS
loads, so the fingerprint passes.

The solver:
1. Launches headless Chromium (Playwright) with swiftshader WebGL
2. Injects a spoof script that overrides WebGLRenderingContext.getParameter
   to return "Google Inc. (Intel)" / "ANGLE (Intel, Intel(R) UHD...)"
3. Navigates to the punish URL
4. Waits for the AWSC JS to render the slider
5. Drags the slider using Playwright's native mouse API (real browser events)
6. Waits for the /slide endpoint response
7. Extracts the ``x5sec`` cookie from the browser context
8. Returns the cookie value so the caller can set it on the QwenStudio client

If the slide response returns ``code: 300`` (failure), the fingerprint
was rejected. This usually means the spoof wasn't convincing enough.
The solver retries up to 3 times with different drag trajectories.

If Playwright is not installed, raises :class:`CaptchaSolverError` with
install instructions.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Dict, Optional

from .exceptions import QwenStudioError

__all__ = [
    "CaptchaSolverError",
    "solve_punish",
    "has_solver",
    "extract_punish_url",
]

#: script injected before any page JS runs — overrides WebGL renderer
#: strings, navigator.webdriver, and other fingerprint signals so the
#: Fireye fingerprinting library (used by AWSC) produces a "real browser"
#: fingerprint instead of detecting headless Chromium + swiftshader.
SPOOF_INIT_SCRIPT = """
// --- WebGL renderer spoof ---
const _getParameter = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(p) {
    if (p === 37445) return "Google Inc. (Intel)";           // UNMASKED_VENDOR_WEBGL
    if (p === 37446) return "ANGLE (Intel, Intel(R) UHD Graphics 630 " +
                          "Direct3D11 vs_5_0 ps_5_0, D3D11)";  // UNMASKED_RENDERER_WEBGL
    return _getParameter.call(this, p);
};
if (typeof WebGL2RenderingContext !== 'undefined') {
    const _getParameter2 = WebGL2RenderingContext.prototype.getParameter;
    WebGL2RenderingContext.prototype.getParameter = function(p) {
        if (p === 37445) return "Google Inc. (Intel)";
        if (p === 37446) return "ANGLE (Intel, Intel(R) UHD Graphics 630 " +
                              "Direct3D11 vs_5_0 ps_5_0, D3D11)";
        return _getParameter2.call(this, p);
    };
}

// --- navigator spoof ---
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {
    get: () => [{name: 'PDF Viewer'}, {name: 'Chrome PDF Viewer'},
                {name: 'Chromium PDF Viewer'}, {name: 'Microsoft Edge PDF Viewer'},
                {name: 'WebKit built-in PDF'}]
});
Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});

// --- permission API spoof ---
if (navigator.permissions && navigator.permissions.query) {
    const _query = navigator.permissions.query.bind(navigator.permissions);
    navigator.permissions.query = (p) =>
        p.name === 'notifications'
            ? Promise.resolve({state: Notification.permission})
            : _query(p);
}
"""

#: Chrome launch args that enable WebGL in headless mode while keeping
#: the fingerprint spoofable. ``--enable-unsafe-swiftshader`` lets Chromium
#: render WebGL via the software path; the spoof script then overrides
#: the renderer string so the fingerprint library can't detect it.
CHROME_ARGS = [
    "--enable-unsafe-swiftshader",
    "--use-gl=swiftshader",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--disable-blink-features=AutomationControlled",
]

#: the Chrome UA the warmup module uses (kept in sync)
WEB_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


class CaptchaSolverError(QwenStudioError):
    """The Baxia slider could not be solved automatically.

    Common causes: Playwright is not installed, the punish page
    didn't render a slider (escalated to QR-code deny), the slide
    endpoint returned ``code: 300`` after 3 retries (fingerprint
    rejected), or the x5sec cookie wasn't set in the response.
    """


def has_solver() -> bool:
    """True iff Playwright is installed with a Chromium binary."""
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except ImportError:
        return False


def extract_punish_url(punish_body: str) -> Optional[str]:
    """Parse the ``url`` field from a punish response body.

    The punish JSON has the shape::

        {"ret": ["FAIL_SYS_USER_VALIDATE", "RGV587_ERROR::..."],
         "data": {"url": "https://chat.qwen.ai:443//api/v2/.../punish?x5secdata=..."}}

    Returns the URL with ``:443`` stripped (the port causes double-slash
    issues with some HTTP clients). Returns ``None`` if no URL is found.
    """
    if not punish_body:
        return None
    m = re.search(r'"url"\s*:\s*"([^"]+)"', punish_body)
    if not m:
        return None
    url = m.group(1).replace("\\/", "/").replace(":443", "")
    return url


def solve_punish(punish_url: str, cookies: Dict[str, str], *,
                 max_retries: int = 3,
                 drag_distance: int = 280,
                 timeout_ms: int = 30_000) -> Dict[str, str]:
    """Solve the Baxia slider and return the cookies to merge.

    Parameters
    ----------
    punish_url
        The URL extracted from a :class:`PunishedError` body (see
        :func:`extract_punish_url`).
    cookies
        The current cookie jar (``{name: value}``) so the browser
        session inherits the authenticated state.
    max_retries
        How many times to retry the drag if the slide endpoint returns
        ``code: 300`` (fingerprint rejected). Each retry uses a
        slightly different drag trajectory.
    drag_distance
        How many pixels to drag the slider. The track is typically 300px
        wide; 280 reaches the end with a small margin.
    timeout_ms
        Per-navigation timeout in milliseconds.

    Returns
    -------
    dict
        ``{name: value}`` for every ``qwen.ai`` cookie the browser
        context holds after the solve — includes the critical
        ``x5sec`` cookie that lifts the punish. The caller should merge
        these into ``QwenStudio.extra_cookies``.

    Raises
    ------
    CaptchaSolverError
        Playwright is not installed, the slider didn't render, or the
        slide endpoint kept returning ``code: 300`` after all retries.
    """
    if not has_solver():
        raise CaptchaSolverError(
            "Playwright is not installed; install it with `pip install "
            "playwright && playwright install chromium` to enable the "
            "automatic Baxia slider solver")

    from playwright.sync_api import sync_playwright

    # convert flat cookie dict -> Playwright cookie format
    pw_cookies = []
    for name, value in cookies.items():
        pw_cookies.append({
            "name": name, "value": value,
            "domain": ".qwen.ai", "path": "/",
            "httpOnly": False, "secure": True, "sameSite": "Lax",
        })

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=CHROME_ARGS)
        try:
            ctx = browser.new_context(
                user_agent=WEB_UA,
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )
            ctx.add_init_script(SPOOF_INIT_SCRIPT)
            ctx.add_cookies(pw_cookies)
            page = ctx.new_page()

            # capture the /slide response
            slide_result: Dict[str, Any] = {}

            def _on_response(resp):
                if "/_____tmd_____/slide" in resp.url:
                    try:
                        body = resp.json()
                    except Exception:
                        body = {}
                    slide_result["code"] = body.get("result", {}).get("code")
                    slide_result["body"] = body
            page.on("response", _on_response)

            # navigate to the punish page
            page.goto(punish_url, wait_until="domcontentloaded",
                      timeout=timeout_ms)
            # let the AWSC JS load and render the slider
            time.sleep(8)

            for attempt in range(1, max_retries + 1):
                slider = page.query_selector("#nc_1_n1z")
                if not slider:
                    raise CaptchaSolverError(
                        "the AWSC slider (#nc_1_n1z) did not render on the "
                        "punish page; the punish may have escalated to a "
                        "QR-code deny flow that the solver cannot handle")

                box = slider.bounding_box()
                if not box:
                    raise CaptchaSolverError(
                        "slider element exists but has no bounding box; "
                        "the page may not have finished rendering")

                start_x = box["x"] + box["width"] / 2
                start_y = box["y"] + box["height"] / 2
                # add small variation per attempt so retries aren't identical
                end_x = start_x + drag_distance + random.uniform(-5, 5)

                # native Playwright mouse drag — dispatches real browser
                # mouse events that the AWSC JS picks up
                page.mouse.move(start_x, start_y)
                time.sleep(0.3)
                page.mouse.down()
                time.sleep(0.1)

                steps = 50 + random.randint(-5, 10)
                for i in range(1, steps + 1):
                    t = i / steps
                    # ease-out: fast at start, slow at end (human-like)
                    eased = 1 - (1 - t) ** 2
                    x = start_x + (end_x - start_x) * eased + \
                        random.uniform(-1, 1)
                    y = start_y + random.uniform(-1.5, 1.5)
                    page.mouse.move(x, y)
                    time.sleep(0.025 + random.uniform(0, 0.02))

                time.sleep(0.2)
                page.mouse.up()
                # wait for the /slide endpoint to respond
                time.sleep(8)

                code = slide_result.get("code")
                if code == 0:
                    break
                # code 300 = fingerprint rejected; retry with a new trajectory
                # (the captcha JS auto-resets the slider for another attempt)

            # extract all cookies from the browser context
            browser_cookies = ctx.cookies("https://chat.qwen.ai/")
        finally:
            browser.close()

    # check if the solve succeeded
    code = slide_result.get("code")
    if code != 0:
        raise CaptchaSolverError(
            f"the /slide endpoint returned code={code} after "
            f"{max_retries} attempts; the fingerprint was rejected "
            f"(response: {slide_result.get('body', {})})")

    # find the x5sec cookie
    result: Dict[str, str] = {}
    for c in browser_cookies:
        if "qwen.ai" in c.get("domain", ""):
            result[c["name"]] = c["value"]

    if "x5sec" not in result:
        raise CaptchaSolverError(
            "the /slide endpoint returned code=0 but no x5sec cookie was "
            "set in the browser context; the punish may require a different "
            "resolution path")

    return result


if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m qwen_studio.captcha <punish_url> [cookie_header]",
              file=sys.stderr)
        sys.exit(2)
    punish_url = sys.argv[1]
    cookie_header = sys.argv[2] if len(sys.argv) > 2 else ""
    cookies = {}
    for pair in cookie_header.split("; "):
        if "=" in pair:
            k, v = pair.split("=", 1)
            cookies[k] = v
    jar = solve_punish(punish_url, cookies)
    print(f"solved! got {len(jar)} cookies")
    print(f"x5sec: {jar.get('x5sec', '(missing)')[:80]}...")
