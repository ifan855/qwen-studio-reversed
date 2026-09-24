# The anti-bot risk engine, pinpointed

During the study the account repeatedly entered a state where
`POST /chat/completions` answered with the Alibaba edge punish flow
(`FAIL_SYS_USER_VALIDATE` / `RGV587_ERROR` JSON, or an HTML page running
`window.location.replace("..._____tmd_____/punish?x5secdata=...")`) while
the same endpoints kept working in a real browser. This page records the
controlled experiments that isolated the cause, and what the library does
about it.

## Experiment 1 - transport matrix (inconclusive on its own)

Same chat, same "pong" prompt, same headers, four transports run back to
back after a quiet period: python `requests` and `curl_cffi` Chrome
impersonation, each with the old (pre-incident) and a fresh (live browser)
cookie jar. **All four passed.** Combined with the earlier failures, this
ruled out a permanent fingerprint ban and showed the block is not
cookie-value-bound either.

## Experiment 2 - the decisive A/B/A

Same minute, same transport (`curl_cffi` Chrome impersonation), same chat,
same Bearer token - only the cookie jar differed:

| Call | Cookie jar | Result |
|---|---|---|
| V1 | **full browser jar (19 cookies)** | SSE OK - `pong` |
| V2 | bare jar (`acw_tc` + `x-ap` only) | **punished (x5sec)** |
| V3 | full browser jar again | SSE OK - `pong` |

## Findings

1. **The discriminator is the anti-bot cookie jar.** `/chat/completions`
   expects the complete `.qwen.ai` cookie set the browser accumulates:
   the session `token`, the WAF cookie `acw_tc`, and the risk-engine
   cookies (`tfstk`, `isg`, `ssxmod_itna`, `ssxmod_itna2`, `cna`, ...).
   With the full jar, even plain `requests` with a "wrong" TLS fingerprint
   passes. Without it, even a byte-perfect Chrome TLS impersonation is
   punished seconds later.
2. **Why the browser always works:** it naturally carries that jar. The
   observed punish episodes began when script traffic arrived with an
   empty or partial jar; the browser kept working throughout.
3. **Trigger dynamics:** a punished (cookie-bare) request itself elevates
   the session's risk score; repeated failures extend the window to even
   cheap endpoints (the WAF may then serve its `aliyun_waf_aa` JS
   challenge on any path). Quiet time (~15-30 min) decays the score.
   Solve-the-challenge behaviour (minting `acw_sc__v2` from the challenge
   JS) is deliberately **out of scope** - that is defeating the control
   rather than carrying valid session material.
4. **Endpoint-selective:** during the incidents only `/chat/completions`
   (and, once escalated, everything) was affected; with a valid jar,
   management endpoints never triggered anything.
5. **Not header validation:** all header mutations (`source`, `version`,
   `Timezone`, `x-request-id`) are accepted even mid-incident (see
   [authentication.md](authentication.md)).
6. **Two punish body shapes exist** - the JSON form
   (`{"ret": ["FAIL_SYS_USER_VALIDATE", ...], "data": {"url": "...tmd.../punish..."}}`)
   and the HTML form (a `<script>window.location.replace("...tmd.../punish...")`
   redirect). The library detects both. A browser "passes" the HTML form
   partly because it can follow the redirect and solve the interactive
   slider; a script never can.

## What the library replicates

- **Zero-setup browser import (v0.2.0+)**: `QwenStudio.from_browser()`
  reads the complete cookie jar straight out of a local Firefox /
  Chrome / Chromium / Brave / Edge profile on Linux (details in
  [browser-session.md](browser-session.md)) - the session token *and* the
  anti-bot set the browser earned, with no manual export step. This is the
  recommended constructor.
- **Full cookie jar support**: pass the browser's cookies via
  `extra_cookies=...`, or load a Playwright storage-state export with
  :meth:`qwen_studio.QwenStudio.cookies_from_browser_state`::

      jar = QwenStudio.cookies_from_browser_state("state.json")
      q = QwenStudio.from_credentials(email, password, extra_cookies=jar)

  Every request sends session token + your jar. The jar - not the password
  - is what keeps completions flowing.
- **Transport impersonation (default since v0.2.0)**: every request is
  sent through a Chrome-impersonated `curl_cffi` session (real Chrome
  TLS/HTTP2), and `curl_cffi` is a hard dependency - script traffic is
  browser-shaped from the first request instead of acquiring that shape
  after being punished. Experiment 2 shows the jar is the deciding
  variable; the impersonation removes the remaining free differentiator
  from the wire. Disable (not recommended) with
  `QwenStudio(..., impersonate=None)`, which emits a `RuntimeWarning`.
- **Cadence shaping**: consecutive calls are automatically spaced
  (`min_interval=4.0s` + up to `jitter=1.5s`), matching the behavioural
  half of the risk score. Tune with
  `QwenStudio(..., min_interval=0, jitter=0)`.
- **Typed, correct failure**: both punish body shapes are detected and
  raised as `PunishedError` (including the HTML x5sec redirect that plain
  `.text` reading misses on streamed responses). Stop and wait minutes;
  do not retry into an elevated score.
- **The continuation gate stands**: none of this changes the server-side
  rejection of external tool-result continuations
  ([local-tools.md](local-tools.md#server-side-gating-of-continuations)).
  That gate is application logic, not the edge.
