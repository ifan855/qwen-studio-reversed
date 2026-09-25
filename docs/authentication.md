# Authentication and the token lifecycle

Qwen Studio uses a **two-tier token model** across two hosts:

| Host | Role |
|---|---|
| `chat.qwen.ai` | serves the SPA and the entire `/api/v2` application interface |
| `auth.qwen.ai` | dedicated auth service owning the refresh exchange |

```
password ──signin──▶ session token (30 d, cookie "token")
                          │
                          └──refresh──▶ access_token (900 s, Bearer)
                                        refresh_token (30 d, rotates on use)
```

## 1. Sign-in

`POST /api/v2/auths/signin` with the **SHA-256 hex digest** of the password
(the plaintext never crosses the wire, but the digest is itself a
password-equivalent credential — one more reason not to leak it):

```json
{"email": "you@example.com", "password": "<sha256-hex>"}
```

Response `data.token` is the 30-day session token. It is normally stored as
the cookie `token` on `.qwen.ai`.

## 2. Refresh

`GET https://auth.qwen.ai/api/v2/auths/refresh` — the one call that goes
cross-host with **cookie** credentials (plus an `x-request-origin` header).
It returns a short-lived `access_token` (900 seconds, JWT with
`type: access_token`) and a rotated 30-day `refresh_token`.

## 3. Using the tokens

Every application endpoint takes `Authorization: Bearer <access_token>`.
One known exception, verified live: `GET /mcp/list` authenticates via the
session **cookie** only and rejects Bearer headers. The library handles
this split for you (`cookie_auth=True` on that call).

## What the client does for you

```python
from qwen_studio import QwenStudio

# full lifecycle: signs in, refreshes, and auto-renews the access token
# ~30 s before expiry on every subsequent call
q = QwenStudio.from_credentials("you@example.com", "password")
```

The four constructors cover every situation:

| Constructor | When to use | Renewal |
|---|---|---|
| `from_browser(...)` | **recommended** on desktops — you are logged in to chat.qwen.ai in a local Firefox/Chrome/Chromium profile (Linux) | full; also carries the complete anti-bot cookie jar (see [browser-session.md](browser-session.md)) |
| `from_credentials(email, pw, warmup=True)` | headless / server / CI — signs in and runs a headless-browser warmup that mints the complete anti-bot jar | full (auto re-sign-in, re-warmup) |
| `from_credentials(email, pw)` | low-volume use only — the four-cookie signin jar is eventually punished on `/chat/completions` | full (auto re-sign-in) |
| `from_session_token(token)` | you exported the cookie from a browser | auto refresh while valid |
| `from_access_token(token)` | you hold a live 15-min token only | none — it will expire |

`QwenStudio.ensure_access_token()` is called internally before every
authenticated request: if the cached access token is within its safety
margin of expiry, it refreshes (or re-signs-in when only credentials were
given) and then proceeds. You never schedule refreshes yourself.

<a id="warmup"></a>
## Warmup: the missing half of `from_credentials`

The `/auths/signin` response returns the 30-day `token` and the WAF cookie
`acw_tc` (and the `x-ap` routing cookie). That is enough for management
endpoints (`/models/`, `/chats/create`, `/files/...`) but **not enough for
`/chat/completions`** — the risk engine expects the *complete* `.qwen.ai`
cookie jar a real browser accumulates (`cna`, `tfstk`, `isg`,
`ssxmod_itna*`, ...). Those cookies are minted by Aliyun's edge and
tracking infrastructure via JavaScript beacons that fire when the SPA
boots; a pure-HTTP client (even one with a perfect Chrome TLS fingerprint)
cannot synthesise them. See [anti-bot.md](anti-bot.md) for the controlled
experiment that proved the jar is the deciding variable.

The warmup is the fix. It launches a headless Chromium for a few seconds
with the session token pre-injected as a cookie, lets the SPA's JS beacons
fire, then extracts and merges the resulting jar. Two backends are
supported:

1. **Playwright** (preferred): install with `pip install qwen-studio[warmup]`
   and run `playwright install chromium` once.
2. **`agent-browser`** CLI (fallback): install with
   `npm install -g agent-browser && agent-browser install`.

```python
# explicit warmup
q = QwenStudio.from_credentials("you@example.com", "password", warmup=True)
# force a specific backend
q = QwenStudio.from_credentials("you@example.com", "password",
                                warmup=True, warmup_backend="playwright")
# or: warmup_backend="agent-browser"

# runtime warmup on an existing client (e.g. after a manual signin)
q.signin()
q.do_warmup()
```

The CLI runs warmup automatically when it would otherwise have a thin jar
(no local browser profile, env-var credentials supplied):

```bash
export QWEN_EMAIL=you@example.com
export QWEN_PASSWORD=...
qwen-studio login                # signs in, warms up, saves the full jar
qwen-studio serve                # serves /v1/chat/completions safely
```

The warmup is also re-run automatically when `ensure_access_token()` has
to re-sign-in (after the 30-day session expires), so long-running servers
stay safe across month boundaries.

## Header contract

The official client sends this header set with every request:

| Header | Value | Validated? |
|---|---|---|
| `source` | `web` / `h5` / `desktop` | **no** — live-probed, all mutations accepted |
| `version` | client version, e.g. `0.3.11` | **no** |
| `timezone` | JS `Date().toString()` style string | **no** |
| `x-request-id` | UUID | **no** — format not validated |
| `User-Agent`, `Origin`, `Referer` | browser-typical values | not for basic auth |

This answers the report's open question #1: the header trio and the
request-id format are *descriptive metadata*, not an enforced contract, for
ordinary authenticated calls. The library reproduces them anyway for
fidelity — the edge risk engine may still score requests that look nothing
like the official client (see the punish discussion in
[streaming.md](streaming.md#anti-bot-punish-rgv587)).

## Security notes

- The session token, access token and refresh token each grant **full
  account access** until they expire. Treat them like passwords.
- The password digest is replayable; use a throwaway account for
  experimentation and rotate anything you typed into a chat.
- Token renewal rotates the refresh token. If you persist it anywhere,
  persist the *latest* one only.
