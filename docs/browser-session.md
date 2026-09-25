# Browser session import: cookies straight from your profile

The anti-bot study (see [anti-bot.md](anti-bot.md)) proved that
`/chat/completions` is gated on the **complete browser cookie jar** - the
session `token` plus the edge cookies the browser accumulates (`acw_tc`,
`tfstk`, `isg`, `ssxmod_itna*`, `cna`, ...). Since v0.2.0 the library can
read that jar **directly from your local browser profile on Linux**, so the
zero-setup setup is literally:

```python
from qwen_studio import QwenStudio

q = QwenStudio.from_browser()               # auto-detect, recommended
# q = QwenStudio.from_browser("firefox")    # or "chrome", "chromium",
# q = QwenStudio.from_browser("brave"), ("edge")
```

The browser must be logged in to `chat.qwen.ai` once. From then on the
client presents exactly the session material the browser presents:

- the session `token` cookie (30-day),
- the full anti-bot jar the risk engine expects,
- a real Chrome TLS/HTTP2 transport fingerprint (curl_cffi impersonation
  is the **default** transport since v0.2.0 and curl_cffi is a hard
  dependency - script traffic is born browser-shaped, not punished into
  becoming browser-shaped),
- human-paced request cadence (`min_interval` + `jitter`).

## How extraction works per browser

| Browser | Store | Decryption |
|---|---|---|
| Firefox | `cookies.sqlite` (SQLite) | none needed |
| Chrome / Chromium / Brave / Edge | `.../Default/Network/Cookies` (SQLite) | `v10`: AES-128-CBC with PBKDF2(keyring password or `"peanuts"`, salt `saltysalt`, 1 iter, 16 bytes); `v11`: AES-256-GCM with the key in `Local State` (`os_crypt.encrypted_key`) |

- Search locations include the native, **snap** (`~/snap/...`) and
  **flatpak** (`~/.var/app/...`) roots for every supported browser.
- Databases are **copied to a temp dir before reading**, so extraction
  works while the browser is running and never touches the profile.
- The libsecret (GNOME) / kwallet (KDE) "… Safe Storage" password is
  looked up best-effort; locked keyrings are skipped, never prompted. If
  your Chrome uses the basic password store, the well-known `"peanuts"`
  fallback key applies.
- Only cookies for the requested domain (default `qwen.ai`) are returned;
  everything else is filtered out before it ever reaches your process.

## Profile selection

With no arguments, every supported profile is scanned and the winner is
the most recently used one that holds a `qwen.ai` `token` cookie (cookie
database mtime as the recency proxy). To pin one:

```python
q = QwenStudio.from_browser("chromium", profile="Profile 1")
```

Inspect what the module can see:

```bash
python -m qwen_studio.browser_cookies                  # all browsers, masked
python -m qwen_studio.browser_cookies --browser firefox --show-values
```

The CLI prints profile paths and cookie names with **masked values**;
`--show-values` prints secrets, so use it only on private machines.

## What from_browser wires up

```python
q = QwenStudio.from_browser()
q.session_token    # the browser's 30-day token
q.extra_cookies    # token excluded - the anti-bot jar (acw_tc, tfstk, ...)
q.cookie_source    # e.g. "firefox:default-release (/home/.../cookies.sqlite)"
```

`auto_refresh` defaults on: right after construction the 30-day cookie is
exchanged at `auth.qwen.ai` for a fresh 900-second Bearer token. If that
call fails (offline, expired cookie), construction still succeeds - the
client remains cookie-capable and every bearer-authenticated call will
surface the problem as a typed `AuthError`.

## Lower-level access

```python
from qwen_studio import browser_cookies as bc
from qwen_studio import load_browser_cookies, qwen_cookie_dict

prof, cookies = bc.find_qwen_jar("chrome")       # (BrowserProfile, [BrowserCookie])
jar = qwen_cookie_dict()                          # {"token": ..., "acw_tc": ...}
q = QwenStudio.from_session_token(jar.pop("token"), extra_cookies=jar)
```

Playwright users who already export `storage_state()` JSON files can keep
doing that - `QwenStudio.cookies_from_browser_state("state.json")` returns
the same flat dict.

## Troubleshooting

- **"no supported browser profile found"** - the module looks in
  `~/.mozilla/firefox`, `~/.config/google-chrome`, `~/.config/chromium`,
  `~/.config/BraveSoftware/Brave-Browser`, `~/.config/microsoft-edge` and
  their snap/flatpak equivalents. A non-standard `--user-data-dir` (or a
  Windows/macOS machine) is not supported; fall back to
  `from_session_token` or a Playwright state export. On headless servers,
  set `QWEN_EMAIL` + `QWEN_PASSWORD` and `from_browser()` will
  automatically fall back to `from_credentials(..., warmup=True)` (see
  [authentication.md#warmup](authentication.md#warmup)).
- **"could not be decrypted (keyring locked or empty)"** - the browser
  stores its cookie key in your desktop keyring. Unlock it, or install
  the `keyring` extra (`pip install ".[keyring]"`) so libsecret lookup
  works. Firefox users never hit this.
- **"none holds a qwen.ai session token"** - the profile exists but is
  not logged in. Open `chat.qwen.ai` in that browser once.
- **Chrome 127+ / v11 cookies** - handled via the `Local State` key. If
  your build stores keys elsewhere and decryption fails, file the exact
  browser version; `v20` (Windows app-bound) is out of scope by design.

## Security notes

- Reading cookies means handling a **full account credential**. The
  module never writes cookies back, never logs values, and the CLI masks
  values by default.
- The extracted jar is held in memory by the client for the session only;
  it is not persisted anywhere by the library.
- Cookies are sent only to the qwen.ai hosts the API talks to
  (`chat.qwen.ai`, `auth.qwen.ai`).
