# qwen-studio

A reverse-engineered, live-verified Python client for the **Qwen Studio**
web API (`chat.qwen.ai`) - plus a CLI that turns it into a self-hosted,
**well-behaved OpenAI-compatible API**.

Built from a black-box protocol study of the web client (v0.3.11): every
endpoint, header and message shape here was captured from the official
client, replicated externally, and verified against the live service.

## Quick start: the CLI (v0.5.0)

```bash
pip install .
pip install ".[warmup]"     # optional: enables the headless-browser warmup
playwright install chromium # one-time browser install for the warmup

# Path A: pull the complete cookie jar from your local browser
qwen-studio login          # writes ./qwen.auth.json (cookies + token)

# Path B: env-var credentials + warmup (headless servers / CI / no browser)
export QWEN_EMAIL=you@example.com
export QWEN_PASSWORD=...
qwen-studio login          # signs in, runs warmup, writes ./qwen.auth.json

# then serve a self-hosted OpenAI-compatible API wrapping Qwen Studio
qwen-studio serve --port 8080 --api-key sk-my-secret
#   system prompts (server-side enforced), tools via the MCP wrap,
#   image/file uploads, 1-hour in-memory conversation history: follow-ups
#   continue in the same Qwen project + chat, unseen histories are replayed,
#   one-shot conversations can be explicitly reaped early after completion; by default the upstream chat/project live for the full session TTL.
```

Any OpenAI client works: `OpenAI(base_url="http://127.0.0.1:8080/v1")`.
Details in [docs/openai-server.md](docs/openai-server.md).

## Library quick start

```python
from qwen_studio import QwenStudio

q = QwenStudio.from_browser()           # cookies straight from your local
                                        # Firefox / Chrome / Chromium profile
model = q.list_model_ids()[0]           # model ids rotate - use the catalogue

chat = q.chats.create(model)            # create a chat
turn = q.chat.send(chat.id, "Hello!", model)
print(turn.text)                        # streamed via the SSE phase machine

q.chats.delete(chat.id)                 # delete it again
```

Log in to chat.qwen.ai in your browser once, and the script above carries
the same session material the browser does - no token copying, no exports.
Every other constructor (`from_credentials`, `from_session_token`,
`from_access_token`) still works unchanged.

## Features

- **CLI + OpenAI-compatible proxy (v0.5.0)** — `qwen-studio login` (browser
  cookies → auth file in the current directory; or env-var credentials
  + headless-browser warmup when no browser profile exists) and
  `qwen-studio serve` (`/v1/chat/completions` streaming + non-streaming,
  `/v1/models`, `/v1/files`; system prompts via the project mechanism,
  OpenAI `tools` through the MCP wrap, image/file content parts, 1-hour
  in-memory conversation history - continuations (incl. tool results)
  stay in the same Qwen project + chat, with lenient matching, forks,
  revival of dead chats and engineered replay of unseen histories; one-shot
  conversations are deleted upstream; see
  [docs/openai-server.md](docs/openai-server.md))
- **Browser session import (v0.2.0)** — pull the complete cookie jar
  (session token + anti-bot set) directly from Firefox, Chrome, Chromium,
  Brave or Edge profiles on Linux, incl. snap/flatpak paths, locked-DB
  safety and v10/v11 cookie decryption (`q = QwenStudio.from_browser()`;
  see [docs/browser-session.md](docs/browser-session.md))
- **Headless-browser warmup (v0.5.0)** — when no local browser profile
  exists (headless servers / CI / containers) and the caller authenticates
  with `QWEN_EMAIL` + `QWEN_PASSWORD`, the library runs a short
  headless-browser session (Playwright or the `agent-browser` CLI) that
  loads the SPA with the session token pre-injected, lets Aliyun's JS
  beacons mint the anti-bot cookies (`cna`, `tfstk`, `isg`,
  `ssxmod_itna*`, ...), and merges the resulting jar into the client.
  This is the fix for the "thin-jar" failure mode that the original
  `from_credentials()` path produces — without the warmup, the session
  carries only the four cookies `/auths/signin` itself returns
  (`token`, `acw_tc`, `x-ap`, `refresh_token`) and gets punished on
  `/chat/completions` after a while. See
  [docs/anti-bot.md](docs/anti-bot.md) and
  [docs/authentication.md](docs/authentication.md#warmup).

- **Chat lifecycle** — create, list, read, delete, batch-delete, pin,
  archive, rename (`q.chats`)
- **Streaming completions** — the real phase machine (thinking summaries,
  tool phases, answer), parsed into typed events (`q.chat`)
- **Hosted MCP tools** — discover the platform's MCP registry, toggle
  servers for your account, chat with them (`q.tools`, `mcp_enabled=True`)
- **Custom tools via MCP wrapping** — turn any Python callable into a
  chat tool through the client-side `local_mcp` mechanism, with the full
  tool-call loop implemented (`q.local_tools(chat_id)`)
- **Images** — STS + direct-OSS upload (no OSS SDK needed) and
  multi-image turns; verified live with **7 images in one turn** — the
  UI's 5-image cap is client-side only (`q.files`, `files=` on send)
- **System prompts** — the official server-side mechanism: projects with
  `custom_instruction` (`q.projects.system_chat(...)`); a `role:system`
  message in the request is provably ignored
- **Robust auth** — 30-day session token → 900-second access token
  lifecycle managed automatically; cookie/Bearer auth split handled
- **Anti-bot resilience, by construction** — browser cookie-jar import +
  **default** Chrome TLS/HTTP2 impersonation (curl_cffi is a hard
  dependency, so script traffic is browser-shaped from request #1) +
  built-in request pacing; the RGV587/x5sec risk engine is dissected in
  [docs/anti-bot.md](docs/anti-bot.md)
- **Typed errors** — quota, rate-limit, punish (anti-bot), not-found and
  bad-request all surface as distinct exceptions

## Installation

```bash
pip install .
# requires: python >= 3.9; curl_cffi (Chrome transport) and cryptography
# (cookie decryption) are core dependencies

pip install ".[keyring]"    # + libsecret lookup for Chrome-family cookie keys
pip install ".[warmup]"     # + Playwright for the headless-browser warmup
                            #   (run `playwright install chromium` once)
                            #   - needed when no local browser profile exists
                            #   - alternative: install agent-browser CLI
pip install ".[fallback]"   # + plain requests fallback (impersonate=None) - not recommended
```

## Documentation

| Doc | Contents |
|---|---|
| [docs/index.md](docs/index.md) | overview, status of verification, responsible use |
| [docs/browser-session.md](docs/browser-session.md) | cookie import from Firefox/Chrome/Chromium on Linux, the zero-setup constructor |
| [docs/authentication.md](docs/authentication.md) | token lifecycle, header contract, what is (not) validated |
| [docs/chats.md](docs/chats.md) | create / list / read / delete chats |
| [docs/streaming.md](docs/streaming.md) | SSE phases, event model, anti-bot punish |
| [docs/mcp-tools.md](docs/mcp-tools.md) | hosted MCP registry + activation + chat |
| [docs/local-tools.md](docs/local-tools.md) | custom tools via MCP wrapping, incl. the server-side gate |
| [docs/capabilities.md](docs/capabilities.md) | live-proven capability study: system prompts, images, history injection, limits |

Examples in [`examples/`](examples/): browser-session import, images & system prompts, create-and-
delete chat, streaming chat, hosted-MCP chat, custom tools.

## The two MCP mechanisms

A key finding of the underlying study — Qwen Studio runs **two distinct
tool systems**:

1. **Hosted MCP (server-side).** Tool availability is per-user *server
   state*, not request content. You toggle servers; the backend attaches
   and executes tools itself; results stream back as content phases.
2. **Client-side MCP (`local_mcp`).** The desktop app's mechanism: the
   client declares tool schemas inline, the model emits `local_tool` calls,
   the client executes them and feeds results back with a
   `role:"function"` continuation. This library wraps step 3 with your
   Python callables — declaration and invocation are verified working
   externally; the continuation step is server-gated for unrecognised
   clients (details in
   [docs/local-tools.md](docs/local-tools.md#server-side-gating-of-continuations)).

## Errors worth knowing

```python
from qwen_studio import exceptions as qe

try:
    turn = q.chat.send(chat.id, "...", model)
except qe.RateLimitedError:      # RateLimited / ParallelLimited / Too_Many_Requests
    ...                          # back off
except qe.QuotaError:            # quotaLimited / ExceedLimit / quota_exhausted
    ...
except qe.PunishedError:         # anti-bot risk engine (RGV587) - STOP, wait minutes
    ...
except qe.NotFoundError:         # e.g. chat already deleted
    ...
```

## Scope and responsible use

This project is an **educational protocol analysis**, not an automation
framework. Automated use of the web endpoints sits outside Alibaba's terms
of service; the supported path for programmatic access is the official
DashScope / OpenAI-compatible API with its own key management. Keep any
experimentation to personal, low-volume use — the anti-bot risk engine
enforces this for you otherwise.

Never hardcode model ids (they rotate), never share session tokens, and
rotate any password you have used in plaintext experimentation.

## License

MIT — see [LICENSE](LICENSE).
