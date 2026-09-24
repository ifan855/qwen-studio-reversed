# The OpenAI-compatible proxy + CLI (`qwen-studio serve`)

Version 0.4.0 turns the library into a self-hosted, **well-behaved
OpenAI-compatible API** backed by your Qwen Studio account, with the
browser-equivalent session handling from v0.2.0 (cookie jar + curl_cffi
Chrome fingerprint) doing the anti-bot work under the hood.

## The two commands

```bash
pip install .            # installs the `qwen-studio` console script

# 1) pull the Qwen cookies out of your local browser into ./qwen.auth.json
qwen-studio login                      # auto-detects firefox/chrome/chromium/brave/edge
qwen-studio login --browser firefox --profile default
qwen-studio login --email you@example.com --password '...'   # credentials fallback
qwen-studio login --email ... --password ... --save-password # persist for re-auth

# 2) serve the OpenAI-compatible API (reads ./qwen.auth.json from the CWD)
qwen-studio serve --host 0.0.0.0 --port 8080 --api-key sk-my-secret
```

`login` writes `qwen.auth.json` (mode 0600) into the **current directory**:
the 30-day session token plus the complete anti-bot cookie jar
(`acw_tc`, `tfstk`, `isg`, `ssxmod_itna*`, ...) - exactly the material the
real browser presents (see docs/anti-bot.md for why that jar is the
decisive discriminator).

### `serve` options

| flag | default | meaning |
|------|---------|---------|
| `--host` / `--port` | 127.0.0.1 / 8080 | bind address |
| `--auth-file` | `./qwen.auth.json` | auth file (or `QWEN_EMAIL`+`QWEN_PASSWORD` env) |
| `--api-key` | none | require `Authorization: Bearer <key>` from clients |
| `--ttl` | 3600 | conversation memory window in seconds |
| `--max-sessions` | 256 | LRU cap; evicted conversations are deleted upstream |
| `--replay` | `both` | unseen-history mode: `both`/`file`/`inline` |
| `--thinking` | off | enable Qwen thinking -> `reasoning_content` deltas |
| `--default-model` | first catalogue entry | used when the client's model id is unknown |
| `--min-interval` | 4.0 | upstream request pacing (seconds) |

## Endpoints

- `POST /v1/chat/completions` (also `/chat/completions`) - streaming (SSE)
  and non-streaming, `system`/`developer` messages, `tools`, image/file
  content parts, `stream_options.include_usage`.
- `GET /v1/models` - the live Qwen catalogue (ids rotate; never hardcode).
- `POST /v1/files` - multipart upload (field `file`); reference from chat
  via `{"type":"file","file":{"file_id":"file-..."}}`.
- `GET /v1/files`, `GET /v1/files/{id}` - list/inspect in-memory uploads.
- `GET /health`.
- `GET /` (and `GET /v1`) - a JSON index of the above (handy when the port
  is opened in a browser).

Errors are OpenAI error objects; upstream states map to 429 (rate/quota)
and 503 (anti-bot punish) so standard SDK retry logic behaves. Every error
response is JSON - including protocol-level ones (a malformed request line,
an unsupported method), which `http.server` would otherwise answer with an
HTML error page that OpenAI clients cannot parse.

### Non-API paths and WebSocket probes

The proxy speaks HTTP JSON+SSE only; it has no WebSocket endpoint. Browsers,
chat UIs and preview/health probes nevertheless open `ws://…/ws` (and
`/socket.io`, …) against whatever port they find. Those requests are answered
with an explicit message instead of a bare rejection:

```json
{"error": {"message": "/ws is a WebSocket endpoint, and this proxy does not
            speak WebSockets - it serves the OpenAI HTTP API only
            (GET /health, GET /v1/models, POST /v1/chat/completions, …) …",
           "type": "upgrade_required", "code": "websocket_unsupported"}}
```

Unknown HTTP paths get a 404 that lists the real endpoints. A client that
simply goes away mid-request (a dropped WebSocket probe, a closed tab, a
cancelled stream) is logged in one line - `… went away (ConnectionResetError)
- connection closed` - rather than as a `socketserver` traceback that reads
like a server crash. Real handler bugs still print in full.

### When the machine cannot reach Qwen

If DNS, TLS or egress to `chat.qwen.ai` / `auth.qwen.ai` fails (sandboxes and
locked-down hosts commonly block it), curl reports it as *"Connection closed
abruptly"*. Those failures are typed as `TransportError` and answered with
`502 upstream_unavailable` naming the host, e.g. `cannot reach chat.qwen.ai:
… Connection closed abruptly …` - instead of a generic 500 with a wall of
transport text. The startup catalogue check reports the same and the server
keeps running (the network may come back). If `serve` logs this for every
request, the machine running it has no route to Qwen: run it where
`chat.qwen.ai` is reachable.

## How the mapping works

### System prompts (accurate)

The completions endpoint ignores `role:"system"` messages (live-verified,
docs/capabilities.md). The proxy therefore creates every conversation that
carries a system message **inside a fresh project whose
`custom_instruction` is your system prompt** - the official server-side
mechanism. The instruction is enforced by the Qwen backend and never
appears on the wire; a token-obedience test through the proxy confirmed it.

### The 1-hour conversation memory and automatic routing

Every served conversation's full OpenAI message history is kept in memory
with a TTL (default 3600 s; `--ttl`). On each request the proxy
canonicalises the incoming `messages` (role/content/tool_calls/tool_call_id,
ignoring client decorations like `reasoning_content` or extra fields) and
compares:

| incoming history | action |
|---|---|
| exact match of a stored history | re-serve the stored assistant reply (no upstream call), TTL reset |
| exact match up to just before the stored reply (stateless client retry) | re-serve that reply, TTL reset |
| stored history + new `user`/`tool` tail | **route to the same Qwen conversation**: only the new user turn is sent upstream, TTL reset |
| anything else (rewound, edited, or never seen) | **replay mode** |

### Replay mode (unseen histories)

A fresh Qwen conversation is created (inside a project when a system prompt
is present, so system-prompt fidelity survives). The complete history -
every message, tool call and tool result - is rendered into
`conversation-history.md`, uploaded through the file pipeline, and *also*
inlined into the engineered prompt (mode `both`; `--replay file|inline`
selects file-only / inline-only). The prompt tells Qwen to treat the file
as its own memory and answer only the latest message. Verified live: a
fabricated 3-turn history ("favourite colour teal, cat named Miso") was
honoured exactly. If the file upload fails, the proxy falls back to
inline-only automatically.

### Tools through MCP

OpenAI `tools` function definitions are declared to Qwen as client-side
MCP (`local_mcp`) tools - the desktop app's mechanism. When the model
invokes one, the proxy replies with standard OpenAI
`finish_reason:"tool_calls"`; your client executes the tool and posts the
`role:"tool"` result back. Because the upstream `role:"function"`
continuation is server-gated (docs/local-tools.md), the tool-result turn is
served through **replay mode**: the history document carries the calls and
their results, and the tools are re-declared so the model can call them
again. Verified live end-to-end: `get_vault_code` round-trip returned the
tool result inside the final answer.

### File / image uploads

- `image_url` parts: `data:` URLs and `https:` URLs are pushed through the
  STS+OSS upload pipeline and attached to the turn. The "5 images" cap is
  web-UI only; the proxy allows up to `--max-images` (default 10) per turn
  (7 verified live in docs/capabilities.md).
- `file` parts with `file_id` (via `/v1/files`) or `file_data` (data: URL)
  and `file_url` parts are attached as documents.
- When images are present the proxy automatically picks a vision-capable
  model from the catalogue if the requested one is not (`vl`/`omni`).

### Hygiene

- One upstream turn at a time (global lock + 4 s request pacing), so the
  proxy behaves like one careful browser session.
- Expired/evicted conversations are deleted upstream (chats + project,
  best effort); Ctrl-C cleans up everything still in memory.
- Sessions dropped after upstream failures so retries rebuild cleanly.

## Client examples

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-my-secret")
stream = client.chat.completions.create(
    model="whatever",   # any id; resolved against the live catalogue
    messages=[{"role": "system", "content": "Always answer in rhyme."},
              {"role": "user", "content": "Tell me about the moon."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

Any OpenAI-protocol tool (Codex CLI, Cursor, Cherry Studio, OpenWebUI,
LangChain, ...) can point at `http://127.0.0.1:8080/v1`.

## Curl quick check

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Authorization: Bearer sk-my-secret' \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"ping"}]}'
```
