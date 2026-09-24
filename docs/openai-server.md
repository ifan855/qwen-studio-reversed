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
| `--oneshot-ttl` | -1 | optional early cleanup for an idle *single-turn* conversation's chat + project; disabled by default so the same project/chat survives for the full session TTL; history is retained until the session expires |
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

Errors are OpenAI error objects; upstream states map to 429 (rate/quota)
and 503 (anti-bot punish) so standard SDK retry logic behaves.

## How the mapping works

### System prompts (accurate)

The completions endpoint ignores `role:"system"` messages (live-verified,
docs/capabilities.md). The proxy therefore creates every conversation that
carries a system message **inside a fresh project whose
`custom_instruction` is your system prompt** - the official server-side
mechanism. The instruction is enforced by the Qwen backend and never
appears on the wire; a token-obedience test through the proxy confirmed it.

### The 1-hour conversation memory and automatic routing

Every served conversation is kept in memory with a TTL (default 3600 s;
`--ttl`): its full OpenAI message history **plus the upstream project and
chat it lives in**. On each request the proxy canonicalises the incoming
`messages` and scores them against *every* stored conversation (not just
the most recent one) - exact/retry beats continuation, a longer stored
history beats a shorter one, an unchanged system prompt beats a changed
one, recency breaks ties.

Canonicalisation is deliberately lenient about what clients do when they
echo our replies back: `developer` = `system`; `"text"` =
`[{"type":"text","text":"text"}]`; `null` = `""`; CRLF and surrounding
whitespace ignored; tool-call ids ignored (tool results are matched by
position) and tool-call arguments compared as JSON values, not strings;
`reasoning_content`, `name` and other decorations dropped.

| incoming history | action |
|---|---|
| exact match of a stored history, or everything up to just before the stored reply (stateless retry) | re-serve the stored reply (no upstream call) |
| stored history + anything new (user turn, **tool results**, client-added messages) | **continue in the same project + chat**: only the new messages go upstream, as one turn |
| same dialog, different system prompt | the conversation's project instruction is updated in place, same chat continues (falls back to a new chat if that is impossible) |
| shares a prefix, then diverges (edited / regenerated earlier turn) | **fork**: new chat in the *same project*, history replayed; the original conversation is untouched |
| never seen | **replay mode** (below) |

If a stored conversation's chat can no longer take a turn - it was explicitly
reaped by `--oneshot-ttl`, deleted, rejects the continuation (`NotFound` /
`Bad_Request`), or failed twice in a row - the conversation is **revived**:
a fresh chat is opened (in the same project when it still exists), the
history is replayed into it, the dead chat is deleted, and from then on it
continues natively again. With the default configuration, the chat is not
reaped early and remains the canonical continuation target until the session
TTL expires. Transient upstream states (rate limit, quota,
anti-bot) never trigger this: they are surfaced so the client retries
against the same chat.

A continuation whose stream is interrupted (client disconnect, upstream
error) no longer destroys the conversation: nothing is committed to
memory, the chat is flagged, and the retry goes to the same chat with a
short note that the cut-off reply never reached the user.

### Replay mode (unseen histories)

A fresh Qwen conversation is created (inside a project when a system prompt
is present, so system-prompt fidelity survives). The history before the
latest turn - every message, tool call and tool result - is rendered into
`conversation-history.md`, uploaded through the file pipeline, and *also*
inlined into the engineered prompt (mode `both`; `--replay file|inline`
selects file-only / inline-only). The latest turn (user message and/or tool
results) follows it. Verified live: a fabricated 3-turn history
("favourite colour teal, cat named Miso") was honoured exactly. If the file
upload fails, the proxy falls back to inline-only automatically.

### Tools through MCP

OpenAI `tools` function definitions are declared to Qwen as client-side
MCP (`local_mcp`) tools - the desktop app's mechanism. When the model
invokes one, the proxy replies with standard OpenAI
`finish_reason:"tool_calls"`; your client executes the tool and posts the
`role:"tool"` result back. The upstream `role:"function"` continuation is
server-gated (docs/local-tools.md), so the results are delivered to the
**same chat** as the next turn (named after the calls that requested them,
tools re-declared so the model can call again). If the chat rejects that
turn, the conversation is revived through a replay automatically.

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
  proxy behaves like one careful browser session. The lock is released
  exactly once per streamed reply - on completion, on client disconnect or
  on stream abandonment - so a dropped connection can never wedge the
  server.
- A first-message stream that dies mid-flight (client reset, timeout) never
  poisons the conversation memory: the exact resend re-runs the turn
  instead of being served an empty cached reply.
- Websocket probes (`GET /ws` and any `Upgrade:` handshake) are answered
  with a clean JSON 404 pointing at `POST /v1/chat/completions` (SSE) and
  the connection is closed immediately - probing clients cannot leave
  half-read keep-alive sockets that reset and spam `ConnectionResetError`
  tracebacks. Routine disconnects are swallowed silently.
- **Session lifetime is authoritative.** A single-turn conversation keeps its
  upstream chat and project alive for the same session lifetime as any other
  conversation, so later turns can continue in the exact same Qwen project/chat
  instead of being forced through replay. The normal session TTL (default
  3600s) deletes the upstream resources when the remembered conversation
  expires. `--oneshot-ttl` can explicitly opt into earlier cleanup, but its timer starts
  only after the response has been fully delivered; an in-flight answer is
  never reaped. The in-memory history still survives until the main TTL. A
  first turn that is aborted or fails is deleted immediately.
- Expired / evicted conversations are deleted upstream (chat, then the
  project once no fork of the conversation still uses it).
- Failed deletions are queued and retried by the sweeper (up to 5 tries)
  instead of leaking; Ctrl-C (`service.shutdown()`) deletes everything
  still held upstream.

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
