# Streaming completions and the phase machine

Qwen Studio's chat pipeline is **not** a plain text delta stream. Each
completion is an SSE stream built around a *phase state machine*; the
client (`q.chat`) parses it into typed events for you.

## One-shot usage

```python
chat = q.chats.create(MODEL)
turn = q.chat.send(chat.id, "Explain MCP in one paragraph.", MODEL)
print(turn.text)              # final answer text
print(turn.response_id)       # server's response message id for this turn
```

`send()` returns a `Turn` wrapping the aggregated `StreamResult`.

## Event-by-event

```python
for ev in q.chat.send(chat.id, "...", MODEL, keep_events=True).result.events:
    if ev.type == "created":
        print("response id:", ev.response_id)
    elif ev.phase == "answer":
        print(ev.content or "", end="", flush=True)
    elif ev.is_tool_phase:
        print(f"\n[tool: {ev.phase}/{ev.status}]")
```

A `ChatEvent` exposes: `type` (`created` / `delta` / `raw`), `phase`,
`status` (`typing` / `finished` / `error`), `content` (incremental text),
`extra` (tool descriptors, thinking summaries, usage) and `raw` (the full
frame).

## The phases you will see

| Phase | Meaning |
|---|---|
| `thinking_summary` | reasoning summary; `extra.summary_title` / `extra.summary_thought` carry the UI content |
| `answer` | the actual reply text, streamed incrementally |
| `web_search` / `search_result` | built-in search tool |
| `tool_call`, `bash`, `code_interpreter`, `fetch_page`, `generate_image`, `edit_file`, `read_file`, `write_file`, `present_file` | built-in platform tools (bundle-enumerated) |
| `local_tool` | client-executed MCP tool call (see [local-tools.md](local-tools.md)) |

Every delta also carries a `status`; a turn ends with an
`answer/finished` phase transition.

## Request envelope

The library builds the exact envelope the web client sends (verified
against captures):

```json
{
  "stream": true, "version": "2.1", "incremental_output": true,
  "chatId": "<id>", "parentId": "", "chat_id": "<id>",
  "chat_mode": "normal", "model": "<id-from-catalogue>", "parent_id": null,
  "messages": [ { "...user message node..." } ],
  "timestamp": 1790212954
}
```

posted to `POST /api/v2/chat/completions?chat_id=<id>` with
`Accept: application/json` and `x-accel-buffering: no` (the latter disables
proxy buffering so frames arrive live).

Useful `send()` options: `thinking=True` enables the thinking phase;
`mcp_enabled=True` attaches the `local_mcp: {}` marker so the backend
applies this account's hosted MCP servers (see
[mcp-tools.md](mcp-tools.md)); `extra_feature_config` merges anything else
into the feature config.

## Anti-bot punish (RGV587)

Beyond the application layer sits Alibaba's edge risk engine. When it
flags your traffic it intercepts requests *before* the app answers, with an
HTTP 200 HTML/JSON page containing:

```
FAIL_SYS_USER_VALIDATE, RGV587_ERROR::SM::..._____tmd_____/punish...
```

The full experiment that isolated what triggers it (a time-decaying risk
score on `/chat/completions`, transport-fingerprint discriminating while
active, endpoint-selective, not cookie-bound) lives in
[anti-bot.md](anti-bot.md). The library counters it with Chrome TLS
impersonation via `curl_cffi` and automatic request pacing. If it still
fires, the library raises `PunishedError` — the only correct reaction is to
**stop and wait minutes**; retrying deepens the block.
