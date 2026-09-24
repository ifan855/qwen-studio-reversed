# Capability study: system prompts, images, history injection, OpenAI shape

This page answers five questions posed about the API surface, each settled
by controlled live probes (paced, full browser jar + Chrome transport;
probe sources under `scripts/` in the study workspace). Findings were
implemented as library functions where the API proved permissive, and
documented as boundaries where it did not.

## Q1 - System prompts: YES, but only via projects (server-side)

The completions endpoint does **not** honour a `role: "system"` message in
the request:

| Attempt | Result |
|---|---|
| `messages: [system, user]`, first turn, `parent_id=None` | stream opens (`response.created`), then **ends with no answer** |
| same but `parent_id=<system fid>` | rejected: `PARENT_NOT_FOUND` - the server validates `parent_id` against its own stored tree |

The official mechanism is **projects**: a project carries a
`custom_instruction`, and every chat created inside it (via
`POST /chats/new` with `project_id`) has that instruction applied by the
backend to every turn. Verified live with a token-obedience test: the
instruction never appears anywhere in the wire traffic, yet the model
follows it:

> project `custom_instruction`: "every reply MUST end with the token
> PROJ-8842" -> model answer to "What is 3+3?": **"6 PROJ-8842"**

Library API:

```python
chat = q.projects.system_chat(model, "Always answer in French.")
turn = q.chat.send(chat.id, "Hello!", model)
# cleanup
q.chats.delete(chat.id)
q.projects.delete(chat.raw["project_id"])
```

## Q2 - Images: YES, and more than 5 in one turn

Uploads use a two-step pipeline (the old `POST /files/` route is dead):

1. `POST /files/getstsToken` with `{filename, filesize (string), filetype}`
   -> STS credentials + pre-registered object (`file_id`, `file_path`,
   `file_url`).
2. Direct `PUT https://{bucket}.{endpoint}/{file_path}` with OSS signature
   V1 + `x-oss-security-token` (implemented in the library with stdlib
   HMAC only).

Messages then reference the upload via a `files` entry. Verified live:

| Probe | Result |
|---|---|
| 1 image (solid red) + "what color?" | **"Red"** |
| **7 images** in one turn (6 colors + text "BANANA-77") | **accepted**; model answered "I see 7 images, and the text \"BANANA:77\" appears in one of them" |

The well-known **"5 images" limit is a client-UI cap**
(`vision: {max_count: 5}` in the web bundle), not an API one. Client
bundle caps (server-overridable via config): images 5 x 20 MB, documents
5 x 20 MB, video 1 x 500 MB, audio 1 x 100 MB.

Library API:

```python
ref = q.files.upload_image(png_bytes, "chart.png")
turn = q.chat.send(chat.id, "Explain this chart.", vis_model,
                   files=[ref.entry()])
```

Vision turns need a vision-capable model - pick one containing
`omni`/`vl` from the catalogue, never hardcode.

## Q3 - Unseen conversation history: rejected two ways

- **Via the messages array: impossible.** The server enforces
  `Invalid input too many messages` for a 3-message body - effectively
  **one message per completion request**; history lives server-side and
  `parent_id` must reference a node the server already knows
  (`PARENT_NOT_FOUND` otherwise).
- **Via `POST /chats/import`: accepted, then voided.** The import
  endpoint (the web client's backup-restore path) validates against a
  strict Open-WebUI-style record
  (`{id, user_id, title, chat: {history, messages, models}, created_at,
  updated_at, archived, ...}` - the exact schema was recovered from its
  own `ValidationError` messages). A **fully fabricated, never-happened
  history** (4 messages, fake uuids, canary word) was accepted:
  `data.success` listed a server-assigned chat id, and a `GET /chats/{id}`
  seconds later showed **all four fabricated messages persisted**.
  However, the chat is then flagged deleted almost immediately:
  `CHAT_NOT_FOUND: This chat has been deleted. Please start a new chat to
  continue.` - completions against it are refused. The imported chat
  cannot serve as a stable base for continuations (observed consistently,
  including with the continuation fired immediately after import).

The library implements `q.chats.build_import_record()` +
`q.chats.import_history()` for completeness (and backup-restore use), with
this boundary documented in the docstrings.

## Q4 - OpenAI compatibility: NO (not natively)

The endpoint is a bespoke envelope, not an OpenAI-compatible surface:

- A bare OpenAI-shaped body (`{model, messages, stream}`) was **punished
  by the risk engine twice out of two attempts** (the RGV587/x5sec HTML
  redirect), alongside the envelope errors it would need anyway
  (`chatId`, `chat_type`, `models`, `fid`-style nodes, etc.).
- Responses are the custom SSE phase machine (`response.created`,
  `thinking_summary`, tool phases, `answer`), not OpenAI chunks.
- History is server-side; the API is chat-centric, not stateless
  messages-in/messages-out.

Per the study's decision rule this means: **extend the library with
functions** (done, above) rather than build a passthrough OpenAI proxy.
(A translating OpenAI-compatible shim remains possible on top of the
library - it would map OpenAI `messages` onto a server-side chat tree and
re-chunk the phase machine - but it is a translation layer, not a
protocol compatibility, and is left out of scope here.)

## Q5 - Limits observed with a browser-proven session

| Dimension | Observed limit |
|---|---|
| Messages per completion request | 1 (server-enforced; >1 -> `Bad_Request: Invalid input too many messages`) |
| Parent linkage | `parent_id` must exist server-side (`PARENT_NOT_FOUND`) |
| Images per turn | at least 7 accepted (UI caps at 5); 20 MB each (client config) |
| Text payload | 32 KB verified fine (canary retrieved verbatim) |
| Video / audio / docs | client config: 1 x 500 MB / 1 x 100 MB / 5 x 20 MB |
| Behavioural | risk engine punishes malformed/foreign bodies (bare OpenAI shape: 2/2 punished) and bare cookie jars; pacing (`min_interval` + `jitter`) keeps the score quiet |
| Model access | catalogue-driven; ids rotate (`q.list_model_ids()`), vision needs an `omni`/`vl` model |

## Probe log

Experiments were run as paced scripts with structured JSON results
(`capability_results*.json`), each chat deleted after use, and every run
aborted immediately on any `PunishedError` (with a 15-30 min cool-down
before resuming). Static findings came from the v0.3.11 web bundle
(`projects` endpoints, `/chats/import`, `getstsToken` flow, upload caps)
cross-checked against live behaviour.
