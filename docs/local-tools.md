# Custom tools: wrapping Python callables into the MCP protocol

This is the library's core trick: the **client-side MCP mechanism**
(`local_mcp`) — in practice the desktop app's way of running its own tool
servers — turned into a wrapper that makes any Python function a
chat-callable tool.

## Quickstart

```python
from qwen_studio import QwenStudio, tool

q = QwenStudio.from_credentials("you@example.com", "password")
MODEL = q.list_model_ids()[0]
chat = q.chats.create(MODEL)

@tool
def get_weather(city: str):
    """Get the current weather for a city"""
    return {"city": city, "temperature": "26C", "condition": "Cloudy"}

session = q.local_tools(chat.id)     # one session per chat
session.register(get_weather)        # schema auto-derived from the signature

turn = session.ask("What is the weather in Shanghai?")
print(turn.text)
```

The JSON schema is derived from type hints (`str→string`, `int→integer`,
`float→number`, `bool→boolean`, defaults become descriptions). Pass
`input_schema=` to `register()` for full control.

## What actually goes over the wire

**1 — Declaration.** The user message's `feature_config.local_mcp` carries
your schemas, shaped exactly like registry entries:

```json
{"LocalTools": {
  "get_weather": {
    "description": "Get the current weather for a city",
    "input_schema": {"type": "object",
                     "properties": {"city": {"type": "string"}},
                     "required": ["city"]}
  }}}
```

**2 — Invocation.** The model streams a delta with `phase = local_tool`
and `extra.local_mcp` describing the call:

```json
{"LocalTools": [{"tool_name": "get_weather", "params": {"city": "Shanghai"}}]}
```

**3 — Execution.** Your Python callable runs locally
(`ToolExecutionError` is converted into the protocol's failure string so
the model sees it).

**4 — Continuation.** Results return as a follow-up completion whose single
message has `role: "function"`, serialised exactly like the official
client: clone of the response turn minus `id`/`parentId`/`content_list`/
`user_action`, `mcp`/`local_mcp` stripped from the feature config, and the
results string in the content:

```json
{"LocalTools": [{"get_weather": "{\"city\": \"Shanghai\", ...}"}]}
```

with the top-level `parentId`/`parent_id` pointing at the tool-emitting
response id. The loop repeats (`max_loops=3` default) if the model calls
more tools, then the final answer is returned.

## Server-side gating of continuations

Honest boundary, established by live experiment (this answers the report's
open question #3):

- **Declaration + invocation are confirmed working** end-to-end from an
  external client: the model happily calls externally-declared tools, on
  both the `web` and `desktop` request surfaces.
- **The result continuation is gated.** From the `web` surface, the Alibaba
  edge risk engine intercepts it (anti-bot punish page, `RGV587`). From the
  `desktop` surface it reaches the application, which answers
  `Bad_Request: "Something wrong with request!"` for the bundle-faithful
  shapes (both the hand-built variant and the true stored-message clone)
  and `Internal_Server_Error` when the parent linkage is missing.

The official desktop client evidently completes this loop — the shapes
above are transcribed from its own serialisation code — so the server
requires something an external session has not replicated (session-bound
turn state or client recognition). When that gate is hit, the library
raises `ContinuationBlockedError` carrying the server's exact reply and
your executed `results`, so instrumented clients can adapt. Everything up
to and including local execution works today; only the final "feed results
back" step is subject to the gate.

## Manual control

Skip `ask()` and drive the loop yourself:

```python
res = session.send("What's the weather in Shanghai?", MODEL)   # step 1
print(res.tool_calls)                                          # requests

results = session.execute_calls(res.tool_calls)                # step 2

res2 = session.send_tool_results(MODEL, res.response_id, results)  # step 3
print(res2.answer)
```

`session.tool` also works as a decorator for inline registration, and
`register("alias_name", fn)` lets you rename a tool for the model. The session
tracks the last upstream response id, so subsequent `send()` calls are linked
as real child turns instead of edits.
