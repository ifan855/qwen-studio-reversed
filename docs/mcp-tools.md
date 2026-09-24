# Hosted MCP tools (server-side)

Qwen Studio ships platform-hosted MCP servers (maps, weather, web search
enhancers, ...). The central protocol finding of the study: **tool
availability is not declared in chat requests.** Which servers are active
is per-user *server state*; the backend attaches the corresponding tool
schemas itself and executes hosted tools server-side, streaming results
into the phase machine.

## Discover the registry

```python
reg = q.tools.registry()              # GET /mcp/list  (cookie-authenticated!)
for sid, server in reg.items():
    print(sid, server.type, server.tool_names())
```

Auth nuance verified live: this one endpoint authenticates with the
session **cookie** and *rejects* Bearer headers — the inverse of every
other application endpoint. The library handles the split transparently.

## Activate for your account

```python
state = q.tools.enable("amap")        # POST /users/user/settings/update
print(state)                          # {"amap": True} once confirmed

q.tools.disable("amap")               # same endpoint, value False
```

`enable()` re-reads `GET /users/user/settings` until the change is visible
and returns the confirmed map. This ordering safeguard matters: the study
observed scripted runs where the toggle was written and a completion sent
immediately — the model replied "NO TOOL" with no tool markers in the
request, because the server-side state had not been re-read in time.

## Chat with hosted tools

```python
chat = q.chats.create(MODEL)
q.tools.enable("amap")

turn = q.chat.send(chat.id,
                   "Get the coordinates of the Oriental Pearl Tower "
                   "with the maps_geo tool.", MODEL,
                   mcp_enabled=True)
print(turn.text)
```

`mcp_enabled=True` sets `feature_config.local_mcp = {}` — the web client's
marker that MCP mode is on. Note what the request does **not** contain: no
tool list, no server names. The backend fills that in from your account
state.

Tool activity shows up in the stream as phases (`web_search`,
`search_result`, or server-side MCP tool phases) with results embedded as
content — you never execute anything yourself for hosted tools.

## Difference from client-side tools

| | Hosted MCP (this page) | Client tools ([local-tools.md](local-tools.md)) |
|---|---|---|
| Who declares | server, from account state | you, inline in the request |
| Who executes | backend | your Python process |
| Stream marker | server tool phases | `local_tool` + `extra.local_mcp` |
| Result path | streamed by server | `role:"function"` continuation |
