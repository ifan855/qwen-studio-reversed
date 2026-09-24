# qwen-studio documentation

A reverse-engineered, live-verified Python client for the **Qwen Studio**
web API (`chat.qwen.ai`, web client v0.3.11 at the time of the study).

The package exposes the protocol surfaces the underlying analysis
mapped:

| Surface | Module | What it does |
|---|---|---|
| **CLI + OpenAI-compatible proxy** | [`openai-server.md`](openai-server.md) | `qwen-studio login` + `qwen-studio serve`: an OpenAI API over your Qwen account - system prompts, MCP tools, uploads, 1-hour history routing |
| **Browser session import** | [`browser-session.md`](browser-session.md) | pull the full cookie jar from Firefox/Chrome/Chromium on Linux - the zero-setup constructor |
| Chat lifecycle | [`chats.md`](chats.md) | create, list, read, delete and batch-delete chats |
| Streaming completions | [`streaming.md`](streaming.md) | the SSE phase machine (thinking → tools → answer) |
| Hosted MCP tools | [`mcp-tools.md`](mcp-tools.md) | registry + per-user activation + MCP-enabled chat |
| Custom (client) tools | [`local-tools.md`](local-tools.md) | wrap any Python callable as an MCP-style chat tool |
| Capability study | [`capabilities.md`](capabilities.md) | system prompts (projects), images >5/turn, history injection, OpenAI shape - proven by live probes |
| Anti-bot resilience | [`anti-bot.md`](anti-bot.md) | the RGV587 risk engine, pinpointed by experiment, and how the library replicates the browser |

Start with the [quickstart](index.md#quickstart), then read
[authentication.md](authentication.md) once to understand the token
lifecycle the client manages for you.

## Status of verification

Everything in these docs was verified against the live service during the
protocol study, with the single exception discussed in
[local-tools.md](local-tools.md#server-side-gating-of-continuations): the
final tool-result continuation of the client-side tool loop is rejected for
unrecognised external client sessions. The library implements the full loop
faithfully and raises a typed exception at that boundary instead of hiding
it.

## Responsible use

Automated access to Qwen Studio sits outside Alibaba's terms of service.
This library exists to document *how the web client works* — keep volumes
at personal-experimentation level, never hammer the endpoints (the anti-bot
risk engine will punish you, see
[streaming.md#anti-bot-punish-rgv587](streaming.md#anti-bot-punish-rgv587)),
and use the official DashScope API for anything serious.
