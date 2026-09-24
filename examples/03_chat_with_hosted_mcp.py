"""Chat with platform-hosted MCP tools (server-side execution)."""
import os

from qwen_studio import QwenStudio

q = QwenStudio.from_credentials(os.environ["QWEN_EMAIL"], os.environ["QWEN_PASSWORD"])
model = q.list_model_ids()[0]

# 1. see what the platform hosts
reg = q.tools.registry()
for sid, server in reg.items():
    print(f"- {sid} ({server.type}): {server.tool_names()}")

# 2. activate one for this account and WAIT for confirmation
target = "amap" if "amap" in reg else next(iter(reg), None)
if not target:
    raise SystemExit("registry empty")
print("enabling:", q.tools.enable(target))  # re-reads settings until confirmed

# 3. chat - the request carries no tool list; the backend applies your state
chat = q.chats.create(model)
turn = q.chat.send(
    chat.id,
    "Use the maps_geo tool to get the coordinates of the Oriental Pearl "
    "Tower in Shanghai, then report them.",
    model,
    mcp_enabled=True,
)
print("phases:", " -> ".join(turn.result.phases))
print("answer:", turn.text[:400])

q.tools.disable(target)
q.chats.delete(chat.id)
