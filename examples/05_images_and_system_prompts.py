"""Images + system prompts (projects) - the v0.3.0 capability additions.

- q.files.upload_image(): STS token + direct OSS PUT (no OSS SDK needed)
- q.chat.send(..., files=[...]): image turns; multiple images per turn work
  (the 5-image cap is client-UI only)
- q.projects.system_chat(): server-side system prompt via project
  custom_instruction (a role:"system" request message is ignored by the API)
"""
from qwen_studio import QwenStudio

q = QwenStudio.from_browser()
model = q.list_model_ids()[0]
vis_model = next((m for m in q.list_model_ids()
                  if any(k in m.lower() for k in ("vl", "omni", "vision"))), model)

# ---- image turn -----------------------------------------------------------
with open("photo.png", "rb") as f:            # any image file
    ref = q.files.upload_image(f.read(), "photo.png")

chat = q.chats.create(vis_model)
turn = q.chat.send(chat.id, "Describe this image in two sentences.",
                   vis_model, files=[ref.entry()])
print(turn.text)
q.chats.delete(chat.id)

# ---- system prompt (server-side, invisible on the wire) --------------------
chat = q.projects.system_chat(model, "Always answer in exactly five words.")
turn = q.chat.send(chat.id, "Explain gravity.", model)
print(turn.text)
q.chats.delete(chat.id)
q.projects.delete(chat.raw["project_id"])
