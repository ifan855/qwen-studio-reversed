"""Create a chat, check it exists, then delete it (single + batch)."""
import os

from qwen_studio import QwenStudio

q = QwenStudio.from_credentials(os.environ["QWEN_EMAIL"], os.environ["QWEN_PASSWORD"])
model = q.list_model_ids()[0]  # ids rotate - always pick from the catalogue

# 1. create
chat = q.chats.create(model)
print("created:", chat.id)

# 2. it appears in the history list
pages = q.chats.list(page=1)
print("history page 1 contains", len(pages), "chats")

# 3. read it back (empty - no messages yet)
record = q.chats.get(chat.id)
print("messages so far:", len(record.get("messages") or []))

# 4. delete single
print("deleted:", q.chats.delete(chat.id))
try:
    q.chats.get(chat.id)
except Exception as e:
    print("after delete, GET raises:", type(e).__name__)

# 5. batch delete example (deletes nothing you care about: fresh ids only)
a, b = q.chats.create(model), q.chats.create(model)
print("batch deleted:", q.chats.delete_many([a.id, b.id]))
