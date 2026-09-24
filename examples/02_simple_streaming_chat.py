"""Streaming chat, event by event, using the phase machine."""
import os

from qwen_studio import QwenStudio

q = QwenStudio.from_credentials(os.environ["QWEN_EMAIL"], os.environ["QWEN_PASSWORD"])
model = q.list_model_ids()[0]
chat = q.chats.create(model)

# one-shot: just the answer
turn = q.chat.send(chat.id, "Say hello in five words.", model)
print("answer:", turn.text)
print("phases:", " -> ".join(turn.result.phases))
print("response_id:", turn.response_id)

# follow-up turn in the same chat, event-by-event this time
result = q.chat.send(chat.id, "Now count from 1 to 5.", model,
                     keep_events=True).result
for ev in result.events:
    if ev.type == "created":
        print("<new response>", ev.response_id)
    elif ev.phase == "answer" and ev.content:
        print(ev.content, end="", flush=True)
print()

q.chats.delete(chat.id)
