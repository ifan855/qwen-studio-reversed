"""Zero-setup: borrow the session of your local browser (Linux).

Log in to chat.qwen.ai in Firefox / Chrome / Chromium / Brave / Edge once,
then run this - the library reads the complete cookie jar (session token
plus the anti-bot set) straight from the browser profile and uses a
Chrome-impersonated transport, so script traffic is browser-equivalent
from the very first request.

See docs/browser-session.md for how extraction works per browser.
"""
from qwen_studio import QwenStudio

q = QwenStudio.from_browser()               # auto-detects the best profile
print("session imported from:", q.cookie_source)
print("anti-bot cookies carried:", sorted(q.extra_cookies))

model = q.list_model_ids()[0]
chat = q.chats.create(model)
turn = q.chat.send(chat.id, "In one short sentence: why is the sky blue?",
                   model)
print("answer:", turn.text)
q.chats.delete(chat.id)
print("chat deleted - done.")
