"""v0.4.0: the two-command flow - browser login file + OpenAI-compatible API.

Run from a directory where you want the auth file to live:

    python examples/06_openai_server.py            # step 1 + step 2

Then point any OpenAI client at http://127.0.0.1:8080/v1 :
    client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
"""
import json
import os

from qwen_studio import QwenStudio
from qwen_studio.openai_api import OpenAICompatService, QwenBackend
from qwen_studio.openai_server import OpenAIProxyServer

AUTH_FILE = os.path.join(os.getcwd(), "qwen.auth.json")

# ------------------------------------------------------------------ step 1
# Pull the complete cookie jar straight from your logged-in browser profile
# (Firefox / Chrome / Chromium / Brave / Edge on Linux) into ./qwen.auth.json
if not os.path.exists(AUTH_FILE):
    from qwen_studio.browser_cookies import find_qwen_jar
    prof, cookies = find_qwen_jar()                     # auto-detect
    jar = {c.name: c.value for c in cookies}
    token = jar.pop("token")
    with open(AUTH_FILE, "w") as f:
        json.dump({"version": 1, "session_token": token,
                   "extra_cookies": jar,
                   "cookie_source": f"{prof.browser}:{prof.name}"},
                  f)
    os.chmod(AUTH_FILE, 0o600)
    print(f"auth file written: {AUTH_FILE} ({prof.browser})")
# (or just run: qwen-studio login)

# ------------------------------------------------------------------ step 2
# Serve the OpenAI-compatible API reading that auth file.
with open(AUTH_FILE) as f:
    auth = json.load(f)

client = QwenStudio.from_session_token(auth["session_token"],
                                       extra_cookies=auth["extra_cookies"])
backend = QwenBackend(client)
service = OpenAICompatService(backend, ttl=3600.0, replay_mode="both")
service.start_sweeper()
server = OpenAIProxyServer(("127.0.0.1", 8080), service)

print("OpenAI-compatible API on http://127.0.0.1:8080/v1  (Ctrl-C to stop)")
try:
    server.serve_forever()
except KeyboardInterrupt:
    service.stop_sweeper()
    for s in list(service.router._sessions.values()):
        service.router.drop(s)
        service._expire_session(s)
    server.server_close()
    print("stopped; conversations cleaned up upstream.")
