"""Chat with custom tools: Python callables wrapped into the MCP protocol.

Declaration + model invocation are verified working externally. The final
tool-result continuation is server-gated for unrecognised client sessions;
if that gate bites, the library raises ContinuationBlockedError carrying
your executed results - see docs/local-tools.md.
"""
import os

from qwen_studio import QwenStudio, tool
from qwen_studio.exceptions import ContinuationBlockedError


@tool
def get_weather(city: str):
    """Get the current weather for a city"""
    return {"city": city, "temperature": "26C", "condition": "Cloudy",
            "humidity": "78%", "source": "local_py_tool"}


@tool
def add_numbers(a: int, b: int):
    """Add two integers"""
    return {"sum": a + b}


q = QwenStudio.from_credentials(os.environ["QWEN_EMAIL"], os.environ["QWEN_PASSWORD"])
model = q.list_model_ids()[0]
chat = q.chats.create(model)

session = q.local_tools(chat.id)
session.register(get_weather)
session.register(add_numbers)

# manual loop, step by step:
res = session.send("What is the weather in Shanghai? Use the get_weather tool.",
                   model)
print("phase stream:", " -> ".join(res.phases))
print("model requested:", res.tool_calls)

results = session.execute_calls(res.tool_calls)  # runs your python locally
print("executed:", results)

try:
    final = session.send_tool_results(model, res.response_id, results)
    print("final answer:", final.answer)
except ContinuationBlockedError as e:
    # server-side gate for external continuations - your tool still ran
    print("continuation blocked by server gate; executed results kept:",
          e.results)

# ...or the full auto loop: session.ask("...", model)
q.chats.delete(chat.id)
