"""qwen-studio - a Python client for the Qwen Studio (chat.qwen.ai) web API.

Reverse-engineered, live-verified client exposing:

- chat lifecycle: create / list / read / delete / batch-delete
- streaming completions with the SSE phase machine
- hosted MCP tools: registry, per-user activation, MCP-enabled chat
- custom tools: wrap Python callables as client-side MCP (``local_mcp``) tools

Educational/research use; automated access sits outside Qwen's terms of
service - see the README.
"""

from .client import QwenStudio
from .chats import Chat, ChatMessage, ChatService
from .chat import ChatCompletion, Turn
from .tools import MCPService, MCPServer
from .local_tools import LocalToolSession, Tool, tool
from .sse import ChatEvent, StreamResult
from .projects import Project, ProjectService
from .files import FileRef, FileService
from .openai_api import (OpenAICompatService, QwenBackend, SessionRouter,
                         Session, view_message, render_history_document)
from . import browser_cookies, exceptions
from .browser_cookies import (BrowserCookie, BrowserCookieError,
                              load_browser_cookies, qwen_cookie_dict)

__version__ = "0.4.0"

__all__ = [
    "QwenStudio",
    "Chat", "ChatMessage", "ChatService",
    "ChatCompletion", "Turn",
    "MCPService", "MCPServer",
    "LocalToolSession", "Tool", "tool",
    "Project", "ProjectService", "FileRef", "FileService",
    "ChatEvent", "StreamResult",
    "OpenAICompatService", "QwenBackend", "SessionRouter", "Session",
    "view_message", "render_history_document",
    "browser_cookies", "BrowserCookie", "BrowserCookieError",
    "load_browser_cookies", "qwen_cookie_dict",
    "exceptions",
    "__version__",
]
