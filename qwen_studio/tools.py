"""Hosted MCP: registry discovery and per-user tool activation.

This is the *server-side* MCP mechanism - the one used by the web client.
The key protocol finding (report section 6.1): tool availability is **not**
declared in chat requests. Which servers are active is per-user server
state, toggled through the settings endpoint; the backend then attaches the
corresponding tool schemas to completions itself and executes hosted tools
server-side, streaming results into the phase machine.

Auth nuance verified live: the registry call ``GET /mcp/list`` authenticates
with the session **cookie** (a Bearer header is rejected), while the settings
write ``POST /users/user/settings/update`` requires the **Bearer** access
token. Timing matters: activation must be confirmed (re-read settings)
before a completion that relies on the tool, otherwise the model answers
without it - the failure mode observed in the report's section 8.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .client import QwenStudio


@dataclass
class MCPServer:
    """One registry entry (server) and its tools."""

    id: str
    type: Optional[str] = None
    description: Optional[str] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def tool_names(self) -> List[str]:
        names = []
        for t in self.tools:
            if isinstance(t, dict) and t:
                names.append(next(iter(t.keys())))
        return names

    def schema(self, tool: str) -> Optional[Dict[str, Any]]:
        """The input schema the registry advertises for ``tool``."""
        for t in self.tools:
            if isinstance(t, dict) and tool in t:
                entry = t[tool] or {}
                return entry.get("input_schema") or entry.get("parameters")
        return None


class MCPService:
    """Registry + activation state for platform-hosted MCP servers."""

    def __init__(self, client: QwenStudio) -> None:
        self.client = client
        self._registry: Optional[Dict[str, MCPServer]] = None

    # ------------------------------------------------------------- registry
    def registry(self, *, language: str = "en-US", refresh: bool = False
                 ) -> Dict[str, MCPServer]:
        """All platform-hosted MCP servers (cookie-authenticated call)."""
        if self._registry is not None and not refresh:
            return self._registry
        d = self.client.request("GET", "/mcp/list",
                                params={"language": language}, cookie_auth=True)
        data = d.get("data") or {}
        reg: Dict[str, MCPServer] = {}
        for sid, sv in data.items():
            if isinstance(sv, dict):
                reg[sid] = MCPServer(id=sid, type=sv.get("type"),
                                     description=sv.get("description"),
                                     tools=sv.get("tools") or [], raw=sv)
        self._registry = reg
        return reg

    def server(self, server_id: str) -> MCPServer:
        reg = self.registry()
        if server_id not in reg:
            raise KeyError(f"unknown MCP server {server_id!r}; "
                           f"known: {sorted(reg)}")
        return reg[server_id]

    # ----------------------------------------------------------- activation
    def enabled(self) -> Dict[str, bool]:
        """The account's current MCP activation map from user settings."""
        d = self.client.request("GET", "/users/user/settings", bearer=True)
        data = d.get("data") or {}
        settings = data.get("settings") or data
        mcp = (settings.get("mcp") or {}) if isinstance(settings, dict) else {}
        return {k: bool(v) for k, v in mcp.items()}

    def enable(self, *server_ids: str, confirm: bool = True,
               wait: float = 1.5) -> Dict[str, bool]:
        """Activate hosted MCP servers for this account.

        Writes ``{"mcp": {server: true}}`` through the settings endpoint and,
        by default, re-reads the settings map until the change is visible -
        the ordering safeguard that the report identified as the difference
        between tool runs that worked and scripted runs that silently fell
        back to "NO TOOL".

        Returns the confirmed activation map for the touched servers.
        """
        if not server_ids:
            return {}
        self.client.request("POST", "/users/user/settings/update",
                            json_body={"mcp": {s: True for s in server_ids}})
        if not confirm:
            return {s: True for s in server_ids}
        deadline = time.time() + 10
        last: Dict[str, bool] = {}
        while time.time() < deadline:
            time.sleep(wait)
            state = self.enabled()
            last = {s: state.get(s, False) for s in server_ids}
            if all(last.values()):
                return last
        return last

    def disable(self, *server_ids: str) -> Dict[str, Any]:
        self.client.request("POST", "/users/user/settings/update",
                            json_body={"mcp": {s: False for s in server_ids}})
        return {s: False for s in server_ids}
