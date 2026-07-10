"""
Generic MCP (Model Context Protocol) client for the AI Agent.

Connects to any MCP servers listed in ~/.config/voice-assistant/mcp.json (same
format as Claude Desktop) and exposes their tools to the agent, namespaced as
`mcp__<server>__<tool>`. JetBrains (PyCharm/IntelliJ) is one such server.

The MCP SDK is asyncio; the rest of the agent is synchronous, so a single event
loop runs on a background thread and sync callers bridge in via
run_coroutine_threadsafe.
"""

import asyncio
import json
import os
import threading

CONFIG_PATH = os.path.expanduser("~/.config/voice-assistant/mcp.json")


def load_config():
    """Return {server_name: {command/args/env or url}} from mcp.json."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"⚠️  mcp.json ignored ({exc})")
        return {}
    return data.get("mcpServers", data if isinstance(data, dict) else {})


class MCPManager:
    """Owns the asyncio loop/thread and one session per configured server."""

    def __init__(self):
        self._loop = None
        self._thread = None
        self._sessions = {}          # name -> ClientSession
        self._stacks = {}            # name -> AsyncExitStack (keeps transports open)
        self.tools = {}              # full_name -> {"server","tool","schema"}
        self.errors = {}             # name -> error string

    # -- loop plumbing ------------------------------------------------------
    def _ensure_loop(self):
        if self._loop:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _run(self, coro, timeout=60):
        self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # -- connecting ---------------------------------------------------------
    async def _connect_one(self, name, cfg):
        from contextlib import AsyncExitStack
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        stack = AsyncExitStack()
        if cfg.get("url"):                         # SSE/HTTP transport
            from mcp.client.sse import sse_client
            read, write = await stack.enter_async_context(sse_client(cfg["url"]))
        else:                                      # stdio transport (spawn a process)
            params = StdioServerParameters(
                command=cfg["command"], args=cfg.get("args", []),
                env={**os.environ, **cfg.get("env", {})})
            read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        self._sessions[name] = session
        self._stacks[name] = stack
        count = 0
        for t in listed.tools:
            full = f"mcp__{name}__{t.name}"
            self.tools[full] = {"server": name, "tool": t.name,
                                "schema": t.inputSchema or {"type": "object", "properties": {}},
                                "description": t.description or ""}
            count += 1
        return count

    def connect(self, servers=None, timeout=45):
        """Connect to all configured (or given) servers. Returns {name: status}."""
        servers = servers if servers is not None else load_config()
        results = {}
        for name, cfg in servers.items():
            if name in self._sessions:
                results[name] = f"already connected ({sum(1 for t in self.tools.values() if t['server']==name)} tools)"
                continue
            try:
                n = self._run(self._connect_one(name, cfg), timeout=timeout)
                self.errors.pop(name, None)
                results[name] = f"connected — {n} tools"
            except Exception as exc:
                self.errors[name] = str(exc)
                results[name] = f"failed: {str(exc)[:120]}"
        return results

    # -- using tools --------------------------------------------------------
    def ollama_tools(self):
        """MCP tools in Ollama function-calling format."""
        return [{"type": "function", "function": {
            "name": full,
            "description": (info["description"] or f"{info['tool']} (via {info['server']})")[:1000],
            "parameters": info["schema"] if info["schema"].get("type") else
                          {"type": "object", "properties": {}},
        }} for full, info in self.tools.items()]

    def has(self, name):
        return name in self.tools

    def call(self, full_name, args, timeout=120):
        info = self.tools.get(full_name)
        if not info:
            return f"Unknown MCP tool: {full_name}"
        session = self._sessions.get(info["server"])
        if session is None:
            return f"MCP server '{info['server']}' is not connected."
        try:
            result = self._run(session.call_tool(info["tool"], args or {}), timeout=timeout)
        except Exception as exc:
            return f"MCP call failed: {exc}"
        parts = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else str(block))
        out = "\n".join(parts).strip()
        if getattr(result, "isError", False):
            return f"(tool error) {out}"
        return out or "(no output)"

    def status(self):
        if not self._sessions and not self.errors:
            return "No MCP servers connected. Add them to ~/.config/voice-assistant/mcp.json and /mcp connect."
        lines = []
        for name in self._sessions:
            n = sum(1 for t in self.tools.values() if t["server"] == name)
            lines.append(f"  🟢 {name}: {n} tools")
        for name, err in self.errors.items():
            lines.append(f"  🔴 {name}: {err[:100]}")
        return "MCP servers:\n" + "\n".join(lines)

    def close(self):
        for name, stack in list(self._stacks.items()):
            try:
                self._run(stack.aclose(), timeout=5)
            except Exception:
                pass
        self._sessions.clear()
        self._stacks.clear()
        self.tools.clear()


MANAGER = MCPManager()
