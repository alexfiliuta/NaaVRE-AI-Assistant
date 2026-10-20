"""MCP SDK client with a fixed lab subprocess and exact tool allowlist."""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from mcp import Client, StdioServerParameters

from .labs import Lab, LabError


class MCPGateway:
    def __init__(self, lab: Lab):
        self.lab = lab
        env = {k: v for k, v in os.environ.items()
               if k.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE"}}
        env.update(NAAVRE_LABS_DIR=str(lab.root.parent), NAAVRE_MCP_LAB_ID=lab.id,
                   NAAVRE_ALLOWED_LABS=lab.id, PYTHONIOENCODING="utf-8")
        self.client = Client(StdioServerParameters(
            command=sys.executable, args=["-m", "naavre_ai_assistant.lab_mcp_server"], env=env,
        ), read_timeout_seconds=45)
        self._schemas: dict[str, dict[str, Any]] = {}

    async def __aenter__(self):
        self.lab._check()
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self.client.__aexit__(*args)

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self.client.list_tools()
        tools = []
        for tool in result.tools:
            item = tool.model_dump(by_alias=True, mode="json", exclude_none=True)
            if item["name"] in self.lab.allowed_tools:
                tools.append(item)
        self._schemas = {tool["name"]: tool["inputSchema"] for tool in tools}
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from jsonschema import Draft202012Validator
        self.lab._check()
        if name not in self.lab.allowed_tools:
            raise LabError("Tool is not allowed in this lab")
        if not self._schemas:
            await self.list_tools()
        if name not in self._schemas:
            raise LabError("MCP server did not expose requested tool")
        Draft202012Validator(self._schemas[name]).validate(arguments)
        result = await self.client.call_tool(name, arguments)
        payload = result.model_dump(by_alias=True, mode="json", exclude_none=True)
        if len(json.dumps(payload)) > 40000:
            raise LabError("MCP tool result exceeds the context limit")
        return payload

    async def resources(self) -> list[dict[str, Any]]:
        result = await self.client.list_resources()
        return [r.model_dump(by_alias=True, mode="json") for r in result.resources]
