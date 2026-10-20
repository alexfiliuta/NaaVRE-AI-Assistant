import json

import pytest

from test_labs import lab_registry
from naavre_ai_assistant.labs import LabError
from naavre_ai_assistant.mcp_gateway import MCPGateway


@pytest.mark.asyncio
async def test_real_stdio_protocol_discovers_and_inspects(lab_registry):
    lab = lab_registry.get("a")
    (lab.root / "data" / "records.csv").write_text("timestamp,temperature_c\n2026-01-01,12.5\n")
    async with MCPGateway(lab) as gateway:
        tools = await gateway.list_tools()
        assert "inspect_dataset" in {t["name"] for t in tools}
        assert "gbif_species_lookup" not in {t["name"] for t in tools}
        result = await gateway.call_tool("inspect_dataset", {"path": "records.csv"})
        assert not result.get("isError")
        assert result["structuredContent"]["headers"] == ["timestamp", "temperature_c"]
        resources = await gateway.resources()
        assert "lab://profile" in {r["uri"] for r in resources}
        with pytest.raises(LabError):
            await gateway.call_tool("shell", {"command": "whoami"})


@pytest.mark.asyncio
async def test_real_stdio_rejects_traversal_and_existing_output(lab_registry):
    async with MCPGateway(lab_registry.get("a")) as gateway:
        denied = await gateway.call_tool("inspect_dataset", {"path": "../../b/secret.csv"})
        assert denied.get("isError")
        first = await gateway.call_tool("save_note", {"filename": "review.md", "text": "reviewed"})
        assert not first.get("isError")
        again = await gateway.call_tool("save_note", {"filename": "review.md", "text": "replace"})
        assert again.get("isError")
        assert (lab_registry.get("a").root / "outputs" / "review.md").read_text() == "reviewed"
