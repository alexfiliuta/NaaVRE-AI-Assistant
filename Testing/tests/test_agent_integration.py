"""Real MCP subprocess + runtime integration, with an injected model (no Ollama)."""
import asyncio
import json

import pytest

from naavre_ai_assistant.agent_runtime import AgentLimits, AgentRuntime
from naavre_ai_assistant.labs import LabRegistry
from naavre_ai_assistant.mcp_gateway import MCPGateway


@pytest.fixture
def lab_registry(tmp_path, monkeypatch):
    monkeypatch.delenv("NAAVRE_ALLOWED_LABS", raising=False)
    lab = tmp_path / "labs" / "environment"
    for area in ("data", "knowledge", "skills", "outputs"):
        (lab / area).mkdir(parents=True)
    (lab / "lab.json").write_text(json.dumps({
        "id": "environment", "name": "Environmental research",
        "description": "A real MCP integration lab", "style": "cite source evidence",
        "frameworks": ["pandas"], "skills": ["inspect-columns"],
        "allowed_tools": ["list_files", "list_skills", "inspect_dataset", "search_knowledge", "save_note"],
        "enable_gbif": False,
    }), encoding="utf-8")
    (lab / "data" / "observations.csv").write_text("species,count\nQuercus robur,3\n", encoding="utf-8")
    (lab / "knowledge" / "protocol.md").write_text("Species counts require exact observed headers.", encoding="utf-8")
    (lab / "skills" / "inspect-columns.md").write_text("Inspect table columns before planning transformations.", encoding="utf-8")
    return LabRegistry(tmp_path / "labs", permitted_labs=["environment"])


async def wait_for(manager, run_id, expected):
    async def poll():
        while True:
            snapshot = manager.get("alice", run_id)
            if snapshot["status"] == expected:
                return snapshot
            if snapshot["status"] in {"completed", "failed", "cancelled", "exhausted"}:
                raise AssertionError(snapshot)
            await asyncio.sleep(.01)
    return await asyncio.wait_for(poll(), 30)


class Decisions:
    def __init__(self, *values):
        self.values = iter(values)
        self.prompts = []

    async def __call__(self, prompt):
        self.prompts.append(json.loads(prompt))
        return json.dumps(next(self.values))


def tool(name, **arguments):
    return {"action": "tool", "tool": name, "arguments": arguments}


def test_real_mcp_inspection_knowledge_and_exact_approved_note(tmp_path, lab_registry):
    async def scenario():
        model = Decisions(
            tool("inspect_dataset", path="observations.csv"),
            tool("search_knowledge", query="Species counts"),
            tool("save_note", filename="reviewed.md", text="Verified columns: species, count."),
            {"action": "finish", "summary": "Use observed species/count columns and cite protocol.md in the notebook plan."},
        )
        manager = AgentRuntime(lab_registry, tmp_path / "agent-runs", model=model,
                               limits=AgentLimits(run_seconds=40, operation_seconds=15))
        run_id = manager.start("alice", "environment", "Inspect and plan species counts")["run_id"]
        pending = await wait_for(manager, run_id, "awaiting_approval")
        output = lab_registry.get("environment").root / "outputs" / "reviewed.md"
        assert not output.exists()
        assert pending["pending_action"]["arguments"] == {
            "filename": "reviewed.md", "text": "Verified columns: species, count."}
        # HTTP snapshots are detached; an approval cannot change these arguments.
        pending["pending_action"]["arguments"]["text"] = "tampered"
        manager.decide("alice", run_id, True)
        result = await wait_for(manager, run_id, "completed")
        await manager._runs[run_id].task
        assert output.read_text(encoding="utf-8") == "Verified columns: species, count."
        assert model.prompts[0]["lab"]["style"] == "cite source evidence"
        assert model.prompts[0]["lab"]["frameworks"] == ["pandas"]
        assert "observations.csv" in json.dumps(model.prompts[0])
        assert "species" in json.dumps(model.prompts[1])
        assert "knowledge/protocol.md" in json.dumps(model.prompts[2])
        observations = [item for item in result["events"] if item["kind"] == "observation"]
        schema = next(item for item in observations if item["tool"] == "inspect_dataset")
        assert schema["result"]["structuredContent"]["headers"] == ["species", "count"]
        assert schema["result"]["isError"] is False
        verified = manager.verified_context("alice", run_id)
        assert verified["agent"]["skills"]
        assert verified["available_files"] == [
            lab_registry.get("environment").resolve("data", "observations.csv").resolve().as_posix()]
        assert list((tmp_path / "agent-runs").rglob("audit.jsonl"))
    asyncio.run(scenario())


def test_inspected_file_declarations_resolve_lab_path_not_tool_absolute_path(lab_registry):
    from naavre_ai_assistant.agent_runtime import inspected_file_declarations
    lab = lab_registry.get("environment")
    context = {"agent": {"observations": [{"tool": "inspect_dataset", "result": {
        "isError": False, "structuredContent": {"path": "data/observations.csv",
            "kernel_path": "/outside/secret.csv", "headers": ["species", "count"]}}}]}}
    result = inspected_file_declarations(lab, context)
    assert result["available_files"] == [lab.resolve("data", "observations.csv").resolve().as_posix()]
    context["agent"]["observations"][0]["result"]["isError"] = True
    assert inspected_file_declarations(lab, context) == {}


def test_inspected_file_declarations_reject_path_escape(lab_registry):
    from naavre_ai_assistant.agent_runtime import inspected_file_declarations
    from naavre_ai_assistant.labs import LabError
    context = {"agent": {"observations": [{"tool": "inspect_dataset", "result": {
        "structuredContent": {"path": "data/../secret.csv", "headers": ["secret"]}}}]}}
    with pytest.raises(LabError):
        inspected_file_declarations(lab_registry.get("environment"), context)


@pytest.mark.parametrize("action", ["deny", "cancel"])
def test_real_mcp_unapproved_write_and_cancel_scope_cleanup(tmp_path, lab_registry, action):
    async def scenario():
        model = Decisions(tool("save_note", filename="unapproved.md", text="Do not write"),
                          {"action": "finish", "summary": "Write denied; observations remain a proposed plan."})
        manager = AgentRuntime(lab_registry, tmp_path / "runs", model=model,
                               limits=AgentLimits(run_seconds=40, operation_seconds=15))
        run_id = manager.start("alice", "environment", "Propose a note")["run_id"]
        await wait_for(manager, run_id, "awaiting_approval")
        task = manager._runs[run_id].task
        if action == "deny":
            manager.decide("alice", run_id, False)
            await wait_for(manager, run_id, "completed")
        else:
            manager.cancel("alice", run_id)
        await asyncio.wait_for(task, 10)
        result = manager.get("alice", run_id)
        assert result["status"] == ("completed" if action == "deny" else "cancelled")
        assert result["pending_action"] is None
        assert not (lab_registry.get("environment").root / "outputs" / "unapproved.md").exists()
        # Enter and exit of the MCP/anyio context occur in the same task.
        assert not any("cleanup reported" in event["message"] for event in result["events"])
    asyncio.run(scenario())


def test_real_gateway_advertises_only_fixed_lab_tools_and_reports_tool_error(lab_registry):
    async def scenario():
        async with MCPGateway(lab_registry.get("environment")) as gateway:
            tools = await gateway.list_tools()
            assert {item["name"] for item in tools} == {
                "list_files", "list_skills", "inspect_dataset", "search_knowledge", "save_note"}
            result = await gateway.call_tool("inspect_dataset", {"path": "../outside.csv"})
            assert result["isError"] is True
            result = await gateway.call_tool("list_files", {})
            assert result["isError"] is False
            assert result["structuredContent"]["files"][0]["path"] == "data/observations.csv"
    asyncio.run(scenario())


def test_real_mcp_repeated_inspection_recovers_to_protocol(tmp_path, lab_registry):
    async def scenario():
        decisions = iter([
            tool("inspect_dataset", path="observations.csv"),
            tool("inspect_dataset", path="data/observations.csv"),
            tool("inspect_dataset", path="observations.csv"),
            tool("search_knowledge", query="Species counts"),
            {"action": "finish", "summary": "Plan species/count analysis using protocol.md; no code executed."},
        ])
        async def model(prompt):
            return json.dumps(next(decisions))
        manager = AgentRuntime(lab_registry, tmp_path / "runs", model=model,
                               limits=AgentLimits(max_steps=4, run_seconds=40, operation_seconds=15))
        source = lab_registry.get("environment").root / "data" / "observations.csv"
        original = source.read_bytes()
        run_id = manager.start("alice", "environment", "Inspect data and the lab protocol")["run_id"]
        result = await wait_for(manager, run_id, "completed")
        await manager._runs[run_id].task
        observed = [e["tool"] for e in result["events"] if e["kind"] == "observation"]
        assert observed == ["list_files", "list_skills", "inspect_dataset", "inspect_dataset", "search_knowledge"]
        assert any(e["kind"] == "reuse" for e in result["events"])
        assert "knowledge/protocol.md" in json.dumps(result)
        assert source.read_bytes() == original
        assert not list((source.parent.parent / "outputs").iterdir())
    asyncio.run(scenario())


def test_real_mcp_stalled_model_has_explanatory_failure_after_cleanup(tmp_path, lab_registry):
    async def scenario():
        async def model(prompt):
            return json.dumps(tool("inspect_dataset", path="observations.csv"))
        manager = AgentRuntime(lab_registry, tmp_path / "runs", model=model,
                               limits=AgentLimits(run_seconds=40, operation_seconds=15))
        run_id = manager.start("alice", "environment", "Inspect and plan")["run_id"]
        result = await wait_for(manager, run_id, "failed")
        await manager._runs[run_id].task
        assert "already completed read" in result["summary"]
        assert "AgentStalled" not in result["summary"]
        assert sum(e["kind"] == "observation" and e["tool"] == "inspect_dataset"
                   for e in result["events"]) == 2
    asyncio.run(scenario())
