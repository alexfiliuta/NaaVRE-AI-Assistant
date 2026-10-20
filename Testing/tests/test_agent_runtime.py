"""Runtime/security tests use injected MCP and model transports; no cloud or execution."""
import asyncio
import base64
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from naavre_ai_assistant.agent_runtime import (
    AgentError, AgentLimits, AgentRuntime, RunConflict, RunNotFound, TERMINAL,
)


class FakeLab:
    id = "environment"
    name = "Environment"
    description = "Research lab"
    style = "reproducible"
    frameworks = ("pandas",)

    def __init__(self, root):
        self.root = root

    def public_dict(self):
        return {key: getattr(self, key) for key in (
            "id", "name", "description", "style", "frameworks")}

    def files(self, area="data"):
        return [{"path": "reference.txt"}]

    def skills(self):
        return [{"id": "inspect", "name": "Inspect", "content": "Use exact columns"}]

    def search(self, query):
        return [{"text": "Observed species", "citation": {"path": "knowledge/reference.txt"}}]

    def add_document(self, filename, content):
        self.upload = filename, content
        return {"path": "knowledge/" + filename, "hash": "abc", "bytes": len(content),
                "private_root": str(self.root)}


class FakeRegistry:
    def __init__(self, root):
        self.value = FakeLab(root)

    def get(self, lab_id):
        if lab_id != self.value.id:
            raise KeyError(lab_id)
        return self.value

    def list_labs(self):
        return [self.value.public_dict()]


def schema(name, properties=None, required=()):
    return {"name": name, "description": name, "inputSchema": {
        "type": "object", "properties": properties or {},
        "required": list(required), "additionalProperties": False,
    }}


CATALOG = [schema("list_files"), schema("list_skills"),
           schema("inspect_dataset", {"path": {"type": "string"}}, ["path"]),
           schema("save_note", {"text": {"type": "string"}}, ["text"])]


class FakeGateway:
    def __init__(self, catalog=None, results=None):
        self.catalog = CATALOG if catalog is None else catalog
        self.results = results or {}
        self.calls = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def list_tools(self):
        return self.catalog

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name in self.results:
            return self.results[name]
        return {"isError": False, "structuredContent": {
            "list_files": {"files": ["observations.csv"]},
            "list_skills": {"skills": [{"name": "Inspect columns", "content": "Use exact columns"}]},
            "inspect_dataset": {"columns": ["species", "count"], "rows": 12},
            "save_note": {"saved": True},
        }[name]}


class FakeModel:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.prompts = []

    async def __call__(self, prompt):
        payload, _ = json.JSONDecoder().raw_decode(prompt)
        self.prompts.append(payload)
        decision = next(self.decisions)
        return decision if isinstance(decision, str) else json.dumps(decision)


def tool(name, **arguments):
    return {"action": "tool", "tool": name, "arguments": arguments}


def finish(summary="Use the observed species/count schema to plan notebook cells."):
    return {"action": "finish", "summary": summary}


def runtime(tmp_path, model, gateway=None, **kwargs):
    gateway = gateway or FakeGateway()
    manager = AgentRuntime(FakeRegistry(tmp_path / "labs" / "environment"),
                           tmp_path / "runs", model=model,
                           gateway_factory=lambda lab: gateway, **kwargs)
    return manager, gateway


async def settled(manager, run_id, status=None):
    for _ in range(500):
        result = manager.get("alice", run_id)
        if result["status"] == status or (status is None and result["status"] in TERMINAL):
            return result
        await asyncio.sleep(.001)
    raise AssertionError(f"Run did not reach expected status: {result['status']}")


def test_observe_choose_result_choose_again_and_persist(tmp_path):
    async def scenario():
        model = FakeModel(tool("inspect_dataset", path="observations.csv"), finish())
        manager, gateway = runtime(tmp_path, model)
        started = manager.start("alice", "environment", "Inspect species counts")
        assert started["status"] == "queued"
        result = await settled(manager, started["run_id"])
        assert result["status"] == "completed"
        assert [name for name, _ in gateway.calls] == ["list_files", "list_skills", "inspect_dataset"]
        assert "observations.csv" in json.dumps(model.prompts[0])
        assert '"rows": 12' in json.dumps(model.prompts[1])
        assert gateway.closed
        context = manager.verified_context("alice", result["run_id"])
        assert context["agent"]["lab"]["frameworks"] == ("pandas",)
        assert context["agent"]["skills"]
        assert context["agent"]["citations"]
        snapshots = list((tmp_path / "runs").rglob("state.json"))
        assert len(snapshots) == 1
        assert json.loads(snapshots[0].read_text())["status"] == "completed"
        audit = snapshots[0].with_name("audit.jsonl").read_text()
        assert "inspect_dataset" in audit and "reasoning" not in audit
        assert not (tmp_path / "labs" / "environment").exists()
    asyncio.run(scenario())


def test_unchanged_read_is_reused_and_agent_moves_to_knowledge_within_budget(tmp_path):
    async def scenario():
        inspect = tool("inspect_dataset", path="observations.csv")
        model = FakeModel(inspect, inspect, inspect,
                          tool("search_knowledge", query="protocol"), finish())
        gateway = FakeGateway(
            catalog=[*CATALOG, schema("search_knowledge", {"query": {"type": "string"}}, ["query"])],
            results={"search_knowledge": {"isError": False, "hits": ["Use the lab protocol"]}},
        )
        manager, _ = runtime(tmp_path, model, gateway, limits=AgentLimits(max_steps=4))
        run_id = manager.start("alice", "environment", "Inspect data and the lab protocol")["run_id"]
        result = await settled(manager, run_id)
        assert result["status"] == "completed"
        assert [name for name, _ in gateway.calls] == [
            "list_files", "list_skills", "inspect_dataset", "inspect_dataset", "search_knowledge"]
        assert len(model.prompts) == 5  # Corrected choice does not spend another workflow step.
        feedback = model.prompts[3]["decision_feedback"]
        assert feedback["observed_result"]["structuredContent"]["rows"] == 12
        assert feedback["completed_read"]["citation"] in manager.verified_context("alice", run_id)["agent"]["citations"]
        assert model.prompts[4]["remaining_steps"] == 1
        from jsonschema import Draft202012Validator
        masked = next(t for t in model.prompts[3]["tools"] if t["name"] == "inspect_dataset")
        validator = Draft202012Validator(masked["inputSchema"])
        assert not validator.is_valid({"path": "observations.csv"})
        assert not validator.is_valid({"path": "data/observations.csv"})
        assert validator.is_valid({"path": "different.csv"})
        assert sum(e["kind"] == "reuse" for e in result["events"]) == 1
    asyncio.run(scenario())


def test_stubborn_model_stops_early_without_false_completion(tmp_path):
    async def scenario():
        inspect = tool("inspect_dataset", path="observations.csv")
        manager, gateway = runtime(tmp_path, FakeModel(*([inspect] * 5)))
        run_id = manager.start("alice", "environment", "Inspect and plan")["run_id"]
        result = await settled(manager, run_id)
        assert result["status"] == "failed"
        assert "already completed read" in result["summary"]
        assert sum(name == "inspect_dataset" for name, _ in gateway.calls) == 2
        assert sum(e["kind"] == "reuse" for e in result["events"]) == 3
        assert gateway.closed
        with pytest.raises(RunConflict):
            manager.verified_context("alice", run_id)
    asyncio.run(scenario())


@pytest.mark.parametrize("calls", [
    [tool("list_files"), tool("list_files", area="data")],  # Initial discovery is the first read.
    [tool("inspect_dataset", path="observations.csv"),
     tool("inspect_dataset", path="data/observations.csv"),
     tool("inspect_dataset", path="observations.csv")],
])
def test_supported_argument_aliases_cannot_bypass_repeat_guard(tmp_path, calls):
    async def scenario():
        model = FakeModel(*calls, finish())
        manager, gateway = runtime(tmp_path, model)
        run_id = manager.start("alice", "environment", "Inspect")["run_id"]
        result = await settled(manager, run_id)
        assert result["status"] == "completed"
        assert sum(e["kind"] == "reuse" for e in result["events"]) == 1
        assert sum(name == calls[0]["tool"] for name, _ in gateway.calls) == 2
    asyncio.run(scenario())


def test_different_requests_are_not_merged_even_when_results_match(tmp_path):
    async def scenario():
        model = FakeModel(tool("inspect_dataset", path="first.csv"),
                          tool("inspect_dataset", path="second.csv"),
                          tool("search_knowledge", query="temperature"),
                          tool("search_knowledge", query="salinity"), finish())
        gateway = FakeGateway(
            catalog=[*CATALOG, schema("search_knowledge", {"query": {"type": "string"}}, ["query"])],
            results={"search_knowledge": {"isError": False, "hits": []}},
        )
        manager, _ = runtime(tmp_path, model, gateway)
        result = await settled(manager, manager.start("alice", "environment", "Compare sources")["run_id"])
        assert result["status"] == "completed" and len(gateway.calls) == 6
        assert not any(e["kind"] == "reuse" for e in result["events"])
    asyncio.run(scenario())


@pytest.mark.parametrize("middle", [
    {"isError": True, "message": "Temporary error"},
    {"isError": False, "structuredContent": {"rows": 13}},
])
def test_error_or_changed_result_resets_confirmation(tmp_path, middle):
    async def scenario():
        class ChangingGateway(FakeGateway):
            async def call_tool(self, name, arguments):
                result = await super().call_tool(name, arguments)
                if name == "inspect_dataset" and sum(n == name for n, _ in self.calls) == 2:
                    return middle
                return result
        inspect = tool("inspect_dataset", path="observations.csv")
        manager, gateway = runtime(tmp_path, FakeModel(inspect, inspect, inspect, finish()), ChangingGateway())
        result = await settled(manager, manager.start("alice", "environment", "Inspect")["run_id"])
        assert result["status"] == "completed"
        assert sum(n == "inspect_dataset" for n, _ in gateway.calls) == 3
        assert not any(e["kind"] == "reuse" for e in result["events"])
    asyncio.run(scenario())


def test_comparison_uses_full_result_not_truncated_preview(tmp_path):
    async def scenario():
        class LargeGateway(FakeGateway):
            async def call_tool(self, name, arguments):
                result = await super().call_tool(name, arguments)
                if name == "inspect_dataset":
                    return {"isError": False, "large": "x" * 10000, "revision": len(self.calls)}
                return result
        inspect = tool("inspect_dataset", path="observations.csv")
        manager, gateway = runtime(tmp_path, FakeModel(inspect, inspect, inspect, finish()), LargeGateway())
        result = await settled(manager, manager.start("alice", "environment", "Inspect")["run_id"])
        assert result["status"] == "completed"
        assert sum(n == "inspect_dataset" for n, _ in gateway.calls) == 3
        assert not any(e["kind"] == "reuse" for e in result["events"])
    asyncio.run(scenario())


def test_json_key_order_does_not_make_results_different(tmp_path):
    async def scenario():
        class ReorderedGateway(FakeGateway):
            async def call_tool(self, name, arguments):
                result = await super().call_tool(name, arguments)
                if name == "inspect_dataset" and len(self.calls) % 2 == 0:
                    return dict(reversed(list(result.items())))
                return result
        inspect = tool("inspect_dataset", path="observations.csv")
        manager, gateway = runtime(tmp_path, FakeModel(inspect, inspect, inspect, finish()), ReorderedGateway())
        result = await settled(manager, manager.start("alice", "environment", "Inspect")["run_id"])
        assert result["status"] == "completed"
        assert sum(n == "inspect_dataset" for n, _ in gateway.calls) == 2
    asyncio.run(scenario())


def test_confirmed_reads_are_run_local_and_invalidated_after_write(tmp_path):
    async def scenario():
        inspect = tool("inspect_dataset", path="observations.csv")
        model = FakeModel(inspect, inspect, tool("save_note", text="reviewed"), inspect, finish(),
                          inspect, finish())
        manager, gateway = runtime(tmp_path, model)
        run_id = manager.start("alice", "environment", "Inspect and save note")["run_id"]
        await settled(manager, run_id, "awaiting_approval")
        manager.decide("alice", run_id, True)
        assert (await settled(manager, run_id))["status"] == "completed"
        second = manager.start("alice", "environment", "Inspect again")["run_id"]
        assert (await settled(manager, second))["status"] == "completed"
        assert sum(n == "inspect_dataset" for n, _ in gateway.calls) == 4
    asyncio.run(scenario())


def test_progress_ledger_survives_event_window_truncation(tmp_path):
    async def scenario():
        inspect = tool("inspect_dataset", path="observations.csv")
        choices = iter([inspect, inspect, inspect, finish()])
        prompts = []
        async def model(prompt):
            payload, _ = json.JSONDecoder().raw_decode(prompt)
            prompts.append(payload)
            decision = next(choices)
            if len(prompts) == 2:
                # This successful tool output is followed by enough audit noise
                # to evict its original observation from the next event window.
                original_observe = manager._observe
                async def noisy_observe(*args, **kwargs):
                    await original_observe(*args, **kwargs)
                    run = args[0]
                    for _ in range(15):
                        manager._event(run, "validation", "x" * 4000)
                manager._observe = noisy_observe
            return json.dumps(decision)
        manager, gateway = runtime(tmp_path, model)
        result = await settled(manager, manager.start("alice", "environment", "Inspect")["run_id"])
        assert result["status"] == "completed"
        assert not any(e["kind"] == "observation" for e in prompts[2]["events"])
        assert any(e["tool"] == "inspect_dataset" and e["unchanged_twice"]
                   for e in prompts[2]["read_progress"])
        assert prompts[3]["decision_feedback"]["observed_result"]["structuredContent"]["rows"] == 12
        assert sum(n == "inspect_dataset" for n, _ in gateway.calls) == 2
    asyncio.run(scenario())


def test_model_decision_retries_malformed_json_and_discards_auxiliary_fields(tmp_path):
    async def scenario():
        model = FakeModel(
            "not-json",
            {**tool("inspect_dataset", path="observations.csv"), "rationale": "untrusted extra"},
            {**finish(), "analysis": "not part of the action record"},
        )
        manager, gateway = runtime(tmp_path, model)
        run_id = manager.start("alice", "environment", "Inspect and explain the data")['run_id']
        result = await settled(manager, run_id)
        assert result['status'] == 'completed'
        assert len(model.prompts) == 3
        assert [name for name, _ in gateway.calls] == ["list_files", "list_skills", "inspect_dataset"]
        serialized = json.dumps(result)
        assert "untrusted extra" not in serialized
        assert "not part of the action record" not in serialized
    asyncio.run(scenario())


@pytest.mark.parametrize("approve", [True, False])
def test_exact_write_confirmation_and_denial(tmp_path, approve):
    async def scenario():
        manager, gateway = runtime(tmp_path, FakeModel(tool("save_note", text="exact note"), finish()))
        run_id = manager.start("alice", "environment", "Plan and save a note")["run_id"]
        result = await settled(manager, run_id, "awaiting_approval")
        assert not any(name == "save_note" for name, _ in gateway.calls)
        result["pending_action"]["arguments"]["text"] = "tampered browser state"
        with pytest.raises(RunNotFound):
            manager.decide("mallory", run_id, True)
        with pytest.raises(AgentError):
            manager.decide("alice", run_id, "true")
        manager.decide("alice", run_id, approve)
        with pytest.raises(RunConflict):
            manager.decide("alice", run_id, True)
        result = await settled(manager, run_id)
        assert result["status"] == "completed" and result["pending_action"] is None
        writes = [args for name, args in gateway.calls if name == "save_note"]
        assert writes == ([{"text": "exact note"}] if approve else [])
        assert any(event["kind"] == "approval" for event in result["events"])
    asyncio.run(scenario())


def test_owner_checks_cancel_and_late_response(tmp_path):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        async def blocked_model(prompt):
            entered.set()
            await release.wait()
            return json.dumps(finish())
        manager, gateway = runtime(tmp_path, blocked_model)
        run_id = manager.start("alice", "environment", "Plan")["run_id"]
        await entered.wait()
        for operation in (manager.get, manager.cancel, manager.verified_context):
            with pytest.raises(RunNotFound):
                operation("mallory", run_id)
        assert manager.cancel("alice", run_id)["status"] == "cancelled"
        release.set()
        await asyncio.sleep(.01)
        assert manager.get("alice", run_id)["status"] == "cancelled"
        assert gateway.closed
        with pytest.raises(RunConflict):
            manager.verified_context("alice", run_id)
    asyncio.run(scenario())


def test_cancel_awaiting_write_never_executes(tmp_path):
    async def scenario():
        manager, gateway = runtime(tmp_path, FakeModel(tool("save_note", text="note")))
        run_id = manager.start("alice", "environment", "Save note")["run_id"]
        await settled(manager, run_id, "awaiting_approval")
        manager.cancel("alice", run_id)
        await asyncio.sleep(.01)
        assert manager.get("alice", run_id)["pending_action"] is None
        assert not any(name == "save_note" for name, _ in gateway.calls)
        with pytest.raises(RunConflict):
            manager.decide("alice", run_id, True)
    asyncio.run(scenario())


@pytest.mark.parametrize("decision", ["not json", {"action": "execute", "code": "print(1)"}])
def test_invalid_model_decisions_fail_without_recording_raw_reasoning(tmp_path, decision):
    async def scenario():
        manager, gateway = runtime(tmp_path, FakeModel(decision))
        run_id = manager.start("alice", "environment", "Plan")["run_id"]
        result = await settled(manager, run_id)
        assert result["status"] == "failed"
        assert "private thoughts" not in json.dumps(result)
        assert len(gateway.calls) == 2
    asyncio.run(scenario())


def test_unknown_tools_invalid_args_exhaustion_and_no_false_completion(tmp_path):
    async def scenario():
        decisions = [tool("execute_python", code="1+1"), tool("inspect_dataset", filename="wrong")]
        manager, gateway = runtime(tmp_path, FakeModel(*decisions), limits=AgentLimits(max_steps=2))
        run_id = manager.start("alice", "environment", "Inspect")["run_id"]
        result = await settled(manager, run_id)
        assert result["status"] == "exhausted"
        assert len(gateway.calls) == 2
        assert sum(e["kind"] == "validation" for e in result["events"]) == 2
    asyncio.run(scenario())


def test_finish_requires_real_successful_read(tmp_path):
    async def scenario():
        gateway = FakeGateway(catalog=[schema("inspect_dataset")])
        manager, _ = runtime(tmp_path, FakeModel(finish()), gateway, limits=AgentLimits(max_steps=1))
        run_id = manager.start("alice", "environment", "Plan")["run_id"]
        result = await settled(manager, run_id)
        assert result["status"] == "exhausted" and gateway.calls == []
    asyncio.run(scenario())


def test_mcp_errors_are_observations_not_evidence_of_success(tmp_path):
    async def scenario():
        failed = {"isError": True, "content": [{"text": "No data available"}]}
        gateway = FakeGateway(results={"list_files": failed, "list_skills": failed})
        manager, _ = runtime(tmp_path, FakeModel(finish()), gateway, limits=AgentLimits(max_steps=1))
        run_id = manager.start("alice", "environment", "Plan")["run_id"]
        result = await settled(manager, run_id)
        assert result["status"] == "exhausted"
        assert "No data available" in json.dumps(result)
    asyncio.run(scenario())


def test_operation_timeout_is_honest_exhaustion(tmp_path):
    async def scenario():
        async def slow_model(prompt):
            await asyncio.sleep(10)
        manager, _ = runtime(tmp_path, slow_model, limits=AgentLimits(operation_seconds=.02))
        run_id = manager.start("alice", "environment", "Plan")["run_id"]
        assert (await settled(manager, run_id))["status"] == "exhausted"
    asyncio.run(scenario())


def test_context_limits_tool_result_truncation_and_detached_request(tmp_path):
    async def scenario():
        gateway = FakeGateway(results={"list_files": {"data": "x" * 100000}})
        model = FakeModel(finish())
        manager, _ = runtime(tmp_path, model, gateway)
        context = {"notebook": "original"}
        run_id = manager.start("alice", "environment", "Plan", context)["run_id"]
        context["notebook"] = "changed"
        result = await settled(manager, run_id)
        assert result["status"] == "completed"
        assert model.prompts[0]["caller_context"]["notebook"] == "original"
        assert any(e.get("result", {}).get("truncated") for e in result["events"])
        assert len(json.dumps(result["context"])) < 35000
        with pytest.raises(AgentError):
            manager.start("alice", "environment", "Plan", {"x": "x" * 15000})
    asyncio.run(scenario())


def test_restart_marks_interrupted_run_failed_and_enforces_disk_owner(tmp_path):
    async def scenario():
        manager, _ = runtime(tmp_path, FakeModel(tool("save_note", text="note")))
        run_id = manager.start("alice", "environment", "Plan")["run_id"]
        await settled(manager, run_id, "awaiting_approval")
        restarted, _ = runtime(tmp_path, FakeModel())
        assert restarted.get("alice", run_id)["status"] == "failed"
        assert restarted.get("alice", run_id)["pending_action"] is None
        with pytest.raises(RunNotFound):
            restarted.get("mallory", run_id)
        with pytest.raises(RunNotFound):
            restarted.get("alice", "../state")
        manager.cancel("alice", run_id)
    asyncio.run(scenario())


def test_admin_allowlist_and_lab_confinement(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        runtime(tmp_path, FakeModel(), allowed_tools=["execute_python"])
    manager, _ = runtime(tmp_path, FakeModel(), permitted_labs=[])
    with pytest.raises(RunNotFound):
        manager.lab("environment")
    with pytest.raises(RunNotFound):
        manager.lab("../environment")
    manager = AgentRuntime(FakeRegistry(tmp_path), tmp_path / "nested-runs", model=FakeModel())
    with pytest.raises(AgentError, match="outside"):
        manager.lab("environment")


def test_remote_schema_refs_are_never_fetched(tmp_path):
    manager, _ = runtime(tmp_path, FakeModel())
    with pytest.raises(AgentError, match="Remote schema"):
        manager._tools([{"name": "list_files", "inputSchema": {"$ref": "https://example.org/schema"}}])
    gbif = schema("gbif_species_lookup", {"name": {"type": "string"}}, ["name"])
    assert manager._tools([gbif]) == {}
    manager.allowed_tools = frozenset({"gbif_species_lookup"})
    assert "gbif_species_lookup" in manager._tools([gbif])


def test_request_owner_uses_username():
    from naavre_ai_assistant.agent_handlers import request_owner
    assert request_owner(SimpleNamespace(username="alice", display_name="Bob")) == "alice"
    assert request_owner({"username": "alice"}) == "alice"
    assert request_owner(b"alice") == "alice"
    from tornado.web import HTTPError
    with pytest.raises(HTTPError):
        request_owner(SimpleNamespace(display_name="Alice"))


def test_local_transport_and_unlimited_model_configuration_are_preserved(tmp_path):
    from naavre_ai_assistant.llm_service import LLMCodeGenerator, LLMConfig
    async def scenario():
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return {"choices": [{"message": {"content": json.dumps(finish())}}]}
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        config = LLMConfig("http://127.0.0.1:11434/v1", "qwen3-coder:30b", None, None)
        service = LLMCodeGenerator(config, client)
        gateway = FakeGateway()
        manager = AgentRuntime(FakeRegistry(tmp_path / "lab"), tmp_path / "runs",
                               service=service, gateway_factory=lambda lab: gateway)
        run_id = manager.start("alice", "environment", "Plan")["run_id"]
        assert (await settled(manager, run_id))["status"] == "completed"
        assert requests[0]["model"] == "qwen3-coder:30b"
        assert "max_tokens" not in requests[0]
        assert service.config.timeout_seconds is None
        assert service.config.max_tokens is None
        assert manager.service is service
        assert manager.limits.operation_seconds is None
        assert manager.limits.run_seconds is None
        assert "frameworks" in requests[0]["messages"][1]["content"]
    asyncio.run(scenario())


def test_lab_revocation_blocks_approval_and_resume(tmp_path):
    async def scenario():
        manager, gateway = runtime(tmp_path, FakeModel(tool("save_note", text="note")))
        run_id = manager.start("alice", "environment", "Plan")["run_id"]
        await settled(manager, run_id, "awaiting_approval")
        manager.permitted_labs = frozenset()
        with pytest.raises(RunNotFound):
            manager.decide("alice", run_id, True)
        with pytest.raises(RunNotFound):
            manager.get("alice", run_id)
        manager._runs[run_id].task.cancel()
        await asyncio.sleep(.01)
        assert not any(name == "save_note" for name, _ in gateway.calls)
    asyncio.run(scenario())


def test_approval_timeout_and_concurrency_budget(tmp_path):
    async def scenario():
        manager, gateway = runtime(tmp_path, FakeModel(tool("save_note", text="note")),
                                   limits=AgentLimits(approval_seconds=.03, max_active=1))
        run_id = manager.start("alice", "environment", "Plan")["run_id"]
        with pytest.raises(RunConflict):
            manager.start("alice", "environment", "Another plan")
        result = await settled(manager, run_id)
        assert result["status"] == "exhausted" and result["pending_action"] is None
        assert not any(name == "save_note" for name, _ in gateway.calls)
    asyncio.run(scenario())


# Use real Tornado/Jupyter request preparation and authentication decorators.
from jupyter_server.auth import IdentityProvider, User
from tornado.testing import AsyncHTTPTestCase, gen_test
from tornado.web import Application
from naavre_ai_assistant.agent_handlers import (
    AgentRunsHandler, AgentRunHandler, AgentDecisionHandler, AgentCancelHandler,
    LabsHandler, LabDocumentsHandler, LabKnowledgeHandler,
)


class TestIdentityProvider(IdentityProvider):
    __test__ = False

    def get_user(self, handler):
        username = handler.request.headers.get("X-Test-User")
        if username:
            handler._token_authenticated = handler.request.headers.get("X-Test-Cookie-Auth") != "true"
            return User(username)
        return None


class TestAgentHTTP(AsyncHTTPTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        self.temporary.cleanup()

    def get_app(self):
        self.manager, self.gateway = runtime(Path(self.temporary.name), FakeModel(
            tool("save_note", text="exact HTTP note"), finish()))
        common = {"runtime": self.manager}
        return Application([
            (r"/labs", LabsHandler, common),
            (r"/labs/([^/]+)/documents", LabDocumentsHandler, common),
            (r"/labs/([^/]+)/knowledge", LabKnowledgeHandler, common),
            (r"/agent/runs", AgentRunsHandler, common),
            (r"/agent/runs/([^/]+)", AgentRunHandler, common),
            (r"/agent/runs/([^/]+)/decision", AgentDecisionHandler, common),
            (r"/agent/runs/([^/]+)/cancel", AgentCancelHandler, common),
        ], identity_provider=TestIdentityProvider(), cookie_secret="test-secret",
            base_url="/", login_url="/login", allow_remote_access=True)

    async def api(self, path, body=None, owner="alice"):
        headers = {"X-Test-User": owner} if owner else {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        return await self.http_client.fetch(
            self.get_url(path), method="POST" if body is not None else "GET",
            headers=headers, body=json.dumps(body) if body is not None else None,
            raise_error=False, follow_redirects=False)

    @gen_test
    async def test_http_jobs_owner_approval_cancel_and_auth(self):
        denied = await self.api("/agent/runs", {"lab_id": "environment", "description": "Plan"}, owner=None)
        assert denied.code == 403
        started = await self.api("/agent/runs", {"lab_id": "environment", "description": "Plan"})
        assert started.code == 202
        run_id = json.loads(started.body)["run_id"]
        await settled(self.manager, run_id, "awaiting_approval")
        assert (await self.api(f"/agent/runs/{run_id}", owner="mallory")).code == 404
        assert (await self.api(f"/agent/runs/{run_id}/cancel", {}, owner="mallory")).code == 404
        assert (await self.api(f"/agent/runs/{run_id}/decision", {"approve": True}, owner="mallory")).code == 404
        assert (await self.api(f"/agent/runs/{run_id}/decision", {"approve": True, "arguments": {}})).code == 400
        assert (await self.api(f"/agent/runs/{run_id}/decision", {"approve": "true"})).code == 400
        assert (await self.api(f"/agent/runs/{run_id}/decision", {"approve": True})).code == 200
        result = await settled(self.manager, run_id)
        assert result["status"] == "completed"
        assert (await self.api(f"/agent/runs/{run_id}/decision", {"approve": True})).code == 409
        assert ("save_note", {"text": "exact HTTP note"}) in self.gateway.calls

    @gen_test
    async def test_http_lab_contract_upload_validation_and_server_permissions(self):
        response = await self.api("/labs")
        assert response.code == 200
        lab = json.loads(response.body)["labs"][0]
        assert set(lab) == {"id", "name", "description", "style", "frameworks", "document_count", "skill_count"}
        assert lab["document_count"] == lab["skill_count"] == 1
        knowledge = await self.api("/labs/environment/knowledge?q=species")
        assert knowledge.code == 200
        assert set(json.loads(knowledge.body)) == {"documents", "skills", "truncated"}
        knowledge = await self.api("/labs/environment/knowledge")
        assert json.loads(knowledge.body)["documents"] == [{"path": "reference.txt"}]
        payload = {"filename": "ref.txt", "content_base64": base64.b64encode(b"reference").decode()}
        response = await self.api("/labs/environment/documents", payload)
        assert response.code == 201 and "private_root" not in json.loads(response.body)
        assert self.manager.registry.value.upload == ("ref.txt", b"reference")
        for filename in ("../ref.txt", "C:\\ref.txt", "/ref.txt", "ref.txt."):
            response = await self.api("/labs/environment/documents", {**payload, "filename": filename})
            assert response.code == 400
        assert (await self.api("/labs/environment/documents", {**payload, "content_base64": "!!!"})).code == 400
        self.manager.permitted_labs = frozenset()
        assert (await self.api("/labs/environment/documents", payload)).code == 404
        assert (await self.api("/labs/environment/knowledge")).code == 404
        assert json.loads((await self.api("/labs")).body) == {"labs": []}

    @gen_test
    async def test_cookie_authenticated_post_requires_xsrf(self):
        response = await self.http_client.fetch(
            self.get_url("/agent/runs"), method="POST",
            headers={"X-Test-User": "alice", "X-Test-Cookie-Auth": "true",
                     "Content-Type": "application/json"},
            body=json.dumps({"lab_id": "environment", "description": "Plan"}),
            raise_error=False)
        assert response.code == 403
        assert not self.manager._runs


def test_separate_agent_deadline_configuration_and_fenced_decisions(tmp_path, monkeypatch):
    from naavre_ai_assistant.agent_handlers import configured_limits
    monkeypatch.setenv("NAAVRE_AGENT_MAX_STEPS", "6")
    monkeypatch.setenv("NAAVRE_AGENT_RUN_SECONDS", "120")
    monkeypatch.setenv("NAAVRE_AGENT_OPERATION_SECONDS", "off")
    monkeypatch.setenv("NAAVRE_AGENT_APPROVAL_SECONDS", "none")
    limits = configured_limits({"naavre_ai_assistant_agent_limits": {"max_steps": 3}})
    assert limits.max_steps == 3 and limits.run_seconds == 120
    assert limits.operation_seconds is None and limits.approval_seconds is None
    with pytest.raises(ValueError):
        AgentLimits(max_steps=1.5)
    with pytest.raises(ValueError):
        AgentLimits(run_seconds=float("nan"))
    async def scenario():
        manager, _ = runtime(tmp_path, FakeModel("```json\n" + json.dumps(finish()) + "\n```"))
        run_id = manager.start("alice", "environment", "Plan")["run_id"]
        assert (await settled(manager, run_id))["status"] == "completed"
    asyncio.run(scenario())


def test_routes_use_server_settings_and_state_environment(tmp_path, monkeypatch):
    from naavre_ai_assistant.agent_handlers import agent_routes
    from naavre_ai_assistant.llm_service import LLMCodeGenerator
    monkeypatch.setenv("NAAVRE_AGENT_STATE_DIR", str(tmp_path / "agent-state"))
    web_app = SimpleNamespace(settings={"base_url": "/user/alice/",
                                        "naavre_ai_assistant_labs_root": str(tmp_path / "labs"),
                                        "naavre_ai_assistant_permitted_labs": []})
    routes = agent_routes(web_app, LLMCodeGenerator())
    assert len(routes) == 7
    assert all(route[0].startswith("/user/alice/naavre-ai-assistant/") for route in routes)
    manager = web_app.settings["naavre_ai_assistant_agent_runtime"]
    assert manager.root == tmp_path / "agent-state"
    assert manager.registry.permitted_labs == frozenset()
