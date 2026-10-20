import json
from types import SimpleNamespace

import pytest
import naavre_ai_assistant.llm_service as llm_service

from naavre_ai_assistant.llm_service import (
    DEFAULT_LOCAL_LLM_BASE_URL,
    DEFAULT_LOCAL_LLM_MODEL,
    LLMCodeGenerator,
    LLMConfig,
    LLMServiceError,
    _generation_retry_guidance,
    strip_code_fence,
)


def valid_analysis(**overrides: object) -> str:
    analysis = {
        "intent": "Select the fifth data record and write it.",
        "summary": "Select the fifth data record and write it to the declared result.",
        "relevant_inputs": ["records.csv: tabular records"],
        "relevant_outputs": ["result.csv: selected record"],
        "data_semantics": {
            "headers_are_schema_labels": True,
            "ordinal_targets_data_records_unless_explicit_header": True,
            "notes": ["The fifth row means the fifth data record after the CSV header."],
        },
        "workflow_steps": ["Select data record five."],
        "visualization_plan": ["Show the selected record for review."],
        "assumptions": ["Only declared metadata is available; raw CSV was not inspected."],
        "validation_requirements": ["Verify that at least five data records exist."],
        "implementation_plan": ["Read the CSV and select data record five."],
        "cell_plan": [
            {
                "title": "Select the requested record",
                "purpose": "Select data record five from the declared input.",
                "inputs": ["records.csv"],
                "outputs": ["selected_record"],
            }
        ],
    }
    analysis.update(overrides)
    return json.dumps(analysis)


class FakeCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = iter(contents)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        content = next(self.contents)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeClient:
    def __init__(self, *contents: str) -> None:
        self.completions = FakeCompletions(list(contents))
        self.chat = SimpleNamespace(completions=self.completions)


def config() -> LLMConfig:
    return LLMConfig(
        base_url=DEFAULT_LOCAL_LLM_BASE_URL,
        model=DEFAULT_LOCAL_LLM_MODEL,
        timeout_seconds=5,
        max_tokens=500,
    )


def test_strip_code_fence() -> None:
    assert strip_code_fence("```python\nprint('ok')\n```") == "print('ok')"
    assert strip_code_fence("print('ok')") == "print('ok')"


def test_generate_cells_uses_two_steps_and_analysis() -> None:
    analysis = json.loads(valid_analysis())
    analysis["workflow_steps"] = [
        "Read the input.",
        "Select data record five.",
        "Write the result.",
    ]
    analysis["cell_plan"] = [
        {
            "title": "Read input",
            "purpose": "Read the declared CSV input.",
            "inputs": ["records.csv"],
            "outputs": ["records"],
        },
        {
            "title": "Select record",
            "purpose": "Select data record five.",
            "inputs": ["records"],
            "outputs": ["selected_record"],
        },
        {
            "title": "Write result",
            "purpose": "Write the selected record.",
            "inputs": ["selected_record"],
            "outputs": ["result.csv"],
        },
    ]
    client = FakeClient(
        f"```json\n{json.dumps(analysis)}\n```",
        "from pathlib import Path\nPath('outputs').mkdir(parents=True, exist_ok=True)",
        "selected_record = records[4]",
        "print(selected_record)",
    )
    service = LLMCodeGenerator(config(), client)

    cells = service.generate_cells(
        description="Write the 5th data row.",
        inputs={"records.csv": "CSV with a header row"},
        outputs={"result.csv": "CSV"},
        metadata={"language": "python"},
        context={"available_files": ["records.csv"], "cells": [{"type": "markdown"}]},
    )

    assert len(cells) == 3
    assert cells[0]["code"].startswith("from pathlib")
    assert cells[1]["title"] == "Select record"
    assert len(client.completions.requests) == 4
    analysis_prompt = client.completions.requests[0]["messages"][1]["content"]
    code_prompt = client.completions.requests[1]["messages"][1]["content"]
    assert "headers are schema labels, not data elements or rows" in analysis_prompt
    assert "against data records" in analysis_prompt
    assert '"headers_are_schema_labels": true' in code_prompt
    assert "notebook-relative outputs/" in code_prompt
    assert '"records.csv": "CSV with a header row"' in code_prompt
    assert '"available_files"' in code_prompt
    assert "Generate cell 1 of 3 only" in code_prompt
    assert "Do not return JSON" in code_prompt
    assert "Target cell plan entry" in code_prompt
    third_cell_prompt = client.completions.requests[3]["messages"][1]["content"]
    assert "selected_record = records[4]" in third_cell_prompt
    assert "Do not collapse" in code_prompt
    assert client.completions.requests[0]["temperature"] == 0.0
    assert client.completions.requests[1]["temperature"] == 0.2


def test_repair_code_uses_two_steps_and_original_metadata() -> None:
    client = FakeClient(valid_analysis(), "print(value)")
    service = LLMCodeGenerator(config(), client)

    service.repair_code(
        "Print the fifth value",
        "print(valeu)",
        "NameError: valeu",
        attempt=2,
        inputs={"records.csv": "CSV schema: value"},
        outputs={"result": "fifth data value"},
        metadata={"language": "python"},
        context={"available_files": ["records.csv"]},
    )

    assert len(client.completions.requests) == 2
    analysis_prompt = client.completions.requests[0]["messages"][1]["content"]
    repair_prompt = client.completions.requests[1]["messages"][1]["content"]
    assert "NameError: valeu" in analysis_prompt
    assert "NameError: valeu" in repair_prompt
    assert "attempt 2" in repair_prompt
    assert '"records.csv": "CSV schema: value"' in repair_prompt
    assert '"headers_are_schema_labels": true' in repair_prompt


def test_explicit_analysis_phase_is_consumed_by_one_code_call() -> None:
    client = FakeClient("print('generated')")
    service = LLMCodeGenerator(config(), client)
    analysis = json.loads(valid_analysis())

    cells = service.generate_cells_from_analysis(
        description="Create a review plot.",
        language="python",
        metadata={"visualization_requested": True},
        context={"available_files": ["records.csv"]},
        analysis=analysis,
    )

    assert cells[0]["code"] == "print('generated')"
    assert len(client.completions.requests) == 1
    prompt = client.completions.requests[0]["messages"][1]["content"]
    assert '"visualization_plan"' in prompt
    assert '"visualization_requested": true' in prompt


def test_analysis_phase_includes_context_without_claiming_file_inspection() -> None:
    client = FakeClient(valid_analysis())
    service = LLMCodeGenerator(config(), client)

    analysis = service.analyze_request(
        "Summarize the notebook workflow.",
        context={"available_files": ["/tmp/data/observations.csv"], "cells": []},
    )

    assert analysis["summary"]
    assert len(client.completions.requests) == 1
    prompt = client.completions.requests[0]["messages"][1]["content"]
    assert "/tmp/data/observations.csv" in prompt
    assert "not raw file content" in prompt


def test_analysis_reconciles_summary_steps_with_executable_cell_plan() -> None:
    analysis = json.loads(valid_analysis())
    analysis["workflow_steps"] = [
        "Read the file.",
        "Validate and transform the records.",
        "Write and visualize the result.",
    ]
    client = FakeClient(json.dumps(analysis))
    service = LLMCodeGenerator(config(), client)

    result = service.analyze_request("Analyze the records.")

    assert result["workflow_steps"] == [
        "Select the requested record: Select data record five from the declared input"
    ]


def test_analysis_replans_combined_jobs_with_correct_dependencies() -> None:
    analysis = json.loads(valid_analysis())
    analysis["cell_plan"] = [
        {
            "title": "Compare stations and export results",
            "purpose": "Compare station values and export the comparison.",
            "inputs": ["records"],
            "outputs": ["comparison.csv"],
        }
    ]
    revised = dict(analysis, cell_plan=[
        {"title": "Compare stations", "purpose": "Compute station comparison", "inputs": ["records"], "outputs": ["comparison"]},
        {"title": "Export results", "purpose": "Write comparison CSV", "inputs": ["comparison"], "outputs": ["comparison.csv"]},
    ])
    client = FakeClient(json.dumps(analysis), json.dumps(revised))
    service = LLMCodeGenerator(config(), client)

    result = service.analyze_request("Compare and export the records.")

    assert [cell["title"] for cell in result["cell_plan"]] == [
        "Compare stations",
        "Export results",
    ]
    assert len(result["workflow_steps"]) == 2
    assert result['cell_plan'][1]['inputs'] == ['comparison']
    assert len(client.completions.requests) == 2


def test_noun_lists_are_not_split_into_fragment_cells():
    analysis = json.loads(valid_analysis())
    analysis['cell_plan'] = [{
        'title': 'Assess quality', 'purpose': 'Assess temperature and salinity quality',
        'inputs': ['records'], 'outputs': ['quality'],
    }]
    service = LLMCodeGenerator(config(), FakeClient(json.dumps(analysis)))
    result = service.analyze_request('Assess quality.')
    assert len(result['cell_plan']) == 1
    assert result['cell_plan'][0]['purpose'] == 'Assess temperature and salinity quality'


def test_grouping_and_aggregation_is_one_focused_analysis_job():
    analysis = json.loads(valid_analysis())
    analysis['cell_plan'] = [{
        'title': 'Summarize mean height',
        'purpose': 'Group measurements by treatment and calculate mean height by date.',
        'inputs': ['validated_rows'], 'outputs': ['treatment_means'],
    }]
    service = LLMCodeGenerator(config(), FakeClient(json.dumps(analysis)))
    result = service.analyze_request('Summarize mean height by treatment and date.')
    assert len(result['cell_plan']) == 1


def test_endpoint_selection_and_height_gain_table_do_not_trigger_replanning():
    purpose = (
        "For each plant in sorted_eligible_records, identify the first and last "
        "height_cm values and compute height_gain_cm as their difference; produce "
        "a per-plant table with plant_id, treatment, first_date, last_date, "
        "first_height_cm, last_height_cm, and height_gain_cm."
    )
    analysis = json.loads(valid_analysis(cell_plan=[{
        "title": "Compute per-plant height gain", "purpose": purpose,
        "inputs": ["sorted_eligible_records"], "outputs": ["plant_gains"],
    }]))
    client = FakeClient(json.dumps(analysis), "plant_gains = []")
    service = LLMCodeGenerator(config(), client)
    cells = service.generate_cells("Compute per-plant height gain.")
    assert len(cells) == 1
    assert cells[0]["purpose"] == purpose.rstrip(".")
    assert len(client.completions.requests) == 2  # one analysis, one code request


def test_planning_progress_reports_stages_without_logging_prompt_or_source(caplog):
    client = FakeClient(valid_analysis(), "private_result = 42")
    service = LLMCodeGenerator(config(), client)
    with caplog.at_level("INFO", logger=llm_service.__name__):
        service.generate_cells("Private scientist request marker")
    assert "planning attempt 1/6 started" in caplog.text
    assert "kind=structured analysis" in caplog.text
    assert "generating cell 1/1 (attempt 1/3)" in caplog.text
    assert "response received: kind=cell 1" in caplog.text
    assert "Private scientist request marker" not in caplog.text
    assert "private_result" not in caplog.text


def test_progress_can_use_jupyter_server_logger(caplog):
    import logging
    logger = logging.getLogger("test.ServerApp")
    service = LLMCodeGenerator(config(), FakeClient(valid_analysis()), logger=logger)
    with caplog.at_level("INFO", logger=logger.name):
        service.analyze_request("Compute a result")
    assert any(record.name == logger.name and "planning attempt" in record.message
               for record in caplog.records)


@pytest.mark.parametrize("purpose", [
    "Identify the first and last heights and compute their difference.",
    "For each plant, identify its endpoints and calculate height gain.",
    "Group records by treatment and compute mean height; build a result table.",
    "Compute height gains and store them in a kernel variable.",
    "Create a line chart, then render the chart inline.",
    "Validate schema and check missing values.",
    "Assess temperature and salinity quality.",
])
def test_cohesive_suboperations_are_not_rejected(purpose):
    assert not llm_service._has_combined_actions(purpose)


@pytest.mark.parametrize("purpose", [
    "Load records and validate the schema.",
    "Normalize heights and compute treatment means.",
    "Compute height gains and export the table to CSV.",
    "For each plant, identify endpoints and compute gain; save results to a file.",
    "Group records and calculate means, then create a chart.",
    "Compute a result. Plot a chart.",
    "Create an SVG chart and write the SVG file.",
])
def test_distinct_pipeline_phases_still_require_separate_cells(purpose):
    assert llm_service._has_combined_actions(purpose)


def test_replanning_reports_all_combined_jobs_in_one_correction():
    plans = [
        {"title": "Load and validate", "purpose": "Read rows and check the schema",
         "inputs": ["input.csv"], "outputs": ["rows"]},
        {"title": "Compute and export", "purpose": "Calculate means and export CSV",
         "inputs": ["rows"], "outputs": ["means.csv"]},
    ]
    with pytest.raises(LLMServiceError) as rejected:
        llm_service._validate_cell_plan(plans)
    assert "cell_plan entry 1" in str(rejected.value)
    assert "cell_plan entry 2" in str(rejected.value)
    assert "Replan all these entries" in str(rejected.value)


def test_plot_creation_after_aggregation_is_split_into_a_new_cell():
    analysis = json.loads(valid_analysis())
    analysis['cell_plan'] = [{
        'title': 'Summarize and create a chart',
        'purpose': 'Group records by date and calculate means, then create a line chart.',
        'inputs': ['records'], 'outputs': ['means', 'trend.svg'],
    }]
    revised = dict(analysis, cell_plan=[
        {'title': 'Summarize means', 'purpose': 'Group records by date and calculate means.',
         'inputs': ['records'], 'outputs': ['means']},
        {'title': 'Create line chart', 'purpose': 'Create a line chart from means.',
         'inputs': ['means'], 'outputs': ['trend.svg']},
    ])
    client = FakeClient(json.dumps(analysis), json.dumps(revised))
    service = LLMCodeGenerator(config(), client)
    result = service.analyze_request('Create an SVG and save the chart.')
    assert [step['title'] for step in result['cell_plan']] == ['Summarize means', 'Create line chart']
    assert len(client.completions.requests) == 2


def test_comma_separated_pipeline_jobs_require_replanning():
    analysis = json.loads(valid_analysis())
    analysis["cell_plan"] = [{
        "title": "Generate mean height trends by treatment over time",
        "purpose": (
            "Read plant growth data, compute mean heights per treatment and date, "
            "create SVG visualization, and display it inline"
        ),
        "inputs": ["plant_growth.csv"],
        "outputs": ["plant_height_trends.svg"],
    }]
    revised = dict(analysis, cell_plan=[
        {"title": "Read plant growth data", "purpose": "Read CSV rows using the verified schema.",
         "inputs": ["plant_growth.csv"], "outputs": ["validated_rows"]},
        {"title": "Summarize height by treatment and date",
         "purpose": "Compute the mean height for each treatment and date.",
         "inputs": ["validated_rows"], "outputs": ["treatment_means"]},
        {"title": "Create the treatment trend chart",
         "purpose": "Render the requested SVG chart file from the calculated means.",
         "inputs": ["treatment_means"], "outputs": ["plant_height_trends.svg"]},
    ])
    client = FakeClient(json.dumps(analysis), json.dumps(revised))

    result = LLMCodeGenerator(config(), client).analyze_request(
        "Read plant growth data, compare mean height by treatment over time, and create a chart."
    )

    assert len(result["cell_plan"]) == 3
    assert "previous analysis was rejected" in client.completions.requests[1]["messages"][1]["content"]


def test_explicit_output_artifact_does_not_get_noop_save_cell():
    analysis = json.loads(valid_analysis())
    analysis['cell_plan'] = [{
        'title': 'Save chart', 'purpose': 'Save the final SVG chart for downstream use.',
        'inputs': ['means'], 'outputs': ['trend.svg'],
    }]
    client = FakeClient(json.dumps(analysis))
    result = LLMCodeGenerator(config(), client).analyze_request('Create and save a chart.')
    assert len(result['cell_plan']) == 1


def test_analysis_adds_a_dedicated_persistence_cell_when_requested() -> None:
    analysis = json.loads(valid_analysis())
    analysis["cell_plan"] = [
        {
            "title": "Assess observations",
            "purpose": "Compute the requested quality measures.",
            "inputs": ["observations.csv"],
            "outputs": ["quality_metrics"],
        }
    ]
    client = FakeClient(json.dumps(analysis))
    service = LLMCodeGenerator(config(), client)

    result = service.analyze_request(
        "Assess the observations and save the results for another workflow."
    )

    assert result["cell_plan"][-1]["title"] == "Save results for downstream use"
    assert " and " not in result["cell_plan"][-1]["purpose"].lower()
    assert len(result["workflow_steps"]) == 2


@pytest.mark.parametrize("analysis", ["", "not JSON", "{}"])
def test_malformed_or_empty_analysis_is_reported(analysis: str) -> None:
    client = FakeClient(analysis, "print('must not be requested')")
    service = LLMCodeGenerator(config(), client)

    with pytest.raises(LLMServiceError, match="analysis"):
        service.generate_code("Print a value")

    assert len(client.completions.requests) == 1


def test_analysis_must_enforce_csv_row_semantics() -> None:
    invalid = json.loads(valid_analysis())
    invalid["data_semantics"]["headers_are_schema_labels"] = False
    service = LLMCodeGenerator(config(), FakeClient(json.dumps(invalid)))

    with pytest.raises(LLMServiceError, match="CSV headers"):
        service.generate_code("Read the fifth row")


def test_analysis_allows_no_declared_inputs_or_outputs() -> None:
    analysis = json.loads(valid_analysis())
    analysis["relevant_inputs"] = []
    analysis["relevant_outputs"] = []
    service = LLMCodeGenerator(config(), FakeClient(json.dumps(analysis), "print(1)"))

    assert service.generate_code("Print one") == "print(1)"


def test_local_defaults_need_no_cloud_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "LOCAL_LLM_BASE_URL",
        "LOCAL_LLM_MODEL",
        "LOCAL_LLM_TIMEOUT_SECONDS",
        "LOCAL_LLM_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    local_config = LLMConfig.from_environment()
    service = LLMCodeGenerator(local_config, FakeClient(valid_analysis(), "pass"))

    assert service.configured
    assert local_config.base_url == DEFAULT_LOCAL_LLM_BASE_URL
    assert local_config.model == DEFAULT_LOCAL_LLM_MODEL
    assert local_config.timeout_seconds is None
    assert local_config.max_tokens is None
    assert service.generate_code("Do work") == "pass"


@pytest.mark.parametrize("language", ["javascript", "", "sql"])
def test_unsupported_language_is_rejected(language: str) -> None:
    service = LLMCodeGenerator(config(), FakeClient("unused"))

    with pytest.raises(ValueError, match="python.*r"):
        service.generate_code("Do work", language=language)


def test_non_string_language_is_rejected() -> None:
    service = LLMCodeGenerator(config(), FakeClient("unused"))

    with pytest.raises(ValueError, match="python.*r"):
        service.generate_code("Do work", language=None)  # type: ignore[arg-type]


def test_oversized_metadata_is_rejected() -> None:
    service = LLMCodeGenerator(config(), FakeClient("unused"))

    with pytest.raises(ValueError, match="metadata exceeds"):
        service.generate_code("Do work", metadata={"content": "x" * 50_001})


def test_oversized_combined_context_is_rejected() -> None:
    service = LLMCodeGenerator(config(), FakeClient("unused"))

    with pytest.raises(ValueError, match="Combined request context exceeds"):
        service.generate_code(
            "x" * 40_000,
            inputs={"value": "x" * 30_000},
            outputs={"value": "x" * 30_000},
        )


def test_configuration_error_disables_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_LLM_TIMEOUT_SECONDS", "not-a-number")

    service = LLMCodeGenerator()

    assert not service.configured
    with pytest.raises(LLMServiceError, match="must be numeric"):
        service.generate_code("Do work")


def test_generated_cells_reject_unavailable_python_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    # This policy must work whether or not the optional analysis extra is installed.
    original_available = llm_service._module_available
    monkeypatch.setattr(llm_service, "_module_available",
                        lambda module: module != 'pandas' and original_available(module))
    invalid_cells = "import pandas as pd\nprint(pd)"
    client = FakeClient(invalid_cells, invalid_cells, invalid_cells)
    service = LLMCodeGenerator(config(), client)

    with pytest.raises(LLMServiceError, match="unavailable module 'pandas'"):
        service.generate_code_from_analysis(
            "Load a value",
            analysis=json.loads(valid_analysis()),
        )


def test_http_transport_targets_local_ollama_without_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[object] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"message": {"role": "assistant", "content": "print('local')"}}
            ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(llm_service, "urlopen", fake_urlopen)
    service = LLMCodeGenerator(config())

    assert service._complete_code("Generate a local test cell") == "print('local')"
    request, timeout = requests[0]
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert request.get_header("Authorization") is None
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["model"] == DEFAULT_LOCAL_LLM_MODEL
    assert payload["options"] == {"temperature": 0.2, "num_predict": 500, "num_ctx": 32768}
    assert timeout == 5


def test_unlimited_configuration_uses_native_ollama_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"message": {"role": "assistant", "content": "print('local')"}}
            ).encode("utf-8")

    def fake_urlopen(*args: object, **kwargs: object) -> FakeResponse:
        requests.append((args, kwargs))
        return FakeResponse()

    monkeypatch.setattr(llm_service, "urlopen", fake_urlopen)
    service = LLMCodeGenerator(
        LLMConfig(
            base_url=DEFAULT_LOCAL_LLM_BASE_URL,
            model=DEFAULT_LOCAL_LLM_MODEL,
            timeout_seconds=None,
            max_tokens=None,
        )
    )

    assert service._complete_code("Generate a local test cell") == "print('local')"
    request_args, request_kwargs = requests[0]
    payload = json.loads(request_args[0].data.decode("utf-8"))
    assert request_kwargs == {}
    assert payload["options"]["num_predict"] == -1
    assert payload["options"]["num_ctx"] == 32768


def test_json_mode_asks_native_ollama_for_valid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[object] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"message": {"content": '{"action":"finish"}'}}).encode()

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        requests.append(request)
        return FakeResponse()

    monkeypatch.setattr(llm_service, "urlopen", fake_urlopen)
    service = LLMCodeGenerator(config())
    response = service._request_content(
        "Choose an agent action", system_message="Return JSON", temperature=0,
        response_kind="agent decision", json_mode=True,
    )

    assert response == '{"action":"finish"}'
    payload = json.loads(requests[0].data.decode("utf-8"))
    assert payload["format"] == "json"


def test_agent_schema_is_sent_to_native_ollama_without_changing_limits(monkeypatch):
    requests = []
    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps({"message": {"content": '{"action":"finish","summary":"Reviewed plan"}'}}).encode()
    def fake_urlopen(request, **kwargs):
        requests.append((request, kwargs))
        return FakeResponse()
    monkeypatch.setattr(llm_service, "urlopen", fake_urlopen)
    schema = {"type": "object", "properties": {"action": {"const": "finish"}, "summary": {"type": "string"}},
              "required": ["action", "summary"]}
    service = LLMCodeGenerator(LLMConfig(DEFAULT_LOCAL_LLM_BASE_URL, DEFAULT_LOCAL_LLM_MODEL, None, None))
    service._request_content("Plan", system_message="Return JSON", temperature=0,
                             json_mode=True, response_schema=schema)
    request, kwargs = requests[0]
    payload = json.loads(request.data)
    assert payload["format"] == schema
    assert payload["options"]["num_predict"] == -1
    assert payload["options"]["num_ctx"] == 32768
    assert kwargs == {}


@pytest.mark.parametrize("kind,expected", [
    ("agent decision", "qwen3.8:27b"), ("structured analysis", "qwen3.5:9b"),
    ("cell 3", "qwen3-coder:30b"), ("code", "qwen3-coder:30b"),
])
def test_role_routing_targets_native_local_model_and_keeps_thinking_separate(monkeypatch, kind, expected):
    requests = []
    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps({"message": {"thinking": "private reasoning", "content": "final answer"}}).encode()
    def fake_urlopen(request, **kwargs):
        requests.append((request, kwargs))
        return FakeResponse()
    monkeypatch.setattr(llm_service, "urlopen", fake_urlopen)
    settings = LLMConfig(DEFAULT_LOCAL_LLM_BASE_URL, DEFAULT_LOCAL_LLM_MODEL, None, None,
                         agent_model="qwen3.8:27b", analysis_model="qwen3.5:9b", thinking=False)
    service = LLMCodeGenerator(settings)
    assert service._request_content("Task", system_message="Test", temperature=0, response_kind=kind) == "final answer"
    request, kwargs = requests[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert payload["model"] == expected and payload["think"] is False
    assert payload["options"]["num_predict"] == -1 and kwargs == {}


def test_new_model_environment_defaults_and_optional_roles(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen3.8:27b")
    monkeypatch.setenv("LOCAL_LLM_AGENT_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("LOCAL_LLM_ANALYSIS_MODEL", "")
    monkeypatch.setenv("LOCAL_LLM_CODE_MODEL", DEFAULT_LOCAL_LLM_MODEL)
    monkeypatch.setenv("LOCAL_LLM_THINK", "auto")
    settings = LLMConfig.from_environment()
    assert settings.configuration_error is None
    assert settings.model_roles() == {"agent": "qwen3.5:4b", "analysis": "qwen3.8:27b", "code": DEFAULT_LOCAL_LLM_MODEL}
    assert settings.thinking is None


@pytest.mark.parametrize("variable,value", [
    ("LOCAL_LLM_MODEL", "glm-5.3:cloud"), ("LOCAL_LLM_AGENT_MODEL", "qwen3.8:cloud"),
    ("LOCAL_LLM_CODE_MODEL", "unknown:model"), ("LOCAL_LLM_MODEL", ""),
    ("LOCAL_LLM_THINK", "sometimes"),
])
def test_invalid_role_or_thinking_configuration_fails_without_cloud_fallback(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    settings = LLMConfig.from_environment()
    assert settings.configuration_error
    service = LLMCodeGenerator(settings, FakeClient("must never be called"))
    with pytest.raises(LLMServiceError):
        service._request_content("Task", system_message="Test", temperature=0)
    assert service._client.completions.requests == []


def test_context_window_is_separate_from_output_cap(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_CONTEXT_WINDOW", "65536")
    monkeypatch.setenv("LOCAL_LLM_MAX_TOKENS", "unlimited")
    monkeypatch.setenv("LOCAL_LLM_TIMEOUT_SECONDS", "unlimited")
    settings = LLMConfig.from_environment()
    assert settings.context_window == 65536
    assert settings.max_tokens is None
    assert settings.timeout_seconds is None
    assert settings.configuration_error is None


def test_repeated_cell_is_regenerated():
    client = FakeClient("result = 1", "# repeated\nresult = 1", "print(result)")
    service = LLMCodeGenerator(config(), client)
    plan = [
        {"title": "Compute", "purpose": "Compute value", "inputs": [], "outputs": ["result"]},
        {"title": "Display", "purpose": "Display value", "inputs": ["result"], "outputs": []},
    ]
    cells = service._complete_cells("Compute a value.", cell_plan=plan, language="python")
    assert len(cells) == 2
    assert cells[1]["code"] == "print(result)"
    assert "repeats an earlier cell" in client.completions.requests[-1]["messages"][1]["content"]


def test_invalid_cell_retry_includes_candidate_and_syntax_detail():
    invalid = "if True print('broken')"
    client = FakeClient(invalid, "print('repaired')")
    service = LLMCodeGenerator(config(), client)
    plan = [
        {"title": "Display", "purpose": "Display a result", "inputs": [], "outputs": []},
    ]

    cells = service._complete_cells("Display a result.", cell_plan=plan, language="python")

    assert cells[0]["code"] == "print('repaired')"
    retry_prompt = client.completions.requests[-1]["messages"][1]["content"]
    assert invalid in retry_prompt
    assert "invalid syntax" in retry_prompt
    assert "invalid syntax at line 1" in retry_prompt


def test_filename_attribute_assignment_is_rejected():
    with pytest.raises(LLMServiceError, match="filename variable"):
        llm_service._validate_python_code_imports(
            "temperature_trend_plot.svg = str(output_file)", cell_number=5
        )


def test_normal_output_path_assignment_is_allowed():
    llm_service._validate_python_code_imports(
        "temperature_trend_plot_path = str(output_file)"
    )


def test_schema_retry_guidance_distinguishes_derived_values_from_raw_columns():
    guidance = _generation_retry_guidance(
        LLMServiceError("Unverified input columns: mean_height_cm"), "python"
    )
    assert "exact verified CSV headers" in guidance
    assert "calculate it from the verified source column" in guidance


def test_copy_from_undefined_filename_variable_is_rejected():
    with pytest.raises(LLMServiceError, match="undefined path variable"):
        llm_service._validate_python_code_imports(
            "import shutil\nshutil.copy2(temperature_trend_plot, plot_output_file)",
            cell_number=6,
            previous_code="temperature_trend_plot_path = 'plot.svg'",
        )


def test_copy_from_defined_path_variable_is_allowed():
    llm_service._validate_python_code_imports(
        "import shutil\nshutil.copy2(temperature_trend_plot_path, plot_output_file)",
        cell_number=6,
        previous_code="temperature_trend_plot_path = 'plot.svg'",
    )


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        (
            "LOCAL_LLM_BASE_URL",
            "https://api.example.invalid/v1",
            "local Ollama endpoint",
        ),
        ("LOCAL_LLM_MODEL", "some-cloud-model", "supported local Ollama model"),
    ],
)
def test_environment_rejects_non_local_llm_configuration(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv(variable, value)

    config_from_environment = LLMConfig.from_environment()

    assert config_from_environment.configuration_error
    assert message in config_from_environment.configuration_error
