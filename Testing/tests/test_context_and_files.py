import copy
import json
import pytest

from naavre_ai_assistant.file_context import (
    file_reference_error, request_file_references, resolve_reference,
)
from naavre_ai_assistant.notebook_context import (
    NOTEBOOK_CONTEXT_BUDGET, prepare_notebook_context,
)
from naavre_ai_assistant.llm_service import LLMCodeGenerator
from test_llm_service import FakeClient, config, valid_analysis


def oversized_notebook():
    return {
        "notebook": {"path": "Untitled7.ipynb", "cell_count": 60, "active_cell_index": 55},
        "available_files": ["observations.csv"],
        "cells": [
            {
                "index": i, "type": "code", "source": "# temperature ö 🌡\n" * 350,
                "metadata": {
                    "naavre": {"inputs": {"observations.csv": "CSV schema"}},
                    "naavre-ai-assistant": {
                        "title": f"Job {i}", "analysis": {"summary": "DUPLICATE_PLAN" * 1000},
                    },
                },
                "outputs": [{"data": {"image/png": "SHOULD_NOT_BE_SENT" * 1000}}],
            } for i in range(60)
        ],
    }


def test_large_notebook_selection_fits_serialized_budget_without_editing_original():
    original = oversized_notebook()
    before = copy.deepcopy(original)
    selected = prepare_notebook_context(original)
    serialized = json.dumps(selected, indent=2, sort_keys=True)
    assert len(serialized) <= NOTEBOOK_CONTEXT_BUDGET
    assert original == before
    assert selected["context_selection"]["truncated"] is True
    assert any(cell["index"] == 55 for cell in selected["cells"])
    assert [cell["index"] for cell in selected["cells"]] == sorted(cell["index"] for cell in selected["cells"])
    assert "DUPLICATE_PLAN" not in serialized
    assert "SHOULD_NOT_BE_SENT" not in serialized
    assert "CSV schema" in serialized


def test_repeated_workflow_with_old_large_notebook_context_reaches_model_in_both_phases():
    client = FakeClient(valid_analysis(), "print(1)", "print(2)")
    service = LLMCodeGenerator(config(), client)
    context = oversized_notebook()
    analysis = service.analyze_request("Print one", context=context)
    assert service.generate_code_from_analysis("Print one", analysis=analysis, context=context) == "print(1)"
    assert service.repair_code_from_analysis(
        "Print two", code="print(undefined)", traceback="NameError", analysis=analysis, context=context,
    ) == "print(2)"
    assert len(client.completions.requests) == 3
    assert all("DUPLICATE_PLAN" not in request["messages"][1]["content"] for request in client.completions.requests)


def test_small_notebook_keeps_source_and_domain_metadata():
    context = {"cells": [{"index": 0, "type": "code", "source": "x = 1", "metadata": {"units": "Celsius"}}]}
    assert prepare_notebook_context(context) == context


def test_resolves_only_unique_extension_and_reports_real_format(tmp_path):
    data = tmp_path / "observations.csv"
    data.touch()
    result = resolve_reference(str(tmp_path / "observations"))
    assert result["status"] == "inferred_extension"
    assert result["resolved"] == data.as_posix()
    assert result["file_type"].startswith("CSV")
    (tmp_path / "observations.tsv").touch()
    with pytest.raises(ValueError, match="Several files match"):
        resolve_reference(str(tmp_path / "observations"))


@pytest.mark.parametrize("extension,format_name", [(".tsv", "tab-separated"), (".json", "JSON document"), (".xlsx", "Excel workbook"), (".parquet", "binary Parquet")])
def test_exact_file_type_and_no_source_content_read(tmp_path, extension, format_name):
    data = tmp_path / f"observations{extension}"
    data.write_text("not actual data: only path metadata may be read", encoding="utf-8")
    result = resolve_reference(str(data))
    assert result["status"] == "file"
    assert result["file_type"].startswith(format_name)


def test_missing_input_is_not_silently_replaced_and_output_name_not_inferred(tmp_path):
    (tmp_path / "results.csv").touch()
    assert resolve_reference(str(tmp_path / "absent.csv"))["status"] == "missing"
    assert resolve_reference(str(tmp_path / "results"), role="output")["status"] == "missing"
    assert resolve_reference(str(tmp_path))["status"] == "directory"


def test_prompt_paths_support_quotes_spaces_and_output_directories(tmp_path):
    data = tmp_path / "my observations.csv"
    data.touch()
    output = tmp_path / "new-output"
    references = request_file_references(f'Read "{data.as_posix()}". Save results under {output.as_posix()}.')
    assert len(references) == 2
    assert references[0]["resolved"] == data.as_posix()
    assert references[1]["requested"] == output.as_posix()
    assert references[1]["role"] == "output"


def test_create_and_generate_paths_are_outputs(tmp_path):
    generated = tmp_path / "generated.csv"
    description = f"Create a synthetic CSV at {generated.as_posix()} and analyze it."
    references = request_file_references(description)
    assert len(references) == 1
    assert references[0]["requested"] == generated.as_posix()
    assert references[0]["role"] == "output"


def test_broken_windows_path_escape_rejected_before_execution():
    bad = 'with open("C:\\tmp\\data\\observations.csv") as f: pass'
    assert "control character" in file_reference_error(bad, [])
    assert file_reference_error('with open(r"C:\\tmp\\data\\observations.csv") as f: pass', []) is None
    assert file_reference_error('with open("C:/tmp/data/observations.csv") as f: pass', []) is None


def test_generation_retries_missing_extension_using_verified_path(tmp_path):
    data = tmp_path / "observations.csv"
    data.write_text("value\n1\n", encoding="utf-8")
    stem = (tmp_path / "observations").as_posix()
    bad = f"with open({stem!r}) as f: pass"
    good = f"with open({data.as_posix()!r}) as f: pass"
    client = FakeClient(bad, good)
    service = LLMCodeGenerator(config(), client)
    result = service.generate_code_from_analysis(f"Read {stem}", analysis=json.loads(valid_analysis()))
    assert result == good
    assert len(client.completions.requests) == 2
    retry_prompt = client.completions.requests[-1]["messages"][1]["content"]
    assert "Use the verified filename" in retry_prompt
    assert data.as_posix() in retry_prompt
    assert data.read_text(encoding="utf-8") == "value\n1\n"
