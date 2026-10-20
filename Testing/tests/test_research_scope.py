"""Scope regression checks, not evidence of model understanding or usability."""
import re
from pathlib import Path

import pytest

from test_labs import lab_registry
from naavre_ai_assistant.labs import LabError
from naavre_ai_assistant.mcp_gateway import MCPGateway


WORKSPACE = Path(__file__).resolve().parents[2]
THESIS = WORKSPACE / "Thesis_Paper"
ORIGINAL_MRQ = (
    "How can Large Language Models enhance scientific workflow composition, "
    "automation, and optimization in a Virtual Research Environment?"
)
SUBQUESTIONS = (
    "How can LLMs use natural-language requests, existing notebook code, and verified "
    "dataset schemas to generate focused, executable notebook cells?",
    "How can the correctness and reproducibility of LLM-generated notebook workflows "
    "and their outputs be assessed?",
    "How can MCP interaction patterns provide verified lab context to support LLM "
    "code understanding and notebook cell generation?",
    "What technical requirements and adaptations are needed to integrate the "
    "assistant into NaaVRE's existing JupyterLab environment?",
)


def test_original_main_question_is_unchanged():
    if not THESIS.is_dir():
        pytest.skip('The optional thesis manuscript is not included in the code repository.')
    introduction = (THESIS / "chapters" / "introduction.tex").read_text(encoding="utf-8")
    assert re.findall(r"^MRQ: (.+)$", introduction, flags=re.MULTILINE) == [ORIGINAL_MRQ]


@pytest.mark.parametrize("relative_path", [
    "chapters/introduction.tex",
    "chapters/methods.tex",
    "Research/implementation_methodology.md",
])
def test_current_question_sets_agree_and_exclude_removed_studies(relative_path):
    if not THESIS.is_dir():
        pytest.skip('The optional thesis manuscript is not included in the code repository.')
    text = (THESIS / relative_path).read_text(encoding="utf-8")
    for number, question in enumerate(SUBQUESTIONS, start=1):
        assert text.count(question) == 1
        assert re.search(rf"RQ{number}[^\n]*{re.escape(question)}", text)
    question_lines = [line for line in text.splitlines() if re.search(r"RQ\d[^\n]*\?", line)]
    assert len(question_lines) == 4
    assert all(not re.search(r"usability|build automation|build failures|CI/CD", line, re.I)
               for line in question_lines)


@pytest.mark.asyncio
async def test_real_mcp_catalogue_stays_within_the_notebook_lab_scope(lab_registry):
    async with MCPGateway(lab_registry.get("a")) as gateway:
        tools = await gateway.list_tools()
        assert {tool["name"] for tool in tools} == {
            "list_files", "inspect_dataset", "search_knowledge", "save_note",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [
    "repair_container_build", "analyze_ci_logs", "run_container_build",
    "record_usability_response", "submit_survey",
])
async def test_out_of_scope_actions_are_rejected_before_dispatch(lab_registry, name):
    gateway = MCPGateway(lab_registry.get("a"))
    with pytest.raises(LabError, match="Tool is not allowed"):
        await gateway.call_tool(name, {})
