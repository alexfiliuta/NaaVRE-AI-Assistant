"""Reproduce planning/generation using a saved demo investigation; never execute cells."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import uuid

import nbformat

from fixture_paths import LABS_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "naavre-ai-assistant"))
from naavre_ai_assistant.llm_service import LLMCodeGenerator, LLMConfig
from naavre_ai_assistant.agent_runtime import inspected_file_declarations
from naavre_ai_assistant.labs import LabRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--think", choices=["auto", "true", "false"], default="false")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--analysis", type=Path, help="Reuse a previously validated plan for a generation-only check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = Path(__file__).resolve().parents[1]
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if state["lab_id"] != "plants" or state["status"] != "completed":
        raise SystemExit("Expected a completed plant demonstration investigation.")
    fixture = LABS_ROOT / "plants" / "data" / "plant_growth.csv"
    before = hashlib.sha256(fixture.read_bytes()).hexdigest()
    folder = root / 'Testing/results' / "planning-regression" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    service = LLMCodeGenerator(replace(LLMConfig.from_environment(), model=args.model,
        agent_model=None, analysis_model=None, code_model=None, context_window=16384,
        thinking={"auto": None, "true": True, "false": False}[args.think]))
    prompt = ("Compare plant height over time for the different treatments. Use the lab data and "
              "protocol, explain the approach, and prepare separate notebook cells for the analysis and a chart.")
    context = state["context"]
    context.update(inspected_file_declarations(LabRegistry(LABS_ROOT).get("plants"), context))
    started = time.monotonic()
    report = {"model": args.model, "thinking": args.think, "source_run": state["run_id"],
              "prompt": prompt, "context_window": 16384, "notebook_executed": False,
              "source_hash_before": before}
    try:
        analysis = (json.loads(args.analysis.read_text(encoding="utf-8")) if args.analysis else
            service.analyze_request(prompt, context=context,
                metadata={"lab_id": "plants", "visualization_requested": True}))
        (folder / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        report.update(status="planning passed", cells=len(analysis["cell_plan"]),
                      planning_reused=bool(args.analysis),
                      planning_seconds=round(time.monotonic() - started, 3))
        print(json.dumps(report), flush=True)
        if args.generate:
            cells = service.generate_cells_from_analysis(prompt, context=context, analysis=analysis,
                metadata={"lab_id": "plants", "visualization_requested": True})
            (folder / "cells.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")
            notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
                "# Plant growth build regression\n\nUnexecuted generated cells: review before running.")])
            for cell in cells:
                notebook.cells.extend([nbformat.v4.new_markdown_cell(
                    cell["title"] + "\n\n" + cell["purpose"]), nbformat.v4.new_code_cell(cell["code"])])
            notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
            nbformat.write(notebook, folder / "plant-growth-review.ipynb")
            report.update(status="source generation passed", generated_cells=len(cells))
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=round(time.monotonic() - started, 3),
                      source_unchanged=hashlib.sha256(fixture.read_bytes()).hexdigest() == before)
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(folder / "report.json"), **report}), flush=True)


if __name__ == "__main__":
    main()
