"""Exercise the real local model and MCP gateway; never execute generated Python.

Run from the project root with its virtualenv. Results are test evidence, not a
scientific evaluation. The agent can only read the selected lab in this check.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "naavre-ai-assistant"))
from naavre_ai_assistant.agent_runtime import AgentRuntime, AgentLimits, TERMINAL
from naavre_ai_assistant.labs import LabRegistry
from naavre_ai_assistant.llm_service import LLMCodeGenerator


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", default="plants")
    parser.add_argument("--prompt", default="Inspect plant_growth.csv and the lab protocol. Explain the correct way to compare height growth between treatments. Do not write any files.")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / "results/agent-live"
    output.mkdir(parents=True, exist_ok=True)
    runtime = AgentRuntime(LabRegistry(), output / "state", allowed_tools=[
        "list_files", "inspect_dataset", "search_knowledge", "list_skills", "gbif_species_lookup"],
        limits=AgentLimits(run_seconds=900, operation_seconds=180))
    started = time.monotonic()
    run_id = runtime.start("local-verification", args.lab, args.prompt)["run_id"]
    while True:
        await asyncio.sleep(1)
        snapshot = runtime.get("local-verification", run_id)
        if snapshot["status"] in TERMINAL:
            break
    snapshot["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (output / f"{run_id}.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": snapshot["status"], "summary": snapshot["summary"],
                      "tools": [e["tool"] for e in snapshot["events"] if e["kind"] == "observation"],
                      "elapsed_seconds": snapshot["elapsed_seconds"]}), flush=True)
    if snapshot["status"] != "completed":
        raise SystemExit(1)
    if args.generate:
        service = LLMCodeGenerator()
        lab = runtime.lab(args.lab)
        prompt = (f"Use {lab.resolve('data', 'plant_growth.csv').as_posix()}. Its verified columns are measurement_date, "
                  "plant_id, treatment, height_cm, leaf_count, and biomass_g. Use these exact source columns. "
                  "Make an SVG line chart of mean height_cm by measurement_date, with one line per treatment, "
                  "and display it inline in the notebook after saving. "
                  f"Save only the chart under {(output / run_id / 'derived').resolve().as_posix()}. "
                  "Keep the input CSV unchanged. Any mean_height_cm value must be calculated from height_cm, "
                  "because mean_height_cm is not an input column.")
        context = runtime.verified_context("local-verification", run_id)
        analysis = await asyncio.to_thread(service.analyze_request, prompt, context=context)
        (output / f"{run_id}-analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        try:
            cells = await asyncio.to_thread(service.generate_cells_from_analysis, prompt, analysis=analysis, context=context)
        except Exception as exc:
            (output / f"{run_id}-generation-error.json").write_text(json.dumps({"prompt": prompt, "analysis": analysis, "error": str(exc)}, indent=2), encoding="utf-8")
            raise
        (output / f"{run_id}-cells.json").write_text(json.dumps({"prompt": prompt, "analysis": analysis, "cells": cells}, indent=2), encoding="utf-8")
        notebook = nbformat.v4.new_notebook()
        notebook.cells = [nbformat.v4.new_markdown_cell(
            "# Plant growth: treatment trends\n\n"
            f"This notebook contains a live Qwen-generated plan with {len(cells)} focused code cells. "
            "The agent inspected the fixture and retrieved the plant-lab protocol. "
            "Review each cell and run them one at a time; no cell output is pre-filled.\n\n"
            f"**Agent summary:** {snapshot['summary']}\n\n"
            f"**Evidence events:** {', '.join(snapshot['context']['agent']['citations'])}"
        )]
        for index, cell in enumerate(cells, start=1):
            notebook.cells.append(nbformat.v4.new_markdown_cell(
                f"### Step {index}: {cell['title']}\n\n{cell['purpose']}"))
            notebook.cells.append(nbformat.v4.new_code_cell(cell["code"]))
        notebook.metadata = {"kernelspec": {"display_name": "Python 3 (ipykernel)",
                                             "language": "python", "name": "python3"}}
        nbformat.validate(notebook)
        notebook_path = output / f"{run_id}-demo.ipynb"
        nbformat.write(notebook, notebook_path)
        print(json.dumps({"notebook": str(notebook_path.resolve()), "executed": False}), flush=True)
        print(json.dumps({"generated_cells": len(cells), "executed": False}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
