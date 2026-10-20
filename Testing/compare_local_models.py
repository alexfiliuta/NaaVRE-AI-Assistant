"""Compare installed local models on fixed lab investigations; never execute code.

Generated notebooks are optional review artifacts, not successful execution claims.
Run from the project directory with its virtual environment.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen
import uuid

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "naavre-ai-assistant"))
from naavre_ai_assistant.agent_runtime import AgentLimits, AgentRuntime
from naavre_ai_assistant.labs import LabRegistry
from naavre_ai_assistant.llm_service import LLMCodeGenerator, LLMConfig, _native_ollama_chat_endpoint


TASKS = [
    ("treatment-trends", "Inspect plant_growth.csv and the lab protocol. Plan a descriptive comparison "
     "of mean height_cm by measurement_date and treatment with an SVG line chart. Keep the source CSV "
     "unchanged. Do not write files during the investigation."),
    ("individual-gain", "Use plant_growth.csv and the lab protocol to plan a comparison of how much "
     "each plant grew between its first and last measurement, then compare treatment averages. "
     "Keep individual plants as the experimental units. Do not write files during the investigation."),
    ("quality-first", "Inspect plant_growth.csv and the protocol. Plan checks for missing measurements, "
     "duplicate plant-date records and unusual heights before comparing treatments. Use actual column "
     "names even if I call them date, plant and height. Do not write files during the investigation."),
]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--models", nargs="+", default=["qwen3-coder:30b", "qwen3.8:27b"])
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--context", type=int, default=16384)
    result.add_argument("--think", choices=["auto", "true", "false"], default="auto")
    result.add_argument("--generate", action="store_true", help="Also generate unexecuted review notebooks")
    result.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/model-comparison")
    return result


async def main():
    args = parser().parse_args()
    if args.repeats < 1 or args.context < 1:
        raise SystemExit("Repeats and context must be positive.")
    settings = LLMConfig.from_environment()
    if settings.configuration_error:
        raise SystemExit(settings.configuration_error)
    endpoint = _native_ollama_chat_endpoint(settings.base_url).removesuffix("/chat") + "/tags"
    with urlopen(endpoint) as response:
        installed = json.load(response).get("models", [])
    with urlopen(endpoint.removesuffix("/tags") + "/version") as response:
        ollama_version = json.load(response).get("version", "unknown")
    available = {item.get("model", item.get("name")): item for item in installed}
    missing = [model for model in args.models if model not in available]
    if missing:
        raise SystemExit("Install these models on nis07 first: " + ", ".join(missing))
    registry = LabRegistry()
    lab = registry.get("plants")
    source = lab.resolve("data", "plant_growth.csv")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    folder = args.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "context": args.context, "thinking": args.think, "repeats": args.repeats,
              "dataset_sha256": before, "models": {model: available[model] for model in args.models},
              "ollama_version": ollama_version, "python_version": sys.version,
              "sampling_temperature": {"agent": 0.1, "analysis": 0.0, "first_code_attempt": 0.2},
              "source_hashes": {name: hashlib.sha256(
                  (Path(__file__).resolve().parents[1] / "naavre-ai-assistant" / "naavre_ai_assistant" / name).read_bytes()).hexdigest()
                  for name in ("agent_runtime.py", "llm_service.py")},
              "execution_performed": False, "results": []}
    # Alternate model order to reduce systematic warm-up/order bias.
    for repeat in range(args.repeats):
        for task_id, prompt in TASKS:
            order = args.models if repeat % 2 == 0 else list(reversed(args.models))
            for model in order:
                service = LLMCodeGenerator(replace(settings, model=model, context_window=args.context,
                    agent_model=None, analysis_model=None, code_model=None,
                    thinking={"auto": None, "true": True, "false": False}[args.think]))
                if not service.configured:
                    raise SystemExit(service.config.configuration_error)
                manager = AgentRuntime(registry, folder / "state", service=service,
                    allowed_tools=["list_files", "inspect_dataset", "search_knowledge", "list_skills"],
                    limits=AgentLimits(max_steps=12))
                started = time.monotonic()
                run_id = manager.start("model-comparison", "plants", prompt)["run_id"]
                await manager._runs[run_id].task
                snapshot = manager.get("model-comparison", run_id)
                row = {"task": task_id, "repeat": repeat + 1, "model": model,
                       "run_id": run_id, "status": snapshot["status"],
                       "investigation_seconds": round(time.monotonic() - started, 3),
                       "tool_calls": sum(e["kind"] == "observation" for e in snapshot["events"]),
                       "repeat_calls_blocked": sum(e["kind"] == "reuse" for e in snapshot["events"]),
                       "generation": "not requested", "notebook_executed": False,
                       "human_scientific_review": "pending"}
                (folder / f"{run_id}.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
                if args.generate and snapshot["status"] == "completed":
                    try:
                        generation_prompt = (prompt + " Build focused Python notebook cells from this plan. "
                            f"The full input path is {source.as_posix()}. Use standard-library CSV and SVG "
                            "operations. Display results inline; do not modify the source or save derived files.")
                        context = manager.verified_context("model-comparison", run_id)
                        generation_start = time.monotonic()
                        analysis = await asyncio.to_thread(service.analyze_request, generation_prompt, context=context)
                        cells = await asyncio.to_thread(service.generate_cells_from_analysis,
                            generation_prompt, analysis=analysis, context=context)
                        notebook = nbformat.v4.new_notebook(cells=[
                            nbformat.v4.new_markdown_cell(f"# {task_id}\n\nModel: {model}. Review and run cells manually.")])
                        for cell in cells:
                            notebook.cells += [nbformat.v4.new_markdown_cell(cell["title"] + "\n\n" + cell["purpose"]),
                                               nbformat.v4.new_code_cell(cell["code"])]
                        notebook.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
                        nbformat.write(notebook, folder / f"{run_id}.ipynb")
                        row.update(generation="validated source", cells=len(cells),
                                   generation_seconds=round(time.monotonic() - generation_start, 3))
                    except Exception as error:
                        row.update(generation="failed", generation_error=str(error)[:1000])
                report["results"].append(row)
                (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(row), flush=True)
    report["dataset_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == before
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str((folder / "report.json").resolve()),
                      "dataset_unchanged": report["dataset_unchanged"], "execution_performed": False}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
