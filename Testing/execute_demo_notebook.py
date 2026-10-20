"""Execute a manually reviewed demo notebook in a fresh Jupyter kernel."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import nbformat
from nbclient import NotebookClient

from fixture_paths import LABS_ROOT


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (LABS_ROOT / "plants" / "data" / "plant_growth.csv").resolve()
ALLOWED_IMPORTS = {"csv", "pathlib", "collections", "xml", "datetime", "IPython", "os"}
FORBIDDEN_CALLS = {"system", "popen", "remove", "unlink", "rmdir", "rename",
                   "chmod", "chown", "spawnl", "spawnv", "execv", "startfile"}


def main(cells_path: str) -> None:
    source = Path(cells_path).resolve()
    run_id = source.name.removesuffix("-cells.json")
    run_dir = (ROOT / 'Testing/results' / "agent-live" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    events = json.loads((ROOT / 'Testing/results' / "agent-live" / f"{run_id}.json").read_text(encoding="utf-8"))
    if FIXTURE.as_posix() not in json.dumps(payload) or str(run_dir).replace("\\", "/") not in json.dumps(payload):
        raise SystemExit("The notebook's input or output is outside the reviewed demonstration scope.")
    for number, cell in enumerate(payload["cells"], 1):
        tree = ast.parse(cell["code"], filename=f"cell-{number}")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = ([item.name.split(".", 1)[0] for item in node.names]
                         if isinstance(node, ast.Import) else [(node.module or "").split(".", 1)[0]])
                if any(name not in ALLOWED_IMPORTS for name in names):
                    raise SystemExit(f"Unexpected import in generated cell {number}: {names}")
            if isinstance(node, ast.Name) and node.id in {"exec", "eval", "compile", "__import__"}:
                raise SystemExit(f"Dynamic execution found in generated cell {number}.")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in FORBIDDEN_CALLS):
                raise SystemExit(f"Filesystem/process mutation helper found in generated cell {number}.")

    nb = nbformat.v4.new_notebook()
    citation_ids = events.get("context", {}).get("agent", {}).get("citations", [])
    nb.cells.append(nbformat.v4.new_markdown_cell(
        "# Plant growth: treatment trends\n\n"
        "This notebook was generated from a live investigation using the plant lab. "
        "The agent inspected the CSV schema and retrieved its lab protocol. The cells "
        "below were reviewed and then executed in a fresh Python kernel.\n\n"
        f"**Investigation summary:** {events.get('summary', '')}\n\n"
        f"**Evidence event IDs:** {', '.join(citation_ids)}\n\n"
        "The CSV is a demonstration fixture; its results are not biological findings."
    ))
    for index, cell in enumerate(payload["cells"], 1):
        nb.cells.append(nbformat.v4.new_markdown_cell(f"## Step {index}: {cell['title']}\n\n{cell['purpose']}"))
        nb.cells.append(nbformat.v4.new_code_cell(cell["code"]))
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (ipykernel)",
                                  "language": "python", "name": "python3"}}
    csv_rows = list(csv.DictReader(FIXTURE.open(encoding="utf-8-sig", newline="")))
    expected = {}
    for row in csv_rows:
        expected.setdefault((row["measurement_date"], row["treatment"]), []).append(float(row["height_cm"]))
    aggregate_outputs = [
        value
        for step in payload.get("analysis", {}).get("cell_plan", [])
        for value in step.get("outputs", [])
        if isinstance(value, str) and value.isidentifier()
        and any(word in f"{step.get('title', '')} {step.get('purpose', '')}".lower()
                for word in ("mean", "average", "aggregate"))
    ]
    if not aggregate_outputs:
        raise SystemExit("Could not identify the planned aggregation output variable.")
    aggregate_name = aggregate_outputs[-1]
    verification_code = f"""
_naavre_aggregate = {aggregate_name}
assert isinstance(_naavre_aggregate, dict), 'The planned aggregate is not a dictionary.'
_naavre_checked = 0
for (_naavre_date, _naavre_treatment), _naavre_values in {expected!r}.items():
    _naavre_target = sum(_naavre_values) / len(_naavre_values)
    _naavre_observed = None
    for _naavre_outer_key, _naavre_inner in _naavre_aggregate.items():
        if not isinstance(_naavre_inner, dict):
            continue
        if str(_naavre_outer_key).startswith(_naavre_date):
            _naavre_observed = _naavre_inner.get(_naavre_treatment)
        elif str(_naavre_outer_key) == _naavre_treatment:
            for _naavre_date_key, _naavre_value in _naavre_inner.items():
                if str(_naavre_date_key).startswith(_naavre_date):
                    _naavre_observed = _naavre_value
    assert isinstance(_naavre_observed, (int, float)), (
        f'Missing numeric mean for {{_naavre_date}}/{{_naavre_treatment}}.'
    )
    assert abs(float(_naavre_observed) - _naavre_target) < 1e-10, (
        f'Incorrect mean for {{_naavre_date}}/{{_naavre_treatment}}.'
    )
    _naavre_checked += 1
assert _naavre_checked == {len(expected)}, 'Not all date/treatment means were checked.'
"""
    nb.cells.append(nbformat.v4.new_code_cell(verification_code, metadata={"tags": ["naavre-validation"]}))
    nbformat.validate(nb)
    input_hash = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    generated = run_dir / f"{run_id}-demo.ipynb"
    nbformat.write(nb, generated)

    executed = NotebookClient(
        nb, timeout=120, kernel_name="python3", allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    validation_cell = executed.cells.pop()
    if validation_cell.execution_count is None:
        raise SystemExit("Independent aggregate verification did not execute.")
    inline_svg = any(
        output.output_type == "display_data" and "image/svg+xml" in output.data
        for cell in executed.cells if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )
    if not inline_svg:
        raise SystemExit("The generated SVG was not displayed inline in the Jupyter kernel.")
    executed_path = run_dir / f"{run_id}-demo-executed.ipynb"
    nbformat.write(executed, executed_path)
    if any(cell.get("execution_count") is None for cell in executed.cells if cell.cell_type == "code"):
        raise SystemExit("At least one generated cell did not run in the Jupyter kernel.")
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != input_hash:
        raise SystemExit("The source fixture changed during notebook execution.")

    summary = {"input_rows": len(csv_rows), "date_treatment_groups": len(expected),
               "input_sha256": input_hash, "notebook": str(executed_path)}
    for cell in executed.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                raise SystemExit("Notebook kernel returned an error output.")

    declared_outputs = payload.get("analysis", {}).get("relevant_outputs", [])
    svg_candidates = [Path(item) for item in declared_outputs if str(item).lower().endswith(".svg")]
    if len(svg_candidates) != 1:
        raise SystemExit("Expected exactly one declared SVG output in the analysis.")
    expected_svg = svg_candidates[0].resolve()
    if not expected_svg.is_relative_to((run_dir / "derived").resolve()):
        raise SystemExit("Generated SVG output is outside the run's derived-output directory.")
    svg = ET.parse(expected_svg).getroot()
    lines = svg.findall(".//{http://www.w3.org/2000/svg}polyline")
    treatments = {row["treatment"] for row in csv_rows}
    if len(lines) != len(treatments):
        raise SystemExit("Generated SVG has an unexpected number of treatment lines.")
    summary.update({"status": "passed", "code_cells_executed": len(payload["cells"]),
                    "treatment_lines": len(lines), "svg": str(expected_svg),
                    "svg_bytes": expected_svg.stat().st_size,
                    "means_verified": len(expected), "svg_displayed_inline": inline_svg})
    (run_dir / f"{run_id}-execution-check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python Testing/execute_demo_notebook.py <run-id-cells.json>")
    main(sys.argv[1])
