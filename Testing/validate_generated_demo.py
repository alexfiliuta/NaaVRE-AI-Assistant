"""Execute a reviewed live-generated demo against its fixed, non-sensitive fixture."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from fixture_paths import LABS_ROOT


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (LABS_ROOT / "plants" / "data" / "plant_growth.csv").resolve()
ALLOWED_IMPORTS = {"csv", "pathlib", "collections", "xml", "datetime"}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "__import__", "openpyxl", "requests", "urlopen"}


def main(path: str) -> None:
    artifact = Path(path).resolve()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    cells = payload["cells"]
    if len(cells) < 2:
        raise SystemExit("Refusing to execute: expected a multi-cell demonstration.")
    if FIXTURE.as_posix() not in json.dumps(payload):
        raise SystemExit("Refusing to execute: the reviewed fixture is not the sole declared input.")
    output_base = (ROOT / 'Testing/results' / "agent-live" / artifact.name.removesuffix("-cells.json")).resolve()
    if str(output_base).replace("\\", "/") not in json.dumps(payload):
        raise SystemExit("Refusing to execute: generated output path is not confined to the demo run folder.")

    source_hash = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    namespace: dict[str, object] = {"__name__": "__naavre_generated_demo__"}
    for index, cell in enumerate(cells, start=1):
        code = cell["code"]
        tree = ast.parse(code, filename=f"generated-cell-{index}")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = ([item.name.split(".", 1)[0] for item in node.names]
                           if isinstance(node, ast.Import) else [(node.module or "").split(".", 1)[0]])
                if any(module not in ALLOWED_IMPORTS for module in modules):
                    raise SystemExit(f"Refusing to execute cell {index}: import outside the reviewed standard-library set.")
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                raise SystemExit(f"Refusing to execute cell {index}: dynamic execution or network helper found.")
        exec(compile(tree, f"generated-cell-{index}", "exec"), namespace, namespace)

    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != source_hash:
        raise SystemExit("Source fixture changed during execution.")
    rows = list(csv.DictReader(FIXTURE.open(encoding="utf-8-sig", newline="")))
    if namespace.get("raw_rows") is None or len(namespace["raw_rows"]) != len(rows):
        raise SystemExit("Generated loading cell did not read all fixture rows.")

    expected: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        expected.setdefault((row["measurement_date"], row["treatment"]), []).append(float(row["height_cm"]))
    actual = namespace.get("mean_heights_by_date_treatment")
    if not isinstance(actual, dict):
        raise SystemExit("Generated aggregation cell did not produce its named output.")
    for (date, treatment), values in expected.items():
        observed = actual[date][treatment]
        reference = sum(values) / len(values)
        if abs(observed - reference) > 1e-10:
            raise SystemExit(f"Generated mean is incorrect for {date}/{treatment}: {observed} != {reference}")

    output_file = Path(namespace["output_file"]).resolve()
    if not output_file.is_relative_to(output_base) or not output_file.is_file() or output_file.stat().st_size < 500:
        raise SystemExit("Generated visualization is missing, too small, or outside this run's output folder.")
    svg = ET.parse(output_file).getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    polylines = svg.findall(".//svg:polyline", ns)
    if len(polylines) != len({row["treatment"] for row in rows}):
        raise SystemExit("SVG line count does not match treatment groups in the fixture.")
    print(json.dumps({"status": "passed", "cells_executed": len(cells), "input_rows": len(rows),
                      "treatment_date_means_checked": len(expected), "svg": str(output_file),
                      "svg_bytes": output_file.stat().st_size, "source_sha256_unchanged": source_hash}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python Testing/validate_generated_demo.py <run-id-cells.json>")
    main(sys.argv[1])
