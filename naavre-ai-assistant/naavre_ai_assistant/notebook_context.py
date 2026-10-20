"""Select notebook context without sending duplicated plans or oversized snapshots."""

from __future__ import annotations

import json
from typing import Any, Mapping


NOTEBOOK_CONTEXT_BUDGET = 40_000
MAX_CONTEXT_CELLS = 40


def _serialized(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _preview(value: Any, depth: int = 0) -> Any:
    """Bound nested metadata while retaining JSON structure and marking omissions."""
    if isinstance(value, str):
        return value if len(value) <= 2_000 else value[:2_000] + " [truncated]"
    if isinstance(value, (Mapping, list)) and depth >= 6:
        return "[nested context omitted]"
    if isinstance(value, Mapping):
        result = {str(k): _preview(v, depth + 1) for k, v in list(value.items())[:40]}
        if len(value) > 40:
            result["_omitted_fields"] = len(value) - 40
        return result
    if isinstance(value, list):
        result = [_preview(v, depth + 1) for v in value[:40]]
        if len(value) > 40:
            result.append("[remaining items omitted]")
        return result
    return value


def metadata_without_analysis(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    metadata = dict(value)
    assistant = metadata.get("naavre-ai-assistant")
    if isinstance(assistant, Mapping):
        metadata["naavre-ai-assistant"] = {
            key: val for key, val in assistant.items() if key != "analysis"
        }
    return metadata


def prepare_notebook_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Fit notebook snapshots under a total *serialized* budget, including old clients.

    Keep the active cell and nearby predecessors first, then return them in notebook
    order. Never modify the notebook or claim omitted source is complete. Non-notebook
    application context is left to the service's usual request validation.
    """
    raw_cells = context.get("cells")
    if not isinstance(raw_cells, list):
        return dict(context)

    result = dict(context)
    cells = []
    for position, raw_cell in enumerate(raw_cells):
        if not isinstance(raw_cell, Mapping):
            continue
        cell = {
            key: raw_cell[key]
            for key in ("index", "type", "source", "source_truncated") if key in raw_cell
        }
        if not isinstance(cell.get("index"), int):
            cell["index"] = position
        cell["metadata"] = metadata_without_analysis(raw_cell.get("metadata"))
        cells.append(cell)
    result["cells"] = cells
    if len(cells) <= MAX_CONTEXT_CELLS and len(_serialized(result)) <= NOTEBOOK_CONTEXT_BUDGET:
        return result

    notebook = context.get("notebook")
    notebook = notebook if isinstance(notebook, Mapping) else {}
    active = notebook.get("active_cell_index", cells[-1]["index"] if cells else 0)
    if not isinstance(active, int):
        active = 0
    # Strip oversized ancillary values before choosing full source snippets.
    result = _preview({key: val for key, val in result.items() if key != "cells"})
    for key in list(result):
        if len(_serialized(result[key])) > 4_000:
            result[key] = {"preview": _serialized(result[key])[:2_000], "truncated": True}
    result["cells"] = []
    result["context_selection"] = {
        "truncated": True,
        "note": "Only selected notebook context is included. Omitted cells may define dependencies; do not assume they are absent.",
    }
    selected: list[dict[str, Any]] = []
    priorities = sorted(cells, key=lambda cell: (abs(cell["index"] - active), cell["index"] > active))
    for original in priorities[:MAX_CONTEXT_CELLS]:
        cell = dict(original)
        metadata = _preview(cell["metadata"])
        if len(_serialized(metadata)) > 4_000:
            metadata = {"preview": _serialized(metadata)[:2_000], "truncated": True}
        cell["metadata"] = metadata
        source = cell.get("source", "")
        source = source if isinstance(source, str) else str(source)
        if len(source) > 6_000:
            cell["source_truncated"] = True
        cell["source"] = source[:6_000]
        while True:
            result["cells"] = sorted([*selected, cell], key=lambda item: item["index"])
            if len(_serialized(result)) <= NOTEBOOK_CONTEXT_BUDGET:
                selected.append(cell)
                break
            if len(cell["source"]) <= 200:
                break
            cell["source"] = cell["source"][:len(cell["source"]) // 2]
            cell["source_truncated"] = True
    result["cells"] = sorted(selected, key=lambda cell: cell["index"])
    return result
