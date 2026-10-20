"""Resolve explicit file names and optionally inspect bounded, read-only table samples."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .input_schema import inspect_table


FILE_TYPES = {
    ".csv": "CSV table; use a CSV reader with a header row",
    ".tsv": "tab-separated table; use a CSV reader with delimiter='\\t'",
    ".json": "JSON document; use json.load",
    ".jsonl": "JSON Lines; parse each nonempty line as JSON",
    ".ndjson": "newline-delimited JSON; parse each nonempty line as JSON",
    ".txt": "text file; determine the delimiter from supplied metadata",
    ".parquet": "binary Parquet; requires an available Parquet reader",
    ".xlsx": "Excel workbook; requires an available workbook reader",
    ".xls": "legacy Excel workbook; requires an available workbook reader",
}
_PATH = re.compile(
    r'''["'](?P<quoted>(?:[A-Za-z]:[\\/]|/)[^"'\r\n]+)["']'''
    r'''|(?<![\w:/])(?P<plain>(?:[A-Za-z]:[\\/]|/)[^\s"'<>]+)'''
)
_OUTPUT_PREFIX = re.compile(
    r"\b(?:save|write|export|output|outputs|create|generate)\b[^.\n]{0,100}\b(?:to|under|in|at|folder|directory)\s*:?\s*$",
    re.IGNORECASE,
)


def resolve_reference(reference: str, *, role: str = "input") -> dict[str, Any]:
    """Only infer an omitted extension when exactly one supported sibling exists."""
    reference = reference.replace("\\", "/")
    path = Path(reference)
    result: dict[str, Any] = {"requested": reference, "role": role}
    if re.match(r"^[A-Za-z]:/", reference) and os.name != "nt":
        return {**result, "status": "different_operating_system", "note": "Windows path cannot be verified on this server."}
    if not path.is_absolute():
        return {**result, "status": "relative", "note": "Resolve relative to the notebook kernel working directory."}
    try:
        if path.is_file():
            return {**result, "status": "file", "resolved": path.as_posix(), "file_type": FILE_TYPES.get(path.suffix.lower(), "unknown; do not assume CSV")}
        if path.is_dir():
            return {**result, "status": "directory", "resolved": path.as_posix(), "note": "This is a directory, not a file. Require an explicit filename if several input files are possible."}
        if role != "output" and not path.suffix:
            candidates = [path.with_suffix(suffix) for suffix in FILE_TYPES if path.with_suffix(suffix).is_file()]
            if len(candidates) == 1:
                resolved = candidates[0]
                return {**result, "status": "inferred_extension", "resolved": resolved.as_posix(), "file_type": FILE_TYPES[resolved.suffix.lower()]}
            if len(candidates) > 1:
                raise ValueError(
                    f"Several files match '{reference}': "
                    + ", ".join(candidate.as_posix() for candidate in candidates)
                    + ". Specify the full filename including its extension."
                )
    except OSError as error:
        return {**result, "status": "unverified", "note": f"Could not check the path: {type(error).__name__}"}
    return {**result, "status": "missing", "file_type": FILE_TYPES.get(path.suffix.lower(), "unspecified"), "note": "May be created only if it is a requested output. Never fabricate a missing input dataset."}


def request_file_references(
    description: str,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    *, inspect_inputs: bool = False,
) -> list[dict[str, Any]]:
    references: dict[str, str] = {}
    for match in _PATH.finditer(description):
        path = (match.group("quoted") or match.group("plain")).rstrip(".,;:)")
        prefix = description[max(0, match.start() - 120):match.start()]
        references[path] = "output" if _OUTPUT_PREFIX.search(prefix) else "input"
    for role, declared in (("input", inputs), ("output", outputs)):
        for key in declared or {}:
            if isinstance(key, str) and re.match(r"^(?:[A-Za-z]:[\\/]|/)", key):
                references[key] = role
    if len(references) > 100:
        raise ValueError("Too many file paths in one request; select the files needed for this workflow.")
    resolved = [resolve_reference(path, role=role) for path, role in references.items()]
    if inspect_inputs:
        for reference in resolved:
            if reference["role"] == "input" and reference["status"] in {"file", "inferred_extension"}:
                schema = inspect_table(reference["resolved"])
                if schema is not None:
                    reference["schema"] = schema
    return resolved


def file_reference_error(code: str, references: list[dict[str, Any]]) -> str | None:
    """Catch known wrong path literals before notebook insertion, without running code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # The normal syntax validator reports this separately.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        if re.match(r"^[A-Za-z]:", value) and any(ord(char) < 32 for char in value):
            return "A Windows path contains an escaped control character. Use forward slashes such as data/observations.csv or a raw Python string."
        normalized = value.replace("\\", "/")
        for reference in references:
            if reference["status"] == "inferred_extension" and normalized == reference["requested"]:
                return f"Use the verified filename '{reference['resolved']}' including its extension instead of '{normalized}'."
    return None
