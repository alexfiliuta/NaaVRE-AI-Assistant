"""Read a bounded CSV/TSV sample and retain exact input column names."""

from __future__ import annotations

import ast
import codecs
import csv
import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


SAMPLE_BYTES = 65_536
SAMPLE_ROWS = 20


def inspect_table(path_string: str) -> dict[str, Any] | None:
    path = Path(path_string)
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return None
    try:
        stat = path.stat()
        with path.open("rb") as stream:
            raw = stream.read(SAMPLE_BYTES)
        encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        limited = stat.st_size > len(raw)
        text = codecs.getincrementaldecoder(encoding)().decode(raw, final=not limited)
        if limited:
            text = text[:text.rfind("\n") + 1]
        if not text.strip():
            raise ValueError(f"Input table '{path}' is empty or has no readable header.")
        # Sniff the header, not dirty numeric observations: malformed data rows
        # must not make a comma-separated file look like a different format.
        header_text = text.splitlines()[0]
        try:
            dialect = csv.Sniffer().sniff(header_text, delimiters=",;\t|")
            delimiter, quotechar = dialect.delimiter, dialect.quotechar
        except csv.Error:
            delimiter, quotechar = ("\t" if path.suffix.lower() == ".tsv" else ","), '"'
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, quotechar=quotechar)
        headers = next(reader)
        if not headers or any(not field.strip() for field in headers):
            raise ValueError(f"Input table '{path}' has blank column names; correct the header first.")
        if len(set(headers)) != len(headers):
            raise ValueError(f"Input table '{path}' has duplicate column names; correct the header first.")
        if len(headers) > 200 or sum(map(len, headers)) > 8_000:
            raise ValueError(f"Input table '{path}' has too many/too-long column names for one request.")
        rows = []
        for row in reader:
            if len(rows) == SAMPLE_ROWS:
                limited = True
                break
            rows.append(row)
        warnings = []
        if any(len(row) != len(headers) for row in rows):
            warnings.append("Some sampled rows have a different field count from the header.")
        columns = []
        for index, name in enumerate(headers):
            values = [row[index] for row in rows if index < len(row)]
            present = [value.strip() for value in values if value.strip()]
            kinds = sorted({_value_type(value) for value in present})
            columns.append({"name": name, "sample_types": kinds or ["unknown"], "sample_missing": len(values) - len(present)})
        identity = [headers, delimiter, quotechar, encoding, stat.st_size, stat.st_mtime_ns]
        return {
            "path": path.as_posix(), "format": path.suffix.lower().lstrip("."),
            "encoding": encoding, "delimiter": delimiter, "quotechar": quotechar,
            "headers": headers, "columns": columns, "sample_rows": len(rows),
            "sample_only": limited, "warnings": warnings,
            "fingerprint": hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
        }
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"Cannot inspect input table '{path}': {error}. Check its encoding and format.") from error


def _value_type(value: str) -> str:
    try:
        number = float(value)
        return "number" if math.isfinite(number) else "non-finite number"
    except ValueError:
        pass
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return "ISO timestamp"
    except ValueError:
        return "text"


def schemas_from_references(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [reference["schema"] for reference in references if isinstance(reference.get("schema"), dict)]


def verify_snapshot(analysis: dict[str, Any], current: list[dict[str, Any]]) -> None:
    previous = analysis.get("input_schemas")
    if previous is None:  # A browser running an older build can still submit its plan.
        return
    if not isinstance(previous, list) or any(not isinstance(item, dict) for item in previous):
        raise ValueError("Invalid input schema snapshot. Analyze the request again.")
    old = {(item.get("path"), item.get("fingerprint")) for item in previous}
    new = {(item["path"], item["fingerprint"]) for item in current}
    if old != new:
        raise ValueError("The input file changed since planning. Analyze the request again before generating code.")


def schema_column_error(code: str, schemas: list[dict[str, Any]], previous_code: str = "") -> str | None:
    """Check common literal column accesses/validation lists without executing code.

    Explicitly constructed derived dictionary keys are allowed. This is a guard
    for column hallucinations, not a complete Python dataflow/type checker.
    """
    if not schemas:
        return None
    headers = {name for schema in schemas for name in schema["headers"]}
    try:
        tree = ast.parse(code)
        earlier = ast.parse(previous_code)
    except SyntaxError:
        return None
    derived = set()
    row_like = {"row", "record", "observation", "data", "df", "r"}
    calculated_names = set()
    # Aggregates are often assigned first, then placed into output records by
    # name. Track these explicit scalar calculations across earlier cells too;
    # constants/configuration strings and converted source lookups do not qualify.
    for root in (earlier, tree):
        for node in ast.walk(root):
            if isinstance(node, ast.Assign):
                value = node.value
                calculated = (
                    isinstance(value, (ast.BinOp, ast.Call, ast.IfExp, ast.UnaryOp))
                    and not _expression_uses_row_like_name(value, row_like)
                ) or (isinstance(value, ast.Name) and value.id in calculated_names)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        calculated_names.discard(target.id)
                        if calculated:
                            calculated_names.add(target.id)
    for root in (earlier, tree):
        for node in ast.walk(root):
            if isinstance(node, ast.For):
                target_names = [target.id for target in ast.walk(node.target) if isinstance(target, ast.Name)]
                iterable = ast.unparse(node.iter).lower()
                if any(word in iterable for word in ("data", "record", "observation", "row")):
                    row_like.update(target_names)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                if node.value.id in row_like:
                    row_like.update(
                        target.id for target in node.targets if isinstance(target, ast.Name)
                    )
            if isinstance(node, ast.Dict):
                # A key is a derived field only when this literal visibly maps
                # from an input row/record. Arbitrary configuration dictionaries
                # must not whitelist misspelled source columns.
                if any(
                    isinstance(value, (ast.Subscript, ast.Call))
                    and _expression_uses_row_like_name(value, row_like)
                    for value in node.values
                ):
                    derived.update(
                        key.value for key in node.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    )
                # Also recognize explicitly calculated fields in a record that
                # retains at least one verified source field, e.g. a grouped
                # record containing measurement_date/treatment/mean_height_cm.
                # A direct row lookup is never a derived value, even when wrapped
                # in float() or another conversion call.
                source_record = any(
                    isinstance(key, ast.Constant) and key.value in headers
                    for key in node.keys
                )
                if source_record:
                    for key, value in zip(node.keys, node.values):
                        if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                                and key.value not in headers
                                and (isinstance(value, (ast.BinOp, ast.Call, ast.IfExp,
                                                        ast.JoinedStr, ast.UnaryOp))
                                     or isinstance(value, ast.Name) and value.id in calculated_names)
                                and not _expression_uses_row_like_name(value, row_like)):
                            derived.add(key.value)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                        derived.add(target.slice.value)
    allowed = headers | derived
    unknown = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                # Check row/table lookups; avoid unrelated config and environment dictionaries.
                if _expression_uses_row_like_name(node.value, row_like):
                    if node.slice.value not in allowed:
                        unknown.add(node.slice.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
            key = node.args[0]
            if _expression_uses_row_like_name(node.func.value, row_like) and isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value not in allowed:
                unknown.add(key.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"pop", "setdefault"} and node.args:
            key = node.args[0]
            if _expression_uses_row_like_name(node.func.value, row_like) and isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value not in allowed:
                unknown.add(key.value)
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
            if any(isinstance(target, ast.Name) and target.id in {"required_columns", "expected_columns", "required_fields"} for target in node.targets):
                # Required source columns are never satisfied by a derived key.
                unknown.update(item.value for item in node.value.elts if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value not in headers)
    if unknown:
        return (f"Unverified input columns: {', '.join(sorted(unknown))}. "
                f"Actual CSV/TSV headers: {', '.join(sorted(headers))}. "
                "Use exact source names. Only use other names after explicitly creating derived fields.")
    return None


def _expression_uses_row_like_name(expression: ast.AST, row_like: set[str]) -> bool:
    """Match variable names, not substrings such as ``metadata`` or ``station_data``."""
    return any(
        isinstance(node, ast.Name) and node.id.lower() in {name.lower() for name in row_like}
        for node in ast.walk(expression)
    )
