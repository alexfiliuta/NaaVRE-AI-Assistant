"""A real MCP 2.2 server exposing a single administrator-selected lab."""
from __future__ import annotations

import csv
import codecs
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from .labs import Lab, LabError, LabRegistry, _exclusive_write, _table_snapshot


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LabError("External service redirect refused")


def create_server(lab: Lab) -> MCPServer:
    server = MCPServer("NaaVRE lab " + lab.id, version="0.2.0", log_level="ERROR")
    read_annotation = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    def list_files(area: str = "data") -> dict[str, Any]:
        """List files inside this lab. area is data, knowledge, skills or outputs."""
        return {"lab_id": lab.id, "files": lab.files(area)}

    def inspect_dataset(path: str) -> dict[str, Any]:
        """Inspect a CSV/TSV in this lab's data directory. path is relative, e.g. observations.csv. Returns exact headers and a bounded preview, not full quality analysis."""
        if path.startswith("data/"):
            path = path[5:]
        file = lab.resolve("data", path)
        if file.suffix.lower() not in {".csv", ".tsv"}:
            raise LabError("Only CSV and TSV schema inspection is available")
        # A full streaming hash identifies the source, but only 64 KiB and five
        # preview rows can enter model context. Do not reject a large CSV just
        # because it is larger than the former 8 MiB in-memory read limit.
        raw, digest, size = _table_snapshot(file)
        encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        limited = size > len(raw)
        try:
            text = codecs.getincrementaldecoder(encoding)().decode(raw, final=not limited)
        except UnicodeError:
            raise LabError("Dataset preview must use UTF-8 or BOM-marked UTF-16") from None
        if limited:
            text = text[:text.rfind("\n") + 1]
        header = text.splitlines()[0] if text else ""
        try:
            delimiter = csv.Sniffer().sniff(header, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = "\t" if file.suffix.lower() == ".tsv" else ","
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        headers = next(reader, [])
        if not headers or any(not h.strip() for h in headers) or len(set(headers)) != len(headers):
            raise LabError("CSV headers are empty, blank or duplicated; correct the file header")
        if len(headers) > 100 or sum(map(len, headers)) > 8000:
            raise LabError("Too many or too-long headers")
        rows = []
        for row in reader:
            if len(rows) == 5:
                break
            rows.append([value[:200] for value in row[:100]])
        return {"lab_id": lab.id, "path": "data/" + path,
                "kernel_path": file.as_posix(), "headers": headers, "delimiter": delimiter,
                "encoding": encoding, "preview": rows, "sample_only": True,
                "bytes": size, "preview_bytes": len(raw), "sample_rows": len(rows),
                "sha256": digest}

    def search_knowledge(query: str) -> dict[str, Any]:
        """Search this lab's papers and documents; returns excerpts with page/hash citations."""
        return {"lab_id": lab.id, "hits": lab.search(query), "retrieval": "lexical"}

    def list_skills() -> dict[str, Any]:
        """Read this lab's curated procedures and applicability guidance."""
        return {"lab_id": lab.id, "skills": lab.skills()}

    def gbif_species_lookup(name: str) -> dict[str, Any]:
        """Look up a scientific species name in GBIF. Sends only this name to api.gbif.org; does not download occurrence data."""
        if not lab.enable_gbif or not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
            raise LabError("GBIF disabled or invalid species name")
        url = "https://api.gbif.org/v1/species/match?" + urlencode({"name": name.strip()})
        request = Request(url, headers={"User-Agent": "NaaVRE-research-coworker/0.2", "Accept": "application/json"})
        try:
            with build_opener(_NoRedirect()).open(request, timeout=15) as response:
                raw = response.read(65537)
            if len(raw) > 65536:
                raise LabError("GBIF response exceeds size limit")
            data = json.loads(raw)
        except Exception as exc:
            raise LabError("GBIF lookup failed: " + type(exc).__name__) from None
        keep = ("usageKey", "scientificName", "canonicalName", "rank", "status", "confidence", "matchType", "kingdom", "family", "genus", "species")
        return {"source": url, "query": name, "match": {k: data[k] for k in keep if k in data}}

    def save_note(filename: str, text: str) -> dict[str, Any]:
        """Create a new UTF-8 .md/.txt note under this lab's outputs. Host confirmation is required. Existing files are never overwritten."""
        if "/" in filename or "\\" in filename or Path(filename).suffix.lower() not in {".md", ".txt"}:
            raise LabError("Use a plain .md or .txt filename")
        if not isinstance(text, str) or not 1 <= len(text) <= 16000:
            raise LabError("Note must contain 1 to 16000 characters")
        file = lab.resolve("outputs", filename)
        content = text.encode("utf-8")
        _exclusive_write(file, content)
        return {"path": "outputs/" + filename, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    functions = [list_files, inspect_dataset, search_knowledge, list_skills, gbif_species_lookup, save_note]
    for fn in functions:
        if fn.__name__ in lab.allowed_tools:
            annotation = read_annotation
            if fn.__name__ == "save_note":
                annotation = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
            elif fn.__name__ == "gbif_species_lookup":
                annotation = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)
            server.tool(annotations=annotation, structured_output=True)(fn)

    @server.resource("lab://profile")
    def profile() -> str:
        return json.dumps(lab.public_dict())

    @server.resource("lab://skills")
    def skills_resource() -> str:
        return json.dumps(lab.skills())

    return server


def main() -> None:
    lab_id = os.environ.get("NAAVRE_MCP_LAB_ID", "")
    server = create_server(LabRegistry().get(lab_id))
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
