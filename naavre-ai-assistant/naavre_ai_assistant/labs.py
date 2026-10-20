"""Admin-configured local lab stores. See the main README for the trust boundary."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

AREAS = frozenset({"data", "knowledge", "skills", "outputs"})
READ_TOOLS = ("list_files", "inspect_dataset", "search_knowledge", "list_skills")
KNOWN_TOOLS = frozenset((*READ_TOOLS, "gbif_species_lookup", "save_note"))
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
MAX_PAGES = 50
MAX_FILES = 64
MAX_SCAN_ENTRIES = 512
PDF_TIMEOUT = 10
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_RESERVED = re.compile(r"(?:con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)", re.I)


class LabError(ValueError):
    """A bounded, public-safe lab error (never includes an absolute path)."""


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise LabError("Invalid lab or skill identifier")
    return value


def _authorized(lab_id: str, permitted: frozenset[str] | None = None) -> None:
    configured = os.environ.get("NAAVRE_ALLOWED_LABS")
    if configured is not None:
        allowed = {_identifier(item.strip()) for item in configured.split(",") if item.strip()}
        if lab_id not in allowed:
            raise LabError("Lab unavailable or access denied")
    if permitted is not None and lab_id not in permitted:
        raise LabError("Lab unavailable or access denied")


def _parts(relative: str, *, empty: bool = False) -> list[str]:
    if not isinstance(relative, str) or len(relative) > 240:
        raise LabError("Invalid relative path")
    if relative == "" and empty:
        return []
    parts = relative.split("/")
    if any(
        not part or part in {".", ".."} or part[-1:] in {".", " "}
        or _RESERVED.match(part) or any(ord(c) < 32 or c in '\\:*?"<>|' for c in part)
        for part in parts
    ):
        raise LabError("Invalid relative path")
    return parts


def _linked(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _confined(base: Path, relative: str, *, empty: bool = False) -> Path:
    """Reject links, including Windows junctions; do not normalize traversal away."""
    parts = _parts(relative, empty=empty)
    candidate = base
    try:
        if _linked(base):
            raise LabError("Linked paths are not allowed")
        for part in parts:
            candidate = candidate / part
            if _linked(candidate):
                raise LabError("Linked paths are not allowed")
        candidate.resolve().relative_to(base.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, LabError):
            raise
        raise LabError("Path is outside the lab area") from None
    return candidate


def _read(path: Path, limit: int) -> bytes:
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
                raise LabError("Only unlinked regular files are supported")
            if info.st_size > limit:
                raise LabError("File exceeds the size limit")
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise LabError("File exceeds the size limit")
        return content
    except OSError:
        raise LabError("File unavailable") from None


def _exclusive_write(path: Path, content: bytes) -> None:
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
            stream.write(content)
    except FileExistsError:
        raise LabError("File already exists; choose another filename") from None
    except OSError:
        raise LabError("Unable to create file") from None


def _table_snapshot(path: Path, *, preview_bytes: int = 65_536,
                    max_bytes: int = 512 * 1024 * 1024) -> tuple[bytes, str, int]:
    """Hash a regular lab table in chunks; retain only a bounded prefix.

    This is file identity plus preview, NOT full-data quality validation.
    Keep the same link protections as document reads and reject changed files.
    """
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink > 1:
                raise LabError("Only unlinked regular files are supported")
            if before.st_size > max_bytes:
                raise LabError("Dataset exceeds the 512 MiB inspection limit")
            digest = hashlib.sha256()
            prefix = bytearray()
            total = 0
            while block := stream.read(1024 * 1024):
                total += len(block)
                if total > max_bytes:
                    raise LabError("Dataset exceeds the 512 MiB inspection limit")
                digest.update(block)
                remaining = preview_bytes - len(prefix)
                if remaining > 0:
                    prefix.extend(block[:remaining])
            after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or total != after.st_size:
            raise LabError("Dataset changed during inspection; inspect again")
        return bytes(prefix), digest.hexdigest(), total
    except OSError:
        raise LabError("File unavailable") from None


def _extract_pdf_worker() -> None:
    """Isolated parser: no package import, network, shell, OCR or embedded actions."""
    try:
        if os.name == "posix":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
            resource.setrlimit(resource.RLIMIT_CPU, (PDF_TIMEOUT, PDF_TIMEOUT))
        from pypdf import PdfReader, apply_configuration

        content = sys.stdin.buffer.read(MAX_DOCUMENT_BYTES + 1)
        if len(content) > MAX_DOCUMENT_BYTES:
            raise LabError("Document too large")
        with apply_configuration(
            maximum_declared_stream_length=8 * 1024**2,
            zlib_maximum_output_length=8 * 1024**2,
            lzw_maximum_output_length=8 * 1024**2,
            array_based_stream_maximum_output_length=8 * 1024**2,
            run_length_maximum_output_length=8 * 1024**2,
            page_tree_maximum_entries=256,
            page_tree_maximum_depth=20,
            xform_maximum_invocations_per_extraction=100,
            jbig2dec_binary=None,
        ):
            reader = PdfReader(io.BytesIO(content), strict=True)
            if reader.is_encrypted or len(reader.pages) > MAX_PAGES:
                raise LabError("Unsupported PDF")
            pages = []
            count = 0
            for page in reader.pages:
                text = page.extract_text() or ""
                count += len(text)
                if count > MAX_TEXT_CHARS:
                    raise LabError("Too much extracted text")
                pages.append(text)
        sys.stdout.write(json.dumps(pages, ensure_ascii=True))
    except Exception:
        sys.exit(2)


def _extract(filename: str, content: bytes) -> list[tuple[int | None, str]]:
    if not content or len(content) > MAX_DOCUMENT_BYTES:
        raise LabError("Document is empty or exceeds the 4 MiB limit")
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--extract-pdf"],
                input=content, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=PDF_TIMEOUT, check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env={key: value for key, value in os.environ.items()
                     if key.upper() in {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH"}},
            )
            texts = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            raise LabError("PDF is invalid, encrypted, or exceeds extraction limits") from None
        pages = [(number, text) for number, text in enumerate(texts, 1)]
    elif suffix in {".txt", ".md", ".markdown"}:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeError:
            raise LabError("Text documents must use UTF-8") from None
        if "\x00" in text or len(text) > MAX_TEXT_CHARS:
            raise LabError("Text document is binary or exceeds the text limit")
        pages = [(None, text)]
    else:
        raise LabError("Supported documents: PDF, UTF-8 text and Markdown")
    if not any(text.strip() for _, text in pages):
        raise LabError("Document has no extractable text; OCR is not available")
    return pages


def _chunks(pages: list[tuple[int | None, str]]) -> Iterable[tuple[int | None, str]]:
    for page, text in pages:
        for start in range(0, len(text), 1400):
            chunk = text[start:start + 1600].strip()
            if chunk:
                yield page, chunk


@dataclass(frozen=True)
class Lab:
    id: str
    name: str
    description: str
    style: str
    frameworks: tuple[str, ...]
    root: Path
    allowed_tools: tuple[str, ...]
    enable_gbif: bool
    _skill_ids: tuple[str, ...] = field(repr=False)
    _permitted: frozenset[str] | None = field(default=None, repr=False)
    _cache: dict[str, list[tuple[int | None, str]]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def _check(self) -> None:
        _authorized(self.id, self._permitted)
        if _linked(self.root.parent) or _linked(self.root) or not self.root.is_dir():
            raise LabError("Lab unavailable or access denied")

    def public_dict(self) -> dict[str, Any]:
        self._check()
        return {"id": self.id, "name": self.name, "description": self.description,
                "style": self.style, "frameworks": list(self.frameworks),
                "document_count": len(self.files("knowledge")),
                "skill_count": len(self._skill_ids),
                "allowed_tools": list(self.allowed_tools), "enable_gbif": self.enable_gbif}

    def resolve(self, area: str, relative: str) -> Path:
        self._check()
        if area not in AREAS:
            raise LabError("Unknown lab area")
        base = _confined(self.root, area)
        if not base.is_dir():
            raise LabError("Lab area unavailable")
        return _confined(base, relative, empty=True)

    def files(self, area: str = "data", relative: str = "") -> list[dict[str, Any]]:
        """Bounded recursive metadata listing; links are never followed."""
        start = self.resolve(area, relative)
        if not start.is_dir():
            raise LabError("Directory unavailable")
        base = self.resolve(area, "")
        found: list[dict[str, Any]] = []
        seen = 0
        pending = [start]
        try:
            while pending:
                directory = pending.pop()
                self.resolve(area, directory.relative_to(base).as_posix()
                             if directory != base else "")
                with os.scandir(directory) as entries:
                    for entry in entries:
                        seen += 1
                        if seen > MAX_SCAN_ENTRIES:
                            raise LabError("Directory exceeds the scan limit")
                        if entry.name.startswith("."):
                            continue
                        path = Path(entry.path)
                        if _linked(path):
                            continue
                        rel = path.relative_to(base).as_posix()
                        self.resolve(area, rel)
                        # Windows DirEntry caches st_nlink=0; use a fresh stat for
                        # the hard-link check or every normal file is omitted.
                        info = path.stat(follow_symlinks=False)
                        if stat.S_ISDIR(info.st_mode):
                            pending.append(path)
                        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                            found.append({"path": f"{area}/{rel}", "bytes": info.st_size})
                            if len(found) > MAX_FILES:
                                raise LabError("Directory exceeds the 64-file limit")
        except OSError:
            raise LabError("Directory unavailable") from None
        return sorted(found, key=lambda item: item["path"])

    def add_document(self, filename: str, content: bytes) -> dict[str, Any]:
        """Explicit host upload, never an MCP tool. Does not replace existing files."""
        if len(_parts(filename)) != 1:
            raise LabError("Document filename must be a basename")
        path = self.resolve("knowledge", filename)
        if not isinstance(content, bytes):
            raise LabError("Document content must be bytes")
        pages = _extract(filename, content)
        files = self.files("knowledge")
        if len(files) >= MAX_FILES or sum(f["bytes"] for f in files) + len(content) > 16 * 1024**2:
            raise LabError("Knowledge store exceeds its file or byte budget")
        _exclusive_write(self.resolve("knowledge", filename), content)
        digest = hashlib.sha256(content).hexdigest()
        self._remember(digest, pages)
        return {"path": f"knowledge/{filename}", "hash": digest, "bytes": len(content),
                "pages": len(pages), "chunks": sum(1 for _ in _chunks(pages))}

    def _remember(self, digest: str, pages: list[tuple[int | None, str]]) -> None:
        if len(self._cache) >= 16:
            self._cache.clear()
        self._cache[digest] = pages

    def search(self, query: str) -> list[dict[str, Any]]:
        self._check()
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
            raise LabError("Query must contain 1 to 500 characters")
        tokens = set(re.findall(r"\w+", query.casefold()))
        if not tokens:
            return []
        hits: list[dict[str, Any]] = []
        total = 0
        deadline = time.monotonic() + 15
        for item in self.files("knowledge"):
            if Path(item["path"]).suffix.lower() not in {".pdf", ".txt", ".md", ".markdown"}:
                continue
            if time.monotonic() > deadline:
                raise LabError("Knowledge search exceeded its time budget")
            total += item["bytes"]
            if total > 16 * 1024**2:
                raise LabError("Knowledge store exceeds the search byte budget")
            try:
                content = _read(self.resolve("knowledge", item["path"][10:]), MAX_DOCUMENT_BYTES)
                digest = hashlib.sha256(content).hexdigest()
                pages = self._cache.get(digest)
                if pages is None:
                    pages = _extract(item["path"], content)
                    self._remember(digest, pages)
            except LabError:
                # Admin-provided unreadable/unsupported documents cannot poison other hits.
                continue
            for page, chunk in _chunks(pages):
                words = set(re.findall(r"\w+", chunk.casefold()))
                score = len(tokens & words)
                if score:
                    hits.append({"text": chunk, "score": score,
                                 "citation": {"path": item["path"], "page": page, "hash": digest}})
        return sorted(hits, key=lambda hit: -hit["score"])[:8]

    def skills(self) -> list[dict[str, Any]]:
        self._check()
        result = []
        for skill_id in self._skill_ids:
            path = self.resolve("skills", f"{skill_id}.md")
            content = _read(path, 16_384)
            try:
                text = content.decode("utf-8-sig")
            except UnicodeError:
                raise LabError("Curated skill must use UTF-8") from None
            if "\x00" in text:
                raise LabError("Curated skill must be text")
            result.append({"id": skill_id, "name": skill_id.replace("-", " "), "content": text,
                           "citation": {"path": f"skills/{skill_id}.md", "page": None,
                                        "hash": hashlib.sha256(content).hexdigest()}})
        return result


class LabRegistry:
    def __init__(self, root: str | Path | None = None, *,
                 permitted_labs: Iterable[str] | None = None):
        from jupyter_core.paths import jupyter_data_dir
        checkout = Path(__file__).resolve().parent.parent
        if (checkout / "pyproject.toml").is_file():
            workspace_labs = checkout.parent / "Labs"
            default = (workspace_labs if workspace_labs.is_dir()
                       else checkout.parent / "Testing" / "fixtures" / "labs")
        else:
            default = Path(jupyter_data_dir()) / "naavre-ai-assistant" / "labs"
        selected = Path(root if root is not None else os.environ.get("NAAVRE_LABS_DIR", str(default)))
        if _linked(selected):
            raise LabError("Linked labs root is not allowed")
        self.root = selected.resolve()
        if isinstance(permitted_labs, str):
            raise LabError("permitted_labs must be a sequence of exact identifiers")
        self.permitted_labs = (None if permitted_labs is None else
                               frozenset(_identifier(item) for item in permitted_labs))

    def get(self, lab_id: str) -> Lab:
        _identifier(lab_id)
        _authorized(lab_id, self.permitted_labs)
        root = _confined(self.root, lab_id)
        try:
            config = json.loads(_read(_confined(root, "lab.json"), 32_768))
        except (ValueError, OSError):
            raise LabError("Lab unavailable or invalid configuration") from None
        fields = {"id", "name", "description", "style", "frameworks", "skills",
                  "allowed_tools", "enable_gbif"}
        if not isinstance(config, dict) or set(config) - fields or config.get("id") != lab_id:
            raise LabError("Invalid lab configuration")
        for key, maximum in (("name", 120), ("description", 2000), ("style", 2000)):
            value = config.get(key, "")
            if not isinstance(value, str) or len(value) > maximum:
                raise LabError("Invalid lab profile")
        frameworks = config.get("frameworks", [])
        skills = config.get("skills", [])
        tools = config.get("allowed_tools", list(READ_TOOLS))
        for value, limit in ((frameworks, 20), (skills, 8), (tools, len(KNOWN_TOOLS))):
            if (not isinstance(value, list) or len(value) > limit
                    or any(not isinstance(item, str) or len(item) > 120 for item in value)
                    or len(set(value)) != len(value)):
                raise LabError("Invalid lab configuration list")
        for skill in skills:
            _identifier(skill)
        if set(tools) - KNOWN_TOOLS or type(config.get("enable_gbif", False)) is not bool:
            raise LabError("Invalid tool allowlist or GBIF setting")
        enable_gbif = config.get("enable_gbif", False)
        if not enable_gbif:
            tools = [name for name in tools if name != "gbif_species_lookup"]
        for area in AREAS:
            if not _confined(root, area).is_dir():
                raise LabError("Each lab needs data, knowledge, skills and outputs directories")
        return Lab(lab_id, config.get("name") or lab_id, config.get("description", ""),
                   config.get("style", ""), tuple(frameworks), root, tuple(tools),
                   enable_gbif, tuple(skills), self.permitted_labs)

    def list_labs(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        if _linked(self.root):
            raise LabError("Linked labs root is not allowed")
        result = []
        try:
            with os.scandir(self.root) as entries:
                for count, entry in enumerate(entries, 1):
                    if count > MAX_SCAN_ENTRIES:
                        raise LabError("Too many lab entries")
                    if not _ID.fullmatch(entry.name) or not entry.is_dir(follow_symlinks=False):
                        continue
                    if _linked(Path(entry.path)):
                        continue
                    try:
                        _authorized(entry.name, self.permitted_labs)
                    except LabError:
                        continue
                    if (Path(entry.path) / "lab.json").exists():
                        result.append(self.get(entry.name).public_dict())
        except OSError:
            raise LabError("Labs root unavailable") from None
        return sorted(result, key=lambda lab: lab["id"])


if __name__ == "__main__":
    if sys.argv[1:] == ["--extract-pdf"]:
        _extract_pdf_worker()
    else:
        sys.exit(2)
