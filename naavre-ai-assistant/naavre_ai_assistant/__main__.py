"""Small administrator CLI; this is not a separate application server."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request
from urllib.parse import urlparse

from . import __version__
from .labs import LabRegistry, READ_TOOLS, _identifier
from .llm_service import LLMConfig
from .llm_transport import open_request


def check(probe: bool = False) -> int:
    """Report safe configuration; optionally check remote model inventory, not inference."""
    config = LLMConfig.from_environment()
    report = {"version": __version__, "backend": "Jupyter Server / Tornado",
              "model_roles": config.model_roles(), "configured": not config.configuration_error,
              "connection": "HTTPS gateway" if (config.base_url or "").startswith("https:") else "loopback / SSH tunnel",
              "frontend_bundled": (Path(__file__).parent / "labextension" / "package.json").is_file()}
    try:
        report["labs"] = [lab["id"] for lab in LabRegistry().list_labs()]
        if config.configuration_error:
            raise ValueError(config.configuration_error)
        if probe:
            parsed = urlparse(config.base_url)
            request = Request(f"{parsed.scheme}://{parsed.netloc}/api/tags",
                              headers=config.connection.headers())
            context = config.connection.ssl_context() if parsed.scheme == "https" else None
            # This short diagnostic timeout does not alter inference timeouts.
            with open_request(request, timeout=10, context=context) as response:
                payload = json.loads(response.read(1024 * 1024))
            available = {model["name"] for model in payload.get("models", [])}
            report["selected_models_installed"] = all(model in available for model in config.model_roles().values())
            if not report["selected_models_installed"]:
                raise ValueError("A selected model is not installed on the configured Ollama server.")
    except Exception:
        # Never print exception text from TLS, HTTP, URLs or secret file parsers.
        report["ok"] = False
        report["error"] = config.configuration_error or "Configuration or connection check failed; review server settings."
    else:
        report["ok"] = report["frontend_bundled"]
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def init_lab(root: Path, lab_id: str, name: str) -> Path:
    """Create an empty local lab, refusing to replace an existing directory."""
    _identifier(lab_id)
    if not name.strip() or len(name) > 120:
        raise ValueError("Lab name must contain 1 to 120 characters.")
    registry = LabRegistry(root)
    target = registry.root / lab_id
    target.mkdir(parents=True, exist_ok=False)
    for area in ("data", "knowledge", "skills", "outputs"):
        (target / area).mkdir()
    profile = {"id": lab_id, "name": name, "description": "", "style": "",
               "frameworks": ["python-standard-library"], "skills": [],
               "allowed_tools": list(READ_TOOLS), "enable_gbif": False}
    (target / "lab.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser("check", help="Check configuration without starting Jupyter")
    diagnostic.add_argument("--probe", action="store_true", help="Also check Ollama model inventory")
    create = commands.add_parser("init-lab", help="Create an empty per-user lab; never downloads data")
    create.add_argument("lab_id")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(args.probe)
    try:
        print(init_lab(args.root, args.lab_id, args.name))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
