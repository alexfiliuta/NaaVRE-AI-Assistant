from __future__ import annotations

from typing import Any

from .handlers import setup_handlers

__version__ = "0.1.0"


def _jupyter_labextension_paths() -> list[dict[str, str]]:
    return [{"src": "labextension", "dest": "naavre-ai-assistant"}]


def _jupyter_server_extension_points() -> list[dict[str, str]]:
    return [{"module": "naavre_ai_assistant"}]


def _load_jupyter_server_extension(server_app: Any) -> None:
    setup_handlers(server_app.web_app)
    server_app.log.info("NaaVRE AI Assistant server extension loaded")


load_jupyter_server_extension = _load_jupyter_server_extension
