from __future__ import annotations

import asyncio
from typing import Any, Mapping

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
from tornado import web

from .llm_service import LLMCodeGenerator, LLMServiceError


class AssistantAPIHandler(APIHandler):
    service: LLMCodeGenerator

    def initialize(self, service: LLMCodeGenerator) -> None:
        self.service = service

    def write_api_error(self, status: int, message: str) -> None:
        self.set_status(status)
        self.finish({"error": message})


class HealthHandler(AssistantAPIHandler):
    @web.authenticated
    def get(self) -> None:
        self.finish(
            {
                "status": "ok",
                "configured": self.service.configured,
                "model": self.service.config.model,
                "configurationError": self.service.config.configuration_error,
            }
        )


class GenerateHandler(AssistantAPIHandler):
    @web.authenticated
    async def post(self) -> None:
        body = self.get_json_body()
        if not isinstance(body, dict):
            self.write_api_error(400, "Request body must be a JSON object.")
            return

        try:
            code = await asyncio.to_thread(
                self.service.generate_code,
                description=body.get("description", ""),
                inputs=_mapping(body.get("inputs"), "inputs"),
                outputs=_mapping(body.get("outputs"), "outputs"),
                language=body.get("language", "python"),
                metadata=_mapping(body.get("metadata"), "metadata"),
            )
        except ValueError as error:
            self.write_api_error(400, str(error))
            return
        except LLMServiceError as error:
            self.write_api_error(502, str(error))
            return
        self.finish({"code": code})


class RepairHandler(AssistantAPIHandler):
    @web.authenticated
    async def post(self) -> None:
        body = self.get_json_body()
        if not isinstance(body, dict):
            self.write_api_error(400, "Request body must be a JSON object.")
            return

        attempt = body.get("attempt", 1)
        if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 10:
            self.write_api_error(400, "attempt must be an integer between 1 and 10.")
            return

        try:
            code = await asyncio.to_thread(
                self.service.repair_code,
                description=body.get("description", ""),
                code=body.get("code", ""),
                traceback=body.get("traceback", ""),
                language=body.get("language", "python"),
                attempt=attempt,
            )
        except ValueError as error:
            self.write_api_error(400, str(error))
            return
        except LLMServiceError as error:
            self.write_api_error(502, str(error))
            return
        self.finish({"code": code})


def setup_handlers(web_app: Any) -> None:
    base_url = web_app.settings.get("base_url", "/")
    service = LLMCodeGenerator()
    common = {"service": service}
    routes = [
        (url_path_join(base_url, "naavre-ai-assistant", "health"), HealthHandler, common),
        (url_path_join(base_url, "naavre-ai-assistant", "generate"), GenerateHandler, common),
        (url_path_join(base_url, "naavre-ai-assistant", "repair"), RepairHandler, common),
    ]
    web_app.add_handlers(".*$", routes)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value
