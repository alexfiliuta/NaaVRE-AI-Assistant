from __future__ import annotations

import asyncio
from typing import Any, Mapping

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
from tornado import web

from .llm_service import LLMCodeGenerator, LLMServiceError, SCHEMA_CONTRACT_VERSION


class AssistantAPIHandler(APIHandler):
    service: LLMCodeGenerator

    def initialize(self, service: LLMCodeGenerator) -> None:
        self.service = service

    def write_api_error(self, status: int, message: str) -> None:
        self.set_status(status)
        self.finish({"error": message})

    def notebook_context(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve run evidence from the server; browser observations are not authoritative."""
        from .agent_handlers import request_owner

        context = dict(_mapping(body.get("context"), "context"))
        context.pop("agent", None)
        run_id = context.pop("agent_run_id", None)
        if run_id is not None:
            if not isinstance(run_id, str):
                raise ValueError("agent_run_id must identify a completed investigation.")
            runtime = self.settings.get("naavre_ai_assistant_agent_runtime")
            if runtime is None:
                raise ValueError("Agent investigation service is unavailable.")
            verified = runtime.verified_context(request_owner(self.current_user), run_id)
            metadata = _mapping(body.get("metadata"), "metadata")
            selected_lab = metadata.get("lab_id")
            if selected_lab and selected_lab != verified["agent"]["lab_id"]:
                raise ValueError("The investigation belongs to another lab. Investigate again.")
            context.update(verified)
        return context


class HealthHandler(AssistantAPIHandler):
    @web.authenticated
    def get(self) -> None:
        self.finish(
            {
                "status": "ok",
                "configured": self.service.configured,
                "provider": "local",
                "model": self.service.config.model,
                "models": self.service.config.model_roles(),
                "thinking": self.service.config.thinking,
                "schemaContract": SCHEMA_CONTRACT_VERSION,
                "configurationError": self.service.config.configuration_error,
            }
        )


class AnalyzeHandler(AssistantAPIHandler):
    @web.authenticated
    async def post(self) -> None:
        body = self.get_json_body()
        if not isinstance(body, dict):
            self.write_api_error(400, "Request body must be a JSON object.")
            return

        mode = body.get("mode", "generate")
        if mode not in {"generate", "repair"}:
            self.write_api_error(400, "mode must be either 'generate' or 'repair'.")
            return

        code = body.get("code") if mode == "repair" else None
        traceback = body.get("traceback") if mode == "repair" else None
        attempt = body.get("attempt") if mode == "repair" else None
        if mode == "repair":
            if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 10:
                self.write_api_error(400, "attempt must be an integer between 1 and 10.")
                return

        try:
            analysis = await asyncio.to_thread(
                self.service.analyze_request,
                description=body.get("description", ""),
                inputs=_mapping(body.get("inputs"), "inputs"),
                outputs=_mapping(body.get("outputs"), "outputs"),
                language=body.get("language", "python"),
                metadata=_mapping(body.get("metadata"), "metadata"),
                context=self.notebook_context(body),
                code=code,
                traceback=traceback,
                attempt=attempt,
            )
        except ValueError as error:
            self.write_api_error(400, str(error))
            return
        except LLMServiceError as error:
            self.log.warning("NaaVRE analysis request failed: %s", error)
            self.write_api_error(502, str(error))
            return
        self.finish({"analysis": analysis})


class GenerateHandler(AssistantAPIHandler):
    @web.authenticated
    async def post(self) -> None:
        body = self.get_json_body()
        if not isinstance(body, dict):
            self.write_api_error(400, "Request body must be a JSON object.")
            return

        try:
            inputs = _mapping(body.get("inputs"), "inputs")
            outputs = _mapping(body.get("outputs"), "outputs")
            metadata = _mapping(body.get("metadata"), "metadata")
            context = self.notebook_context(body)
            analysis = _mapping(body.get("analysis"), "analysis")
            if not analysis:
                analysis = await asyncio.to_thread(
                    self.service.analyze_request,
                    description=body.get("description", ""),
                    inputs=inputs,
                    outputs=outputs,
                    language=body.get("language", "python"),
                    metadata=metadata,
                    context=context,
                )
            cells = await asyncio.to_thread(
                self.service.generate_cells_from_analysis,
                description=body.get("description", ""),
                inputs=inputs,
                outputs=outputs,
                language=body.get("language", "python"),
                metadata=metadata,
                context=context,
                analysis=analysis,
            )
        except ValueError as error:
            self.write_api_error(400, str(error))
            return
        except LLMServiceError as error:
            self.log.warning("NaaVRE cell-generation request failed: %s", error)
            self.write_api_error(502, str(error))
            return
        self.finish({"cells": cells, "analysis": analysis})


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
            inputs = _mapping(body.get("inputs"), "inputs")
            outputs = _mapping(body.get("outputs"), "outputs")
            metadata = _mapping(body.get("metadata"), "metadata")
            context = self.notebook_context(body)
            analysis = _mapping(body.get("analysis"), "analysis")
            if not analysis:
                analysis = await asyncio.to_thread(
                    self.service.analyze_request,
                    description=body.get("description", ""),
                    inputs=inputs,
                    outputs=outputs,
                    language=body.get("language", "python"),
                    metadata=metadata,
                    context=context,
                    code=body.get("code", ""),
                    traceback=body.get("traceback", ""),
                    attempt=attempt,
                )
            code = await asyncio.to_thread(
                self.service.repair_code_from_analysis,
                description=body.get("description", ""),
                code=body.get("code", ""),
                traceback=body.get("traceback", ""),
                language=body.get("language", "python"),
                attempt=attempt,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                context=context,
                analysis=analysis,
            )
        except ValueError as error:
            self.write_api_error(400, str(error))
            return
        except LLMServiceError as error:
            self.log.warning("NaaVRE repair request failed: %s", error)
            self.write_api_error(502, str(error))
            return
        self.finish({"code": code, "analysis": analysis})


def setup_handlers(web_app: Any, *, logger: Any = None) -> None:
    from .agent_handlers import agent_routes

    base_url = web_app.settings.get("base_url", "/")
    service = LLMCodeGenerator(logger=logger)
    common = {"service": service}
    routes = [
        (url_path_join(base_url, "naavre-ai-assistant", "health"), HealthHandler, common),
        (url_path_join(base_url, "naavre-ai-assistant", "analyze"), AnalyzeHandler, common),
        (url_path_join(base_url, "naavre-ai-assistant", "generate"), GenerateHandler, common),
        (url_path_join(base_url, "naavre-ai-assistant", "repair"), RepairHandler, common),
    ]
    routes.extend(agent_routes(web_app, service))
    web_app.add_handlers(".*$", routes)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value
