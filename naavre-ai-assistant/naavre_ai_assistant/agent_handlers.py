"""Authenticated lab and asynchronous agent APIs for the Jupyter extension.

Configuration is server-owned. Request bodies cannot select filesystem roots,
commands, transports, URLs, or tool allowlists. Multi-user deployments should set
permitted_labs (or NAAVRE_ALLOWED_LABS); runs are additionally owner-scoped.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import functools
import os
from pathlib import Path
from typing import Any

from jupyter_core.paths import jupyter_data_dir
from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
from tornado import web

from .agent_runtime import AgentError, AgentLimits, AgentRuntime, RunConflict, RunNotFound, bounded


MAX_DOCUMENT_BYTES = 4 * 1024 * 1024


def request_owner(user: Any) -> str:
    """Use the stable authenticated identity, never a browser-provided owner."""
    if isinstance(user, bytes):
        user = user.decode("utf-8")
    if isinstance(user, str) and user:
        return user
    if isinstance(user, dict):
        value = user.get("username") or user.get("name")
    else:
        value = getattr(user, "username", None)
    if not isinstance(value, str) or not value:
        raise web.HTTPError(403, "An authenticated user identity is required.")
    return value


def api_errors(method):
    @functools.wraps(method)
    async def wrapped(self, *args, **kwargs):
        try:
            return await method(self, *args, **kwargs)
        except RunNotFound:
            self.set_status(404)
            self.finish({"error": "Lab or run not found."})
        except RunConflict as error:
            self.set_status(409)
            self.finish({"error": str(error)})
        except (AgentError, ValueError, KeyError, TypeError) as error:
            self.set_status(400)
            self.finish({"error": str(error)[:500]})
        except OSError:
            self.log.exception("NaaVRE agent storage request failed")
            self.set_status(500)
            self.finish({"error": "The lab or agent storage operation failed."})
    return wrapped


class AgentAPIHandler(APIHandler):
    def initialize(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    @property
    def owner(self):
        return request_owner(self.current_user)

    def body(self, allowed, required=()) -> dict:
        body = self.get_json_body()
        if not isinstance(body, dict) or set(body) - set(allowed) or set(required) - set(body):
            raise AgentError("Request body contains missing or unsupported fields.")
        return body


class LabsHandler(AgentAPIHandler):
    @web.authenticated
    @api_errors
    async def get(self):
        self.owner
        labs = []
        for item in await asyncio.to_thread(self.runtime.registry.list_labs):
            lab_id = item.get("id") if isinstance(item, dict) else item.id
            try:
                lab = self.runtime.lab(lab_id)
            except RunNotFound:
                continue
            info = lab.public_dict()
            info = {key: info.get(key) for key in ("id", "name", "description", "style", "frameworks")}
            if isinstance(item, dict) and "document_count" in item:
                info["document_count"] = item["document_count"]
            else:
                info["document_count"] = len(await asyncio.to_thread(lab.files, "knowledge"))
            info["skill_count"] = len(await asyncio.to_thread(lab.skills))
            labs.append(info)
        self.finish({"labs": labs})


class LabDocumentsHandler(AgentAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self, lab_id):
        self.owner
        lab = self.runtime.lab(lab_id)
        body = self.body({"filename", "content_base64"}, {"filename", "content_base64"})
        filename = body["filename"]
        if (not isinstance(filename, str) or not filename or len(filename) > 200
                or any(char in filename for char in '/\\:\x00')
                or filename in {".", ".."} or filename != filename.strip()
                or filename.endswith(".")):
            raise AgentError("filename must be a safe basename.")
        encoded = body["content_base64"]
        if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_DOCUMENT_BYTES + 2) // 3):
            raise AgentError("Document exceeds the 4 MiB upload limit.")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise AgentError("content_base64 is invalid.") from error
        if not content or len(content) > MAX_DOCUMENT_BYTES:
            raise AgentError("Document must contain between 1 byte and 4 MiB.")
        metadata = await asyncio.to_thread(lab.add_document, filename, content)
        self.set_status(201)
        self.finish({key: metadata[key] for key in ("path", "hash", "pages", "chunks", "bytes")
                     if key in metadata})


class LabKnowledgeHandler(AgentAPIHandler):
    @web.authenticated
    @api_errors
    async def get(self, lab_id):
        self.owner
        lab = self.runtime.lab(lab_id)
        query = self.get_query_argument("q", "")
        if len(query) > 2000:
            raise AgentError("Knowledge query is too long.")
        documents, skills = await asyncio.gather(
            asyncio.to_thread(lab.search, query) if query.strip()
            else asyncio.to_thread(lab.files, "knowledge"), asyncio.to_thread(lab.skills))
        self.finish({"documents": [bounded(item, 4000) for item in documents[:20]],
                     "skills": [bounded(item, 4000) for item in skills[:20]],
                     "truncated": len(documents) > 20 or len(skills) > 20})


class AgentRunsHandler(AgentAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self):
        body = self.body({"lab_id", "description", "context"}, {"lab_id", "description"})
        result = self.runtime.start(self.owner, body["lab_id"], body["description"], body.get("context"))
        self.set_status(202)
        self.finish(result)


class AgentRunHandler(AgentAPIHandler):
    @web.authenticated
    @api_errors
    async def get(self, run_id):
        self.finish(self.runtime.get(self.owner, run_id))


class AgentDecisionHandler(AgentAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self, run_id):
        body = self.body({"approve"}, {"approve"})
        self.finish(self.runtime.decide(self.owner, run_id, body["approve"]))


class AgentCancelHandler(AgentAPIHandler):
    @web.authenticated
    @api_errors
    async def post(self, run_id):
        self.finish(self.runtime.cancel(self.owner, run_id))


def _names(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def configured_limits(settings):
    """Server-only job budgets, separate from LOCAL_LLM_* transport settings.

    NAAVRE_AGENT_MAX_STEPS defaults to 12. RUN_SECONDS and OPERATION_SECONDS have
    no deadline by default. APPROVAL_SECONDS defaults to 300. Empty/none/off values
    disable a deadline. The naavre_ai_assistant_agent_limits mapping takes priority
    over these NAAVRE_AGENT_* environment variables and accepts AgentLimits fields.
    """
    configured = {}
    for name in ("max_steps", "run_seconds", "operation_seconds", "approval_seconds"):
        value = os.getenv("NAAVRE_AGENT_" + name.upper())
        if value is None:
            continue
        if name == "max_steps":
            configured[name] = int(value)
        else:
            configured[name] = None if value.strip().lower() in {"", "none", "off"} else float(value)
    overrides = settings.get("naavre_ai_assistant_agent_limits", {})
    if not isinstance(overrides, dict):
        raise ValueError("naavre_ai_assistant_agent_limits must be a mapping.")
    configured.update(overrides)
    return AgentLimits(**configured)


def agent_routes(web_app, service):
    from .labs import LabRegistry

    settings = web_app.settings
    permitted = _names(settings.get("naavre_ai_assistant_permitted_labs"))
    registry = LabRegistry(root=settings.get("naavre_ai_assistant_labs_root"),
                           permitted_labs=permitted)
    state_root = settings.get("naavre_ai_assistant_state_root") or os.getenv("NAAVRE_AGENT_STATE_DIR")
    state_root = state_root or Path(jupyter_data_dir()) / "naavre-ai-assistant" / "agent-runs"
    runtime = AgentRuntime(
        registry, state_root, service=service, permitted_labs=permitted,
        allowed_tools=_names(settings.get("naavre_ai_assistant_agent_tools", os.getenv("NAAVRE_AGENT_TOOLS"))),
        limits=configured_limits(settings),
    )
    settings["naavre_ai_assistant_agent_runtime"] = runtime
    base = url_path_join(settings.get("base_url", "/"), "naavre-ai-assistant")
    common = {"runtime": runtime}
    return [(url_path_join(base, suffix), handler, common) for suffix, handler in (
        ("labs", LabsHandler),
        (r"labs/([A-Za-z0-9_-]+)/documents", LabDocumentsHandler),
        (r"labs/([A-Za-z0-9_-]+)/knowledge", LabKnowledgeHandler),
        ("agent/runs", AgentRunsHandler),
        (r"agent/runs/([a-f0-9]{32})", AgentRunHandler),
        (r"agent/runs/([a-f0-9]{32})/decision", AgentDecisionHandler),
        (r"agent/runs/([a-f0-9]{32})/cancel", AgentCancelHandler),
    )]
