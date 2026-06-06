from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from dotenv import load_dotenv
from openai import OpenAI


MAX_DESCRIPTION_LENGTH = 50_000
MAX_CODE_LENGTH = 100_000
MAX_TRACEBACK_LENGTH = 50_000
MAX_MAPPING_LENGTH = 50_000


class LLMServiceError(RuntimeError):
    """An expected configuration or upstream LLM failure."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None
    base_url: str | None
    model: str
    timeout_seconds: float
    max_tokens: int
    configuration_error: str | None = None

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        load_dotenv()
        configuration_error = None
        try:
            timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
            max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        except ValueError:
            timeout_seconds = 60
            max_tokens = 2000
            configuration_error = (
                "OPENAI_TIMEOUT_SECONDS and OPENAI_MAX_TOKENS must be numeric."
            )
        if not configuration_error and (timeout_seconds <= 0 or max_tokens <= 0):
            configuration_error = (
                "OPENAI_TIMEOUT_SECONDS and OPENAI_MAX_TOKENS must be greater than zero."
            )
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE") or None,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            configuration_error=configuration_error,
        )


class LLMCodeGenerator:
    """Generate and repair notebook code through an OpenAI-compatible API."""

    def __init__(self, config: LLMConfig | None = None, client: Any | None = None) -> None:
        self.config = config or LLMConfig.from_environment()
        self._client = client

    @property
    def configured(self) -> bool:
        return not self.config.configuration_error and bool(self.config.api_key or self._client)

    def generate_code(
        self,
        description: str,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        language: str = "python",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        description = _bounded_text(description, "description", MAX_DESCRIPTION_LENGTH)
        language = _normalize_language(language)
        input_text = _bounded_mapping(inputs or {}, "inputs")
        output_text = _bounded_mapping(outputs or {}, "outputs")
        metadata_text = _bounded_mapping(metadata or {}, "metadata")
        prompt = f"""Create one executable {language} Jupyter cell for a NaaVRE workflow component.

Scientific intent:
{description}

Declared inputs:
{input_text}

Declared outputs:
{output_text}

Cell metadata:
{metadata_text}

Requirements:
- Return only executable {language} source code, with no Markdown fence or explanation.
- Preserve the declared component interface and use clear variable names.
- Files shared with downstream NaaVRE components must be read from or written under /tmp/data/.
- Create /tmp/data/ before writing there when needed.
- Do not embed credentials, install packages, start services, or execute shell commands.
- Prefer deterministic operations and explicit error messages suitable for scientific workflows.
"""
        return self._complete(prompt)

    def repair_code(
        self,
        description: str,
        code: str,
        traceback: str,
        language: str = "python",
        attempt: int = 1,
    ) -> str:
        description = _bounded_text(description, "description", MAX_DESCRIPTION_LENGTH)
        code = _bounded_text(code, "code", MAX_CODE_LENGTH)
        traceback = _bounded_text(traceback, "traceback", MAX_TRACEBACK_LENGTH)
        language = _normalize_language(language)
        prompt = f"""Repair an executable {language} Jupyter cell for a NaaVRE workflow component.

Scientific intent:
{description}

Current code:
{code}

Execution error from repair attempt {attempt}:
{traceback}

Return the complete corrected cell as source code only. Preserve the scientific intent and public
input/output variable names. Keep shared file I/O under /tmp/data/. Do not hide errors with broad
exception handlers, install packages, embed credentials, run shell commands, or return Markdown.
"""
        return self._complete(prompt)

    def _complete(self, prompt: str) -> str:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research software engineer producing reviewable, "
                            "reproducible notebook code for NaaVRE."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=self.config.max_tokens,
            )
        except Exception as error:
            raise LLMServiceError(
                "The configured LLM request failed. Check the server log and provider configuration."
            ) from error

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMServiceError("The configured LLM returned an empty response.")
        return strip_code_fence(content)

    def _get_client(self) -> Any:
        if self.config.configuration_error:
            raise LLMServiceError(self.config.configuration_error)
        if self._client is not None:
            return self._client
        if not self.config.api_key:
            raise LLMServiceError(
                "OPENAI_API_KEY is not configured on the Jupyter Server. "
                "Set it before starting JupyterLab."
            )
        kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout_seconds,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)
        return self._client


def strip_code_fence(content: str) -> str:
    stripped = content.strip()
    match = re.fullmatch(r"```(?:python|py|r)?\s*\n?(.*?)\n?```", stripped, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else stripped).strip()


def _bounded_text(value: str, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    if len(value) > limit:
        raise ValueError(f"{name} exceeds the {limit:,}-character limit.")
    return value.strip()


def _normalize_language(language: str) -> str:
    if not isinstance(language, str):
        raise ValueError("language must be either 'python' or 'r'.")
    normalized = language.strip().lower()
    aliases = {"py": "python", "python3": "python", "rscript": "r"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"python", "r"}:
        raise ValueError("language must be either 'python' or 'r'.")
    return normalized


def _pretty_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Metadata must be JSON serializable: {error}") from error


def _bounded_mapping(value: Mapping[str, Any], name: str) -> str:
    serialized = _pretty_json(value)
    if len(serialized) > MAX_MAPPING_LENGTH:
        raise ValueError(f"{name} exceeds the {MAX_MAPPING_LENGTH:,}-character limit.")
    return serialized
