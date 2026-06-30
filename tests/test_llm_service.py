from types import SimpleNamespace

import pytest

from naavre_ai_assistant.llm_service import (
    LLMCodeGenerator,
    LLMConfig,
    LLMServiceError,
    strip_code_fence,
)


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeClient:
    def __init__(self, content: str) -> None:
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def config(api_key: str | None = "test") -> LLMConfig:
    return LLMConfig(
        api_key=api_key,
        base_url=None,
        model="test-model",
        timeout_seconds=5,
        max_tokens=500,
    )


def test_strip_code_fence() -> None:
    assert strip_code_fence("```python\nprint('ok')\n```") == "print('ok')"
    assert strip_code_fence("print('ok')") == "print('ok')"


def test_generate_code_adds_naavre_constraints() -> None:
    client = FakeClient("```python\nfrom pathlib import Path\nPath('/tmp/data').mkdir()\n```")
    service = LLMCodeGenerator(config(), client)

    code = service.generate_code(
        description="Write the result file.",
        inputs={"records": "list"},
        outputs={"result.csv": "CSV"},
    )

    assert code.startswith("from pathlib")
    prompt = client.completions.last_request["messages"][1]["content"]
    assert "/tmp/data/" in prompt
    assert '"records": "list"' in prompt
    assert client.completions.last_request["temperature"] == 0.2


def test_repair_code_includes_traceback() -> None:
    client = FakeClient("print(value)")
    service = LLMCodeGenerator(config(), client)

    service.repair_code("Print the value", "print(valeu)", "NameError: valeu", attempt=2)

    prompt = client.completions.last_request["messages"][1]["content"]
    assert "NameError: valeu" in prompt
    assert "attempt 2" in prompt


def test_missing_key_is_reported_without_network_call() -> None:
    service = LLMCodeGenerator(config(api_key=None))

    with pytest.raises(LLMServiceError, match="OPENAI_API_KEY"):
        service.generate_code("Print a value")


@pytest.mark.parametrize("language", ["javascript", "", "sql"])
def test_unsupported_language_is_rejected(language: str) -> None:
    service = LLMCodeGenerator(config(), FakeClient("pass"))

    with pytest.raises(ValueError, match="python.*r"):
        service.generate_code("Do work", language=language)


def test_non_string_language_is_rejected() -> None:
    service = LLMCodeGenerator(config(), FakeClient("pass"))

    with pytest.raises(ValueError, match="python.*r"):
        service.generate_code("Do work", language=None)  # type: ignore[arg-type]


def test_oversized_metadata_is_rejected() -> None:
    service = LLMCodeGenerator(config(), FakeClient("pass"))

    with pytest.raises(ValueError, match="metadata exceeds"):
        service.generate_code("Do work", metadata={"content": "x" * 50_001})


def test_configuration_error_disables_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "not-a-number")

    service = LLMCodeGenerator()

    assert not service.configured
    with pytest.raises(LLMServiceError, match="must be numeric"):
        service.generate_code("Do work")