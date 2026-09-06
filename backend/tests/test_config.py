from pathlib import Path

import pytest

from vectornest.core.config import (
    DEFAULT_PROJECT_ROOT,
    load_settings,
)
from vectornest.core.exceptions import ValidationError


def clear_ai_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove AI-related environment variables."""

    variables = [
        "VECTORNEST_AI_PROVIDER",
        "VECTORNEST_OLLAMA_HOST",
        "VECTORNEST_EMBEDDING_MODEL",
        "VECTORNEST_EMBEDDING_DIMENSION",
        "VECTORNEST_LLM_MODEL",
        "VECTORNEST_DATA_DIR",
        "GEMINI_API_KEY",
    ]

    for variable in variables:
        monkeypatch.delenv(
            variable,
            raising=False,
        )


def test_load_settings_uses_ollama_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_ai_environment(monkeypatch)

    settings = load_settings()

    assert settings.ai_provider == "ollama"
    assert settings.ollama_host == (
        "http://127.0.0.1:11434"
    )
    assert settings.gemini_api_key is None
    assert settings.embedding_model == (
        "nomic-embed-text"
    )
    assert settings.embedding_dimension == 768
    assert settings.llm_model == "llama3.2"
    assert isinstance(
        settings.data_dir,
        Path,
    )


def test_load_settings_uses_gemini_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_ai_environment(monkeypatch)

    monkeypatch.setenv(
        "VECTORNEST_AI_PROVIDER",
        "gemini",
    )
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-api-key",
    )

    settings = load_settings()

    assert settings.ai_provider == "gemini"
    assert settings.gemini_api_key == (
        "test-api-key"
    )
    assert settings.embedding_model == (
        "gemini-embedding-2"
    )
    assert settings.embedding_dimension == 768
    assert settings.llm_model == (
        "gemini-3.7-flash"
    )


def test_gemini_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_ai_environment(monkeypatch)

    monkeypatch.setenv(
        "VECTORNEST_AI_PROVIDER",
        "gemini",
    )

    with pytest.raises(
        ValidationError,
        match=(
            "GEMINI_API_KEY is required when "
            "VECTORNEST_AI_PROVIDER=gemini"
        ),
    ):
        load_settings()


def test_gemini_rejects_blank_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_ai_environment(monkeypatch)

    monkeypatch.setenv(
        "VECTORNEST_AI_PROVIDER",
        "gemini",
    )
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "   ",
    )

    with pytest.raises(
        ValidationError,
        match="GEMINI_API_KEY is required",
    ):
        load_settings()


def test_invalid_ai_provider_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_ai_environment(monkeypatch)

    monkeypatch.setenv(
        "VECTORNEST_AI_PROVIDER",
        "openai",
    )

    with pytest.raises(
        ValidationError,
        match=(
            "VECTORNEST_AI_PROVIDER must be either "
            "'ollama' or 'gemini'"
        ),
    ):
        load_settings()


def test_ai_provider_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_ai_environment(monkeypatch)

    monkeypatch.setenv(
        "VECTORNEST_AI_PROVIDER",
        "  OLLAMA  ",
    )

    settings = load_settings()

    assert settings.ai_provider == "ollama"


def test_relative_data_dir_resolves_from_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_ai_environment(monkeypatch)

    monkeypatch.setenv(
        "VECTORNEST_DATA_DIR",
        "data/custom-vectornest",
    )

    settings = load_settings()

    assert settings.data_dir == (
        DEFAULT_PROJECT_ROOT
        / "data"
        / "custom-vectornest"
    ).resolve()


def test_absolute_data_dir_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clear_ai_environment(monkeypatch)

    data_dir = tmp_path / "vectornest-data"

    monkeypatch.setenv(
        "VECTORNEST_DATA_DIR",
        str(data_dir),
    )

    settings = load_settings()

    assert settings.data_dir == (
        data_dir.resolve()
    )


def test_load_settings_uses_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clear_ai_environment(monkeypatch)

    monkeypatch.setenv(
        "VECTORNEST_AI_PROVIDER",
        "ollama",
    )
    monkeypatch.setenv(
        "VECTORNEST_OLLAMA_HOST",
        "http://ollama:11434",
    )
    monkeypatch.setenv(
        "VECTORNEST_EMBEDDING_MODEL",
        "custom-embedding",
    )
    monkeypatch.setenv(
        "VECTORNEST_EMBEDDING_DIMENSION",
        "1024",
    )
    monkeypatch.setenv(
        "VECTORNEST_LLM_MODEL",
        "custom-llm",
    )
    monkeypatch.setenv(
        "VECTORNEST_DATA_DIR",
        str(tmp_path),
    )

    settings = load_settings()

    assert settings.ai_provider == "ollama"
    assert settings.ollama_host == (
        "http://ollama:11434"
    )
    assert settings.embedding_model == (
        "custom-embedding"
    )
    assert settings.embedding_dimension == 1024
    assert settings.llm_model == "custom-llm"
    assert settings.data_dir == (
        tmp_path.resolve()
    )


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-1",
        "abc",
    ],
)
def test_embedding_dimension_must_be_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_ai_environment(monkeypatch)

    monkeypatch.setenv(
        "VECTORNEST_EMBEDDING_DIMENSION",
        value,
    )

    with pytest.raises(ValidationError):
        load_settings()