from pathlib import Path

import pytest

from vectornest.core.config import load_settings
from vectornest.core.exceptions import ValidationError


def test_load_settings_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "VECTORNEST_OLLAMA_HOST",
        raising=False,
    )
    monkeypatch.delenv(
        "VECTORNEST_EMBEDDING_MODEL",
        raising=False,
    )
    monkeypatch.delenv(
        "VECTORNEST_EMBEDDING_DIMENSION",
        raising=False,
    )
    monkeypatch.delenv(
        "VECTORNEST_LLM_MODEL",
        raising=False,
    )
    monkeypatch.delenv(
        "VECTORNEST_DATA_DIR",
        raising=False,
    )

    settings = load_settings()

    assert settings.ollama_host == (
        "http://127.0.0.1:11434"
    )
    assert settings.embedding_model == (
        "nomic-embed-text"
    )
    assert settings.embedding_dimension == 768
    assert settings.llm_model == "llama3.2"
    assert isinstance(
        settings.data_dir,
        Path,
    )


def test_load_settings_uses_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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

    assert settings.ollama_host == (
        "http://ollama:11434"
    )
    assert settings.embedding_model == (
        "custom-embedding"
    )
    assert settings.embedding_dimension == 1024
    assert settings.llm_model == "custom-llm"
    assert settings.data_dir == tmp_path


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
    monkeypatch.setenv(
        "VECTORNEST_EMBEDDING_DIMENSION",
        value,
    )

    with pytest.raises(ValidationError):
        load_settings()