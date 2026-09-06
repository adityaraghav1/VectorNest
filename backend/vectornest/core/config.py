"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from vectornest.core.exceptions import ValidationError

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = DEFAULT_PROJECT_ROOT / "data" / "vectornest"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for VectorNest."""

    ai_provider: str
    ollama_host: str
    gemini_api_key: str | None
    embedding_model: str
    embedding_dimension: int
    llm_model: str
    data_dir: Path
    cors_origins: tuple[str, ...]


def load_settings() -> Settings:
    """Load VectorNest configuration from environment variables."""

    ai_provider = os.getenv(
        "VECTORNEST_AI_PROVIDER",
        "ollama",
    ).strip().lower()

    if ai_provider not in {"ollama", "gemini"}:
        raise ValidationError(
            "VECTORNEST_AI_PROVIDER must be either "
            "'ollama' or 'gemini'."
        )

    ollama_host = os.getenv(
        "VECTORNEST_OLLAMA_HOST",
        "http://127.0.0.1:11434",
    )

    gemini_api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if gemini_api_key is not None:
        gemini_api_key = gemini_api_key.strip()

    if ai_provider == "gemini" and not gemini_api_key:
        raise ValidationError(
            "GEMINI_API_KEY is required when "
            "VECTORNEST_AI_PROVIDER=gemini."
        )

    if ai_provider == "gemini":
        embedding_model = os.getenv(
            "VECTORNEST_EMBEDDING_MODEL",
            "gemini-embedding-2",
        )

        llm_model = os.getenv(
            "VECTORNEST_LLM_MODEL",
            "gemini-3.7-flash",
        )
    else:
        embedding_model = os.getenv(
            "VECTORNEST_EMBEDDING_MODEL",
            "nomic-embed-text",
        )

        llm_model = os.getenv(
            "VECTORNEST_LLM_MODEL",
            "llama3.2",
        )

    embedding_dimension = _read_positive_int(
        "VECTORNEST_EMBEDDING_DIMENSION",
        default=768,
    )

    data_dir = _read_data_dir()
    cors_origins = _read_cors_origins()

    return Settings(
        ai_provider=ai_provider,
        ollama_host=ollama_host,
        gemini_api_key=gemini_api_key,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        llm_model=llm_model,
        data_dir=data_dir,
        cors_origins=cors_origins,
    )


def _read_data_dir() -> Path:
    """Resolve the configured data directory."""

    raw_value = os.getenv(
        "VECTORNEST_DATA_DIR"
    )

    if raw_value is None:
        return DEFAULT_DATA_DIR

    data_dir = Path(
        raw_value
    ).expanduser()

    if not data_dir.is_absolute():
        data_dir = (
            DEFAULT_PROJECT_ROOT
            / data_dir
        )

    return data_dir.resolve()


def _read_cors_origins() -> tuple[str, ...]:
    """Read allowed frontend origins."""

    raw_value = os.getenv(
        "VECTORNEST_CORS_ORIGINS",
        (
            "http://127.0.0.1:5500,"
            "http://localhost:5500"
        ),
    )

    origins = tuple(
        origin.strip()
        for origin in raw_value.split(",")
        if origin.strip()
    )

    if not origins:
        raise ValidationError(
            "VECTORNEST_CORS_ORIGINS must contain "
            "at least one origin."
        )

    return origins


def _read_positive_int(
    variable_name: str,
    default: int,
) -> int:
    """Read a positive integer environment variable."""

    raw_value = os.getenv(
        variable_name,
        str(default),
    )

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValidationError(
            f"{variable_name} must be an integer."
        ) from exc

    if value <= 0:
        raise ValidationError(
            f"{variable_name} must be greater than zero."
        )

    return value