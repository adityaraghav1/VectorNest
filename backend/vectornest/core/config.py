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

    ollama_host: str
    embedding_model: str
    embedding_dimension: int
    llm_model: str
    data_dir: Path


def load_settings() -> Settings:
    """Load VectorNest configuration from environment variables."""

    ollama_host = os.getenv(
        "VECTORNEST_OLLAMA_HOST",
        "http://127.0.0.1:11434",
    )

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

    return Settings(
        ollama_host=ollama_host,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        llm_model=llm_model,
        data_dir=data_dir,
    )


def _read_data_dir() -> Path:
    """Resolve the configured data directory."""

    raw_value = os.getenv("VECTORNEST_DATA_DIR")

    if raw_value is None:
        return DEFAULT_DATA_DIR

    data_dir = Path(raw_value).expanduser()

    if not data_dir.is_absolute():
        data_dir = DEFAULT_PROJECT_ROOT / data_dir

    return data_dir.resolve()


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