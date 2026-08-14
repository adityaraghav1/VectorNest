"""The canonical in-memory representation of one vector record."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from vectornest.core.exceptions import DimensionMismatchError, ValidationError

VectorArray = NDArray[np.float32]
MetadataValue = str | int | float | bool | None
Metadata = dict[str, MetadataValue]


def _utc_now() -> datetime:
    """Return a timezone-aware timestamp suitable for persisted records."""
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class VectorRecord:
    """A vector, optional source text, and filterable metadata.

    Vectors are converted to contiguous float32 arrays at the system boundary.
    float32 halves memory use compared with float64 and is standard in vector search.
    """

    id: str
    vector: VectorArray | list[float]
    metadata: Metadata = field(default_factory=dict)
    document: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Normalize inputs and enforce storage-safe domain invariants."""
        self.id = self._validate_id(self.id)
        self.vector = self._normalize_vector(self.vector)
        self.metadata = self._validate_metadata(self.metadata)
        self.document = self._validate_document(self.document)
        self.created_at = self._validate_timestamp(self.created_at, "created_at")
        self.updated_at = self._validate_timestamp(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValidationError("updated_at cannot be earlier than created_at.")

    @property
    def dimension(self) -> int:
        """Return the number of coordinates in the vector."""
        return int(self.vector.size)

    def ensure_dimension(self, expected_dimension: int) -> None:
        """Raise a clear error when this record cannot enter a collection."""
        if isinstance(expected_dimension, bool) or not isinstance(expected_dimension, int):
            raise ValidationError("Expected dimension must be an integer.")
        if expected_dimension <= 0:
            raise ValidationError("Expected dimension must be greater than zero.")
        if self.dimension != expected_dimension:
            raise DimensionMismatchError(
                f"Record '{self.id}' has dimension {self.dimension}; expected {expected_dimension}."
            )

    @staticmethod
    def _validate_id(record_id: str) -> str:
        if not isinstance(record_id, str):
            raise ValidationError("Record id must be a string.")
        normalized_id = record_id.strip()
        if not normalized_id:
            raise ValidationError("Record id cannot be empty.")
        try:
            UUID(normalized_id)
        except ValueError:
            pass  # Human-readable IDs are valid as long as they are non-empty strings.
        return normalized_id

    @staticmethod
    def _normalize_vector(vector: VectorArray | list[float]) -> VectorArray:
        try:
            normalized = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValidationError("Vector must contain numeric values.") from error
        if normalized.ndim != 1 or normalized.size == 0:
            raise ValidationError("Vector must be a non-empty, one-dimensional sequence.")
        if not np.isfinite(normalized).all():
            raise ValidationError("Vector values must all be finite numbers.")
        return np.ascontiguousarray(normalized)

    @staticmethod
    def _validate_metadata(metadata: Metadata) -> Metadata:
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary.")
        validated: Metadata = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or not key.strip():
                raise ValidationError("Metadata keys must be non-empty strings.")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValidationError(
                    f"Metadata value for '{key}' must be a scalar JSON-compatible value."
                )
            if isinstance(value, float) and not np.isfinite(value):
                raise ValidationError(f"Metadata value for '{key}' must be finite.")
            normalized_key = key.strip()
            if normalized_key in validated:
                raise ValidationError(
                    f"Metadata contains duplicate key after normalization: '{normalized_key}'."
                )
            validated[normalized_key] = value
        return validated

    @staticmethod
    def _validate_document(document: str | None) -> str | None:
        if document is not None and not isinstance(document, str):
            raise ValidationError("Document must be a string or None.")
        return document

    @staticmethod
    def _validate_timestamp(timestamp: Any, field_name: str) -> datetime:
        if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
            raise ValidationError(f"{field_name} must be a timezone-aware datetime.")
        return timestamp.astimezone(timezone.utc)
