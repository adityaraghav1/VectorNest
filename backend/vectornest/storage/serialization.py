"""Serialization helpers for VectorNest domain models."""

from datetime import datetime
from typing import Any

from vectornest.core.exceptions import ValidationError
from vectornest.core.types import DistanceMetric
from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord


def serialize_collection(collection: CollectionConfig) -> dict[str, Any]:
    """Convert a collection configuration into JSON-compatible data."""
    return {
        "name": collection.name,
        "dimension": collection.dimension,
        "distance_metric": collection.distance_metric.value,
        "description": collection.description,
    }


def deserialize_collection(data: dict[str, Any]) -> CollectionConfig:
    """Reconstruct a CollectionConfig from persisted data."""
    if not isinstance(data, dict):
        raise ValidationError("Collection data must be a dictionary.")

    try:
        metric = DistanceMetric(data["distance_metric"])
        return CollectionConfig(
            name=data["name"],
            dimension=data["dimension"],
            distance_metric=metric,
            description=data.get("description"),
        )
    except (KeyError, ValueError, TypeError) as error:
        raise ValidationError("Invalid serialized collection data.") from error


def serialize_record(record: VectorRecord) -> dict[str, Any]:
    """Convert a vector record into JSON-compatible data."""
    return {
        "id": record.id,
        "vector": record.vector.tolist(),
        "metadata": record.metadata,
        "document": record.document,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


def deserialize_record(data: dict[str, Any]) -> VectorRecord:
    """Reconstruct a VectorRecord from persisted data."""
    if not isinstance(data, dict):
        raise ValidationError("Record data must be a dictionary.")

    try:
        created_at = datetime.fromisoformat(data["created_at"])
        updated_at = datetime.fromisoformat(data["updated_at"])

        return VectorRecord(
            id=data["id"],
            vector=data["vector"],
            metadata=data.get("metadata", {}),
            document=data.get("document"),
            created_at=created_at,
            updated_at=updated_at,
        )
    except (KeyError, ValueError, TypeError) as error:
        raise ValidationError("Invalid serialized record data.") from error