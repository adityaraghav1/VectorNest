from datetime import UTC, datetime

import pytest

from vectornest.core.exceptions import ValidationError
from vectornest.core.types import DistanceMetric
from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord
from vectornest.storage.serialization import (
    deserialize_collection,
    deserialize_record,
    serialize_collection,
    serialize_record,
)


def test_collection_round_trip() -> None:
    collection = CollectionConfig(
        name="documents",
        dimension=3,
        distance_metric=DistanceMetric.COSINE,
        description="Test collection",
    )

    serialized = serialize_collection(collection)
    restored = deserialize_collection(serialized)

    assert restored.name == collection.name
    assert restored.dimension == collection.dimension
    assert restored.distance_metric == collection.distance_metric
    assert restored.description == collection.description


def test_record_round_trip() -> None:
    created_at = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
    updated_at = datetime(2026, 8, 18, 13, 0, tzinfo=UTC)

    record = VectorRecord(
        id="doc-1",
        vector=[0.1, 0.2, 0.3],
        metadata={"category": "AI", "page": 1},
        document="Vector databases are useful for semantic search.",
        created_at=created_at,
        updated_at=updated_at,
    )

    serialized = serialize_record(record)
    restored = deserialize_record(serialized)

    assert restored.id == record.id
    assert restored.dimension == record.dimension
    assert restored.vector.tolist() == pytest.approx(record.vector.tolist())
    assert restored.metadata == record.metadata
    assert restored.document == record.document
    assert restored.created_at == record.created_at
    assert restored.updated_at == record.updated_at


def test_serialized_collection_uses_metric_string() -> None:
    collection = CollectionConfig(
        name="test",
        dimension=3,
        distance_metric=DistanceMetric.EUCLIDEAN,
    )

    serialized = serialize_collection(collection)

    assert serialized["distance_metric"] == "euclidean"


def test_serialized_record_uses_json_compatible_values() -> None:
    record = VectorRecord(
        id="doc-1",
        vector=[1.0, 2.0],
    )

    serialized = serialize_record(record)

    assert isinstance(serialized["vector"], list)
    assert isinstance(serialized["created_at"], str)
    assert isinstance(serialized["updated_at"], str)


def test_deserialize_collection_rejects_invalid_data() -> None:
    with pytest.raises(ValidationError):
        deserialize_collection(
            {
                "name": "test",
                "dimension": 3,
                "distance_metric": "invalid",
            }
        )


def test_deserialize_record_rejects_invalid_vector() -> None:
    with pytest.raises(ValidationError):
        deserialize_record(
            {
                "id": "doc-1",
                "vector": [1.0, float("nan")],
                "metadata": {},
                "document": None,
                "created_at": "2026-08-18T12:00:00+00:00",
                "updated_at": "2026-08-18T12:00:00+00:00",
            }
        )