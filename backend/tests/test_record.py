from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from vectornest.core.exceptions import DimensionMismatchError, ValidationError
from vectornest.models.record import VectorRecord


def test_record_normalizes_vector_to_contiguous_float32() -> None:
    record = VectorRecord(id="lesson-1", vector=[1, 2, 3], metadata={"topic": "vectors"})

    assert record.vector.dtype == np.float32
    assert record.vector.flags.c_contiguous
    assert record.dimension == 3


@pytest.mark.parametrize("vector", [[], [[1.0, 2.0]], [1.0, float("nan")]])
def test_record_rejects_invalid_vectors(vector: object) -> None:
    with pytest.raises(ValidationError):
        VectorRecord(id="invalid", vector=vector)  


def test_record_rejects_non_scalar_metadata() -> None:
    with pytest.raises(ValidationError, match="scalar JSON-compatible"):
        VectorRecord(id="lesson-1", vector=[1.0], metadata={"tags": ["ml"]}) 


def test_record_rejects_duplicate_metadata_keys_after_normalization() -> None:
    with pytest.raises(ValidationError, match="duplicate key"):
        VectorRecord(id="lesson-1", vector=[1.0], metadata={"topic": "ml", " topic ": "ai"})


def test_record_checks_collection_dimension() -> None:
    record = VectorRecord(id="lesson-1", vector=[1.0, 2.0])

    with pytest.raises(DimensionMismatchError, match="expected 3"):
        record.ensure_dimension(3)


def test_record_rejects_reversed_timestamps() -> None:
    created = datetime.now(UTC)
    with pytest.raises(ValidationError, match="updated_at"):
        VectorRecord(
            id="lesson-1",
            vector=[1.0],
            created_at=created,
            updated_at=created - timedelta(seconds=1),
        )
