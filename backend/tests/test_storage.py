import pytest

from vectornest.core.exceptions import (
    CollectionNotFoundError,
    DimensionMismatchError,
    DuplicateRecordError,
    RecordNotFoundError,
)
from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord
from vectornest.storage.engine import InMemoryStorage


def make_storage() -> InMemoryStorage:
    storage = InMemoryStorage()

    storage.create_collection(
        CollectionConfig(
            name="documents",
            dimension=3,
        )
    )

    return storage


def make_record(
    record_id: str,
    vector: list[float] | None = None,
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        vector=vector or [1.0, 2.0, 3.0],
        metadata={"source": "test"},
    )


def test_create_and_get_collection() -> None:
    storage = make_storage()

    config = storage.get_collection("documents")

    assert config.name == "documents"
    assert config.dimension == 3


def test_insert_get_and_count_record() -> None:
    storage = make_storage()
    record = make_record("r1")

    storage.insert_record("documents", record)

    assert storage.get_record("documents", "r1") is record
    assert storage.count("documents") == 1


def test_insert_rejects_duplicate_id() -> None:
    storage = make_storage()

    storage.insert_record(
        "documents",
        make_record("r1"),
    )

    with pytest.raises(DuplicateRecordError):
        storage.insert_record(
            "documents",
            make_record("r1"),
        )


def test_insert_rejects_dimension_mismatch() -> None:
    storage = make_storage()

    with pytest.raises(DimensionMismatchError):
        storage.insert_record(
            "documents",
            make_record("r1", [1.0, 2.0]),
        )


def test_update_replaces_existing_record() -> None:
    storage = make_storage()

    storage.insert_record(
        "documents",
        make_record("r1"),
    )

    updated = make_record(
        "r1",
        [4.0, 5.0, 6.0],
    )

    storage.update_record(
        "documents",
        updated,
    )

    assert storage.get_record(
        "documents",
        "r1",
    ) is updated


def test_update_requires_existing_record() -> None:
    storage = make_storage()

    with pytest.raises(RecordNotFoundError):
        storage.update_record(
            "documents",
            make_record("missing"),
        )


def test_delete_record() -> None:
    storage = make_storage()

    storage.insert_record(
        "documents",
        make_record("r1"),
    )

    storage.delete_record(
        "documents",
        "r1",
    )

    assert storage.count("documents") == 0

    with pytest.raises(RecordNotFoundError):
        storage.get_record(
            "documents",
            "r1",
        )


def test_list_records_preserves_insertion_order() -> None:
    storage = make_storage()

    storage.insert_record(
        "documents",
        make_record("r1"),
    )

    storage.insert_record(
        "documents",
        make_record("r2"),
    )

    assert [
        record.id
        for record in storage.list_records("documents")
    ] == ["r1", "r2"]


def test_missing_collection_is_rejected() -> None:
    storage = InMemoryStorage()

    with pytest.raises(CollectionNotFoundError):
        storage.get_collection("documents")


def test_delete_collection_removes_records() -> None:
    storage = make_storage()

    storage.insert_record(
        "documents",
        make_record("r1"),
    )

    storage.delete_collection("documents")

    with pytest.raises(CollectionNotFoundError):
        storage.get_collection("documents")