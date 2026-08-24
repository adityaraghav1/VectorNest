import pytest

from vectornest.core.exceptions import (
    CollectionNotFoundError,
    DimensionMismatchError,
    DuplicateRecordError,
    RecordNotFoundError,
)
from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord
from vectornest.storage.persistent import PersistentStorage


def make_collection() -> CollectionConfig:
    return CollectionConfig(
        name="documents",
        dimension=3,
    )


def make_record(
    record_id: str,
    vector: list[float] | None = None,
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        vector=vector or [1.0, 2.0, 3.0],
        metadata={
            "source": "test",
        },
    )


def test_collection_survives_storage_restart(
    tmp_path,
) -> None:
    storage = PersistentStorage(tmp_path)

    storage.create_collection(
        make_collection()
    )

    restarted = PersistentStorage(tmp_path)

    collection = restarted.get_collection(
        "documents"
    )

    assert collection.name == "documents"
    assert collection.dimension == 3


def test_record_survives_storage_restart(
    tmp_path,
) -> None:
    storage = PersistentStorage(tmp_path)

    storage.create_collection(
        make_collection()
    )

    storage.insert_record(
        "documents",
        make_record("r1"),
    )

    restarted = PersistentStorage(tmp_path)

    record = restarted.get_record(
        "documents",
        "r1",
    )

    assert record.id == "r1"
    assert record.vector.tolist() == pytest.approx(
        [1.0, 2.0, 3.0]
    )


def test_persistent_storage_count(
    tmp_path,
) -> None:
    storage = PersistentStorage(tmp_path)

    storage.create_collection(
        make_collection()
    )

    storage.insert_record(
        "documents",
        make_record("r1"),
    )

    storage.insert_record(
        "documents",
        make_record("r2"),
    )

    assert storage.count("documents") == 2


def test_persistent_storage_rejects_duplicate_record(
    tmp_path,
) -> None:
    storage = PersistentStorage(tmp_path)

    storage.create_collection(
        make_collection()
    )

    storage.insert_record(
        "documents",
        make_record("r1"),
    )

    with pytest.raises(DuplicateRecordError):
        storage.insert_record(
            "documents",
            make_record("r1"),
        )


def test_persistent_storage_rejects_dimension_mismatch(
    tmp_path,
) -> None:
    storage = PersistentStorage(tmp_path)

    storage.create_collection(
        make_collection()
    )

    with pytest.raises(DimensionMismatchError):
        storage.insert_record(
            "documents",
            make_record(
                "r1",
                [1.0, 2.0],
            ),
        )


def test_persistent_storage_updates_record(
    tmp_path,
) -> None:
    storage = PersistentStorage(tmp_path)

    storage.create_collection(
        make_collection()
    )

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

    restored = storage.get_record(
        "documents",
        "r1",
    )

    assert restored.vector.tolist() == pytest.approx(
        [4.0, 5.0, 6.0]
    )


def test_persistent_storage_deletes_record(
    tmp_path,
) -> None:
    storage = PersistentStorage(tmp_path)

    storage.create_collection(
        make_collection()
    )

    storage.insert_record(
        "documents",
        make_record("r1"),
    )

    storage.delete_record(
        "documents",
        "r1",
    )

    with pytest.raises(RecordNotFoundError):
        storage.get_record(
            "documents",
            "r1",
        )


def test_persistent_storage_lists_records(
    tmp_path,
) -> None:
    storage = PersistentStorage(tmp_path)

    storage.create_collection(
        make_collection()
    )

    storage.insert_record(
        "documents",
        make_record("r1"),
    )

    storage.insert_record(
        "documents",
        make_record("r2"),
    )

    records = storage.list_records(
        "documents"
    )

    assert [
        record.id
        for record in records
    ] == ["r1", "r2"]


def test_delete_collection_removes_persisted_data(
    tmp_path,
) -> None:
    storage = PersistentStorage(tmp_path)

    storage.create_collection(
        make_collection()
    )

    storage.insert_record(
        "documents",
        make_record("r1"),
    )

    storage.delete_collection(
        "documents"
    )

    restarted = PersistentStorage(tmp_path)

    with pytest.raises(CollectionNotFoundError):
        restarted.get_collection(
            "documents"
        )