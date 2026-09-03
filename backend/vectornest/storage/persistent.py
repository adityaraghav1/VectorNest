"""File-backed persistent storage for VectorNest."""

import json
import shutil
from pathlib import Path
from typing import Any

from vectornest.core.exceptions import (
    CollectionNotFoundError,
    DuplicateCollectionError,
    DuplicateRecordError,
    RecordNotFoundError,
    ValidationError,
)
from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord
from vectornest.storage.serialization import (
    deserialize_collection,
    deserialize_record,
    serialize_collection,
    serialize_record,
)


class PersistentStorage:
    """Persist VectorNest collections and records as JSON files."""

    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)
        self.collections_path = self.root_path / "collections"

        try:
            self.collections_path.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as error:
            raise ValidationError(
                f"Unable to initialize storage at '{self.root_path}'."
            ) from error

    def create_collection(
        self,
        collection: CollectionConfig,
    ) -> None:
        """Create and persist an empty collection."""
        collection_path = self._collection_path(collection.name)

        if collection_path.exists():
            raise DuplicateCollectionError(
        f"Collection '{collection.name}' already exists."
    )

        try:
            collection_path.mkdir(parents=True)
        except OSError as error:
            raise ValidationError(
                f"Unable to create collection '{collection.name}'."
            ) from error

        self._write_json(
            collection_path / "collection.json",
            serialize_collection(collection),
        )

        self._write_json(
            collection_path / "records.json",
            {},
        )

    def get_collection(
        self,
        name: str,
    ) -> CollectionConfig:
        """Load a collection configuration from disk."""
        collection_file = (
            self._collection_path(name)
            / "collection.json"
        )

        if not collection_file.exists():
            raise CollectionNotFoundError(
                f"Collection '{name}' does not exist."
            )

        data = self._read_json(collection_file)

        if not isinstance(data, dict):
            raise ValidationError(
                f"Collection '{name}' contains invalid persisted data."
            )

        return deserialize_collection(data)

    def delete_collection(
        self,
        name: str,
    ) -> None:
        """Delete a collection and every record inside it."""
        collection_path = self._collection_path(name)

        if not collection_path.exists():
            raise CollectionNotFoundError(
                f"Collection '{name}' does not exist."
            )

        try:
            shutil.rmtree(collection_path)
        except OSError as error:
            raise ValidationError(
                f"Unable to delete collection '{name}'."
            ) from error

    def insert_record(
        self,
        collection_name: str,
        record: VectorRecord,
    ) -> None:
        """Insert and persist one vector record."""
        collection = self.get_collection(collection_name)

        record.ensure_dimension(collection.dimension)

        records = self._load_record_data(collection_name)

        if record.id in records:
            raise DuplicateRecordError(
                f"Record '{record.id}' already exists "
                f"in collection '{collection_name}'."
            )

        records[record.id] = serialize_record(record)

        self._save_record_data(
            collection_name,
            records,
        )

    def get_record(
        self,
        collection_name: str,
        record_id: str,
    ) -> VectorRecord:
        """Load one record by ID."""
        self.get_collection(collection_name)

        records = self._load_record_data(collection_name)

        if record_id not in records:
            raise RecordNotFoundError(
                f"Record '{record_id}' does not exist "
                f"in collection '{collection_name}'."
            )

        data = records[record_id]

        if not isinstance(data, dict):
            raise ValidationError(
                f"Record '{record_id}' contains invalid persisted data."
            )

        return deserialize_record(data)

    def update_record(
        self,
        collection_name: str,
        record: VectorRecord,
    ) -> None:
        """Replace an existing persisted record."""
        collection = self.get_collection(collection_name)

        record.ensure_dimension(collection.dimension)

        records = self._load_record_data(collection_name)

        if record.id not in records:
            raise RecordNotFoundError(
                f"Record '{record.id}' does not exist "
                f"in collection '{collection_name}'."
            )

        records[record.id] = serialize_record(record)

        self._save_record_data(
            collection_name,
            records,
        )

    def delete_record(
        self,
        collection_name: str,
        record_id: str,
    ) -> None:
        """Delete one persisted record."""
        self.get_collection(collection_name)

        records = self._load_record_data(collection_name)

        if record_id not in records:
            raise RecordNotFoundError(
                f"Record '{record_id}' does not exist "
                f"in collection '{collection_name}'."
            )

        del records[record_id]

        self._save_record_data(
            collection_name,
            records,
        )

    def list_records(
        self,
        collection_name: str,
    ) -> list[VectorRecord]:
        """Load every record from a collection."""
        self.get_collection(collection_name)

        records = self._load_record_data(collection_name)

        result: list[VectorRecord] = []

        for data in records.values():
            if not isinstance(data, dict):
                raise ValidationError(
                    "Persisted record data must be a dictionary."
                )

            result.append(
                deserialize_record(data)
            )

        return result

    def count(
        self,
        collection_name: str,
    ) -> int:
        """Return the number of persisted records."""
        self.get_collection(collection_name)

        return len(
            self._load_record_data(collection_name)
        )

    def _collection_path(
        self,
        name: str,
    ) -> Path:
        return self.collections_path / name

    def _records_path(
        self,
        collection_name: str,
    ) -> Path:
        return (
            self._collection_path(collection_name)
            / "records.json"
        )

    def _load_record_data(
        self,
        collection_name: str,
    ) -> dict[str, Any]:
        records_file = self._records_path(collection_name)

        if not records_file.exists():
            raise CollectionNotFoundError(
                f"Collection '{collection_name}' does not exist."
            )

        data = self._read_json(records_file)

        if not isinstance(data, dict):
            raise ValidationError(
                "Persisted records must be stored as a dictionary."
            )

        return data

    def _save_record_data(
        self,
        collection_name: str,
        records: dict[str, Any],
    ) -> None:
        self._write_json(
            self._records_path(collection_name),
            records,
        )

    @staticmethod
    def _read_json(
        path: Path,
    ) -> Any:
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)
        except (
            OSError,
            json.JSONDecodeError,
        ) as error:
            raise ValidationError(
                f"Unable to read persisted data from '{path}'."
            ) from error

    @staticmethod
    def _write_json(
        path: Path,
        data: Any,
    ) -> None:
        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    data,
                    file,
                    indent=2,
                )

            temporary_path.replace(path)

        except OSError as error:
            temporary_path.unlink(
                missing_ok=True,
            )

            raise ValidationError(
                f"Unable to persist data to '{path}'."
            ) from error