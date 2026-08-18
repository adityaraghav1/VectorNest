"""In-memory storage engine for VectorNest."""

from vectornest.core.exceptions import (
    CollectionNotFoundError,
    DuplicateRecordError,
    RecordNotFoundError,
)
from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord


class InMemoryStorage:
    """Store collections and vector records in memory."""

    def __init__(self) -> None:
        self._collections: dict[str, CollectionConfig] = {}
        self._records: dict[str, dict[str, VectorRecord]] = {}

    def create_collection(self, collection: CollectionConfig) -> None:
        """Create a new empty collection."""
        if collection.name in self._collections:
            raise ValueError(
                f"Collection '{collection.name}' already exists."
            )

        self._collections[collection.name] = collection
        self._records[collection.name] = {}

    def get_collection(self, name: str) -> CollectionConfig:
        """Return a collection configuration."""
        if name not in self._collections:
            raise CollectionNotFoundError(
                f"Collection '{name}' does not exist."
            )

        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        """Delete a collection and all of its records."""
        self.get_collection(name)

        del self._collections[name]
        del self._records[name]

    def insert_record(
        self,
        collection_name: str,
        record: VectorRecord,
    ) -> None:
        """Insert a record into a collection."""
        collection = self.get_collection(collection_name)

        record.ensure_dimension(collection.dimension)

        records = self._records[collection_name]

        if record.id in records:
            raise DuplicateRecordError(
                f"Record '{record.id}' already exists."
            )

        records[record.id] = record

    def get_record(
        self,
        collection_name: str,
        record_id: str,
    ) -> VectorRecord:
        """Return a record by ID."""
        self.get_collection(collection_name)

        records = self._records[collection_name]

        if record_id not in records:
            raise RecordNotFoundError(
                f"Record '{record_id}' does not exist "
                f"in collection '{collection_name}'."
            )

        return records[record_id]

    def update_record(
        self,
        collection_name: str,
        record: VectorRecord,
    ) -> None:
        """Replace an existing record."""
        collection = self.get_collection(collection_name)

        record.ensure_dimension(collection.dimension)

        records = self._records[collection_name]

        if record.id not in records:
            raise RecordNotFoundError(
                f"Record '{record.id}' does not exist "
                f"in collection '{collection_name}'."
            )

        records[record.id] = record

    def delete_record(
        self,
        collection_name: str,
        record_id: str,
    ) -> None:
        """Delete a record from a collection."""
        self.get_collection(collection_name)

        records = self._records[collection_name]

        if record_id not in records:
            raise RecordNotFoundError(
                f"Record '{record_id}' does not exist "
                f"in collection '{collection_name}'."
            )

        del records[record_id]

    def list_records(
        self,
        collection_name: str,
    ) -> list[VectorRecord]:
        """Return records in insertion order."""
        self.get_collection(collection_name)

        return list(self._records[collection_name].values())

    def count(self, collection_name: str) -> int:
        """Return the number of records in a collection."""
        self.get_collection(collection_name)

        return len(self._records[collection_name])