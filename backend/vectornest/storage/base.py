"""Common storage contract used by VectorNest services."""

from typing import Protocol

from vectornest.models.collection import CollectionConfig
from vectornest.models.record import VectorRecord


class StorageBackend(Protocol):
    """Define the storage operations required by VectorNest services."""

    def create_collection(self, collection: CollectionConfig) -> None:
        """Create an empty collection."""

    def get_collection(self, name: str) -> CollectionConfig:
        """Return one collection configuration."""

    def delete_collection(self, name: str) -> None:
        """Delete a collection and all of its records."""

    def insert_record(
        self,
        collection_name: str,
        record: VectorRecord,
    ) -> None:
        """Insert one record."""

    def get_record(
        self,
        collection_name: str,
        record_id: str,
    ) -> VectorRecord:
        """Return one record by ID."""

    def update_record(
        self,
        collection_name: str,
        record: VectorRecord,
    ) -> None:
        """Replace an existing record."""

    def delete_record(
        self,
        collection_name: str,
        record_id: str,
    ) -> None:
        """Delete one record."""

    def list_records(
        self,
        collection_name: str,
    ) -> list[VectorRecord]:
        """Return all records in a collection."""

    def count(self, collection_name: str) -> int:
        """Return the number of records in a collection."""