"""Metadata filtering for VectorNest records."""

from collections.abc import Iterable
from dataclasses import dataclass

from vectornest.core.exceptions import ValidationError
from vectornest.models.record import MetadataValue, VectorRecord


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """Represent exact-match metadata conditions."""

    conditions: dict[str, MetadataValue]

    def __post_init__(self) -> None:
        if not isinstance(self.conditions, dict):
            raise ValidationError(
                "Metadata filter conditions must be a dictionary."
            )

        normalized: dict[str, MetadataValue] = {}

        for key, value in self.conditions.items():
            if not isinstance(key, str) or not key.strip():
                raise ValidationError(
                    "Metadata filter keys must be non-empty strings."
                )

            normalized_key = key.strip()

            if normalized_key in normalized:
                raise ValidationError(
                    "Metadata filter contains duplicate keys "
                    "after normalization."
                )

            normalized[normalized_key] = value

        object.__setattr__(
            self,
            "conditions",
            normalized,
        )

    def matches(self, record: VectorRecord) -> bool:
        """Return whether a record satisfies every condition."""
        return all(
            record.metadata.get(key) == expected_value
            for key, expected_value in self.conditions.items()
        )


def filter_records(
    records: Iterable[VectorRecord],
    metadata_filter: MetadataFilter | None,
) -> list[VectorRecord]:
    """Return records that satisfy the metadata filter."""
    if metadata_filter is None:
        return list(records)

    return [
        record
        for record in records
        if metadata_filter.matches(record)
    ]