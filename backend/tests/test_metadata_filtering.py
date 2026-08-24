import pytest

from vectornest.core.exceptions import ValidationError
from vectornest.models.record import VectorRecord
from vectornest.query.filters import MetadataFilter, filter_records


def make_record(
    record_id: str,
    metadata: dict,
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        vector=[1.0, 0.0],
        metadata=metadata,
    )


def test_filter_matches_single_condition() -> None:
    records = [
        make_record(
            "r1",
            {"source": "resume.pdf"},
        ),
        make_record(
            "r2",
            {"source": "notes.pdf"},
        ),
    ]

    metadata_filter = MetadataFilter(
        {
            "source": "resume.pdf",
        }
    )

    filtered = filter_records(
        records,
        metadata_filter,
    )

    assert [
        record.id
        for record in filtered
    ] == ["r1"]


def test_filter_requires_all_conditions() -> None:
    records = [
        make_record(
            "r1",
            {
                "source": "resume.pdf",
                "category": "AI",
            },
        ),
        make_record(
            "r2",
            {
                "source": "resume.pdf",
                "category": "backend",
            },
        ),
    ]

    metadata_filter = MetadataFilter(
        {
            "source": "resume.pdf",
            "category": "AI",
        }
    )

    filtered = filter_records(
        records,
        metadata_filter,
    )

    assert [
        record.id
        for record in filtered
    ] == ["r1"]


def test_missing_metadata_key_does_not_match() -> None:
    record = make_record(
        "r1",
        {
            "source": "resume.pdf",
        },
    )

    metadata_filter = MetadataFilter(
        {
            "category": "AI",
        }
    )

    assert not metadata_filter.matches(record)


def test_none_filter_returns_all_records() -> None:
    records = [
        make_record("r1", {}),
        make_record("r2", {}),
    ]

    assert filter_records(
        records,
        None,
    ) == records


def test_empty_filter_matches_all_records() -> None:
    records = [
        make_record("r1", {}),
        make_record("r2", {}),
    ]

    filtered = filter_records(
        records,
        MetadataFilter({}),
    )

    assert filtered == records


def test_filter_rejects_empty_key() -> None:
    with pytest.raises(ValidationError):
        MetadataFilter(
            {
                "": "value",
            }
        )