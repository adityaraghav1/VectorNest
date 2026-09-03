import numpy as np

from vectornest.models.record import VectorRecord
from vectornest.services.projection import ProjectionService


def test_projection_returns_two_coordinates_per_record() -> None:
    records = [
        VectorRecord(
            id="a",
            vector=np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        ),
        VectorRecord(
            id="b",
            vector=np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            ),
        ),
        VectorRecord(
            id="c",
            vector=np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        ),
    ]

    service = ProjectionService()

    points, query = service.project_records(records)

    assert len(points) == 3
    assert query is None

    for record_id, x, y in points:
        assert record_id in {"a", "b", "c"}
        assert isinstance(x, float)
        assert isinstance(y, float)


def test_projection_includes_query_point() -> None:
    records = [
        VectorRecord(
            id="a",
            vector=np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
        ),
        VectorRecord(
            id="b",
            vector=np.array(
                [0.0, 1.0],
                dtype=np.float32,
            ),
        ),
    ]

    query_vector = np.array(
        [0.8, 0.2],
        dtype=np.float32,
    )

    service = ProjectionService()

    points, query = service.project_records(
        records,
        query_vector,
    )

    assert len(points) == 2
    assert query is not None
    assert isinstance(query[0], float)
    assert isinstance(query[1], float)


def test_projection_handles_single_record() -> None:
    records = [
        VectorRecord(
            id="only",
            vector=np.array(
                [1.0, 2.0, 3.0],
                dtype=np.float32,
            ),
        )
    ]

    service = ProjectionService()

    points, query = service.project_records(records)

    assert len(points) == 1
    assert query is None