"""Simple 2D vector projection utilities."""

import numpy as np

from vectornest.core.exceptions import ValidationError
from vectornest.models.record import VectorRecord


class ProjectionService:
    def project_records(
        self,
        records: list[VectorRecord],
        query_vector: np.ndarray | None = None,
    ) -> tuple[list[tuple[str, float, float]], tuple[float, float] | None]:
        if not records:
            return [], None

        vectors = [record.vector for record in records]

        if query_vector is not None:
            vectors.append(query_vector)

        matrix = np.vstack(vectors).astype(np.float32)

        if matrix.ndim != 2:
            raise ValidationError("Projection input must be a 2D matrix.")

        centered = matrix - np.mean(matrix, axis=0)

        _, _, vh = np.linalg.svd(
            centered,
            full_matrices=False,
        )

        component_count = min(2, vh.shape[0])

        components = vh[:component_count]

        projected = centered @ components.T

        if component_count == 1:
            projected = np.column_stack(
                (
                    projected[:, 0],
                    np.zeros(projected.shape[0]),
                )
            )

        record_points = [
            (
                record.id,
                float(projected[index, 0]),
                float(projected[index, 1]),
            )
            for index, record in enumerate(records)
        ]

        query_point = None

        if query_vector is not None:
            query_point = (
                float(projected[-1, 0]),
                float(projected[-1, 1]),
            )

        return record_points, query_point