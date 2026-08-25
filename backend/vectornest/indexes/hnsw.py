"""Hierarchical Navigable Small World index for VectorNest."""

import heapq
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from vectornest.core.exceptions import (
    RecordNotFoundError,
    ValidationError,
)
from vectornest.core.types import DistanceMetric
from vectornest.indexes.base import VectorIndex
from vectornest.indexes.results import SearchResult
from vectornest.metrics.functions import (
    VectorInput,
    calculate_metric,
)
from vectornest.models.record import VectorRecord


@dataclass(slots=True)
class _HNSWNode:
    """Represent one vector and its neighbours across graph layers."""

    record: VectorRecord
    level: int
    neighbours: dict[int, set[str]] = field(
        default_factory=dict
    )

    def neighbours_at(self, level: int) -> set[str]:
        """Return the neighbour set for one graph level."""
        return self.neighbours.setdefault(
            level,
            set(),
        )


class HNSWIndex(VectorIndex):
    """Approximate nearest-neighbour index using hierarchical graphs."""

    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        max_connections: int = 8,
        ef_search: int = 32,
        level_probability: float = 0.5,
        random_seed: int | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValidationError(
                "Index dimension must be greater than zero."
            )

        if max_connections <= 0:
            raise ValidationError(
                "max_connections must be greater than zero."
            )

        if ef_search <= 0:
            raise ValidationError(
                "ef_search must be greater than zero."
            )

        if not 0.0 < level_probability < 1.0:
            raise ValidationError(
                "level_probability must be between zero and one."
            )

        self.dimension = dimension
        self.metric = metric
        self.max_connections = max_connections
        self.ef_search = ef_search
        self.level_probability = level_probability

        self._nodes: dict[str, _HNSWNode] = {}
        self._entry_point: str | None = None
        self._max_level = -1
        self._random = random.Random(random_seed)

    def add(self, record: VectorRecord) -> None:
        """Add a record to the hierarchical graph."""
        record.ensure_dimension(self.dimension)

        if record.id in self._nodes:
            self.remove(record.id)

        level = self._random_level()

        node = _HNSWNode(
            record=record,
            level=level,
        )

        for graph_level in range(level + 1):
            node.neighbours_at(graph_level)

        if not self._nodes:
            self._nodes[record.id] = node
            self._entry_point = record.id
            self._max_level = level
            return

        old_entry = self._entry_point
        old_max_level = self._max_level

        self._nodes[record.id] = node

        for graph_level in range(
            min(level, old_max_level) + 1
        ):
            neighbours = self._select_neighbours(
                record=record,
                level=graph_level,
                limit=self.max_connections,
            )

            for neighbour_id in neighbours:
                node.neighbours_at(
                    graph_level
                ).add(neighbour_id)

                self._nodes[
                    neighbour_id
                ].neighbours_at(
                    graph_level
                ).add(record.id)

            self._trim_connections(
                record.id,
                graph_level,
            )

            for neighbour_id in list(
                node.neighbours_at(graph_level)
            ):
                self._trim_connections(
                    neighbour_id,
                    graph_level,
                )

        if level > old_max_level:
            self._entry_point = record.id
            self._max_level = level
        else:
            self._entry_point = old_entry

    def add_many(
        self,
        records: Iterable[VectorRecord],
    ) -> None:
        """Add multiple records to the index."""
        for record in records:
            self.add(record)

    def remove(self, record_id: str) -> None:
        """Remove a record from every graph layer."""
        if record_id not in self._nodes:
            raise RecordNotFoundError(
                f"Record '{record_id}' does not exist in the index."
            )

        node = self._nodes[record_id]

        for graph_level, neighbours in node.neighbours.items():
            for neighbour_id in list(neighbours):
                if neighbour_id not in self._nodes:
                    continue

                self._nodes[
                    neighbour_id
                ].neighbours_at(
                    graph_level
                ).discard(record_id)

        del self._nodes[record_id]

        if not self._nodes:
            self._entry_point = None
            self._max_level = -1
            return

        if self._entry_point == record_id:
            self._recalculate_entry_point()

    def get(self, record_id: str) -> VectorRecord:
        """Return a record by ID."""
        try:
            return self._nodes[record_id].record
        except KeyError as error:
            raise RecordNotFoundError(
                f"Record '{record_id}' does not exist in the index."
            ) from error

    def search(
        self,
        query: VectorInput,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search upper layers greedily, then explore layer zero."""
        if limit <= 0:
            raise ValidationError(
                "Search limit must be greater than zero."
            )

        if self._entry_point is None:
            return []

        entry_id = self._entry_point

        for level in range(
            self._max_level,
            0,
            -1,
        ):
            entry_id = self._greedy_search_layer(
                query=query,
                entry_id=entry_id,
                level=level,
            )

        effective_ef = max(
            self.ef_search,
            limit,
        )

        results = self._search_layer(
            query=query,
            entry_id=entry_id,
            level=0,
            ef=effective_ef,
        )

        results.sort(
            key=lambda result: result.score,
            reverse=self._is_similarity_metric(),
        )

        return results[:limit]

    def count(self) -> int:
        """Return the number of indexed records."""
        return len(self._nodes)

    def clear(self) -> None:
        """Remove all nodes and reset the hierarchy."""
        self._nodes.clear()
        self._entry_point = None
        self._max_level = -1

    def _random_level(self) -> int:
        """Generate a random graph level using geometric promotion."""
        level = 0

        while self._random.random() < self.level_probability:
            level += 1

        return level

    def _select_neighbours(
        self,
        record: VectorRecord,
        level: int,
        limit: int,
    ) -> list[str]:
        """Select nearby existing nodes available at a graph level."""
        scored: list[tuple[str, float]] = []

        for node_id, node in self._nodes.items():
            if node_id == record.id:
                continue

            if node.level < level:
                continue

            score = calculate_metric(
                record.vector,
                node.record.vector,
                self.metric,
            )

            scored.append(
                (
                    node_id,
                    score,
                )
            )

        scored.sort(
            key=lambda item: item[1],
            reverse=self._is_similarity_metric(),
        )

        return [
            node_id
            for node_id, _ in scored[:limit]
        ]

    def _trim_connections(
        self,
        node_id: str,
        level: int,
    ) -> None:
        """Keep only the strongest neighbours at one graph layer."""
        node = self._nodes[node_id]

        neighbours = node.neighbours_at(level)

        if len(neighbours) <= self.max_connections:
            return

        scored = [
            (
                neighbour_id,
                calculate_metric(
                    node.record.vector,
                    self._nodes[
                        neighbour_id
                    ].record.vector,
                    self.metric,
                ),
            )
            for neighbour_id in neighbours
        ]

        scored.sort(
            key=lambda item: item[1],
            reverse=self._is_similarity_metric(),
        )

        keep = {
            neighbour_id
            for neighbour_id, _ in scored[
                : self.max_connections
            ]
        }

        removed = neighbours - keep
        node.neighbours[level] = keep

        for neighbour_id in removed:
            self._nodes[
                neighbour_id
            ].neighbours_at(
                level
            ).discard(node_id)

    def _greedy_search_layer(
        self,
        query: VectorInput,
        entry_id: str,
        level: int,
    ) -> str:
        """Move greedily toward a better node within one layer."""
        current_id = entry_id

        current_score = calculate_metric(
            query,
            self._nodes[
                current_id
            ].record.vector,
            self.metric,
        )

        improved = True

        while improved:
            improved = False

            current_node = self._nodes[
                current_id
            ]

            for neighbour_id in current_node.neighbours.get(
                level,
                set(),
            ):
                score = calculate_metric(
                    query,
                    self._nodes[
                        neighbour_id
                    ].record.vector,
                    self.metric,
                )

                if self._is_better(
                    score,
                    current_score,
                ):
                    current_id = neighbour_id
                    current_score = score
                    improved = True
                    break

        return current_id

    def _search_layer(
        self,
        query: VectorInput,
        entry_id: str,
        level: int,
        ef: int,
    ) -> list[SearchResult]:
        """Explore promising candidates within one graph layer."""
        entry_score = calculate_metric(
            query,
            self._nodes[
                entry_id
            ].record.vector,
            self.metric,
        )

        entry_cost = self._score_to_cost(
            entry_score
        )

        candidate_heap: list[
            tuple[float, str, float]
        ] = [
            (
                entry_cost,
                entry_id,
                entry_score,
            )
        ]

        best_heap: list[
            tuple[float, str, float]
        ] = [
            (
                -entry_cost,
                entry_id,
                entry_score,
            )
        ]

        visited = {
            entry_id,
        }

        while candidate_heap:
            (
                current_cost,
                current_id,
                _,
            ) = heapq.heappop(
                candidate_heap
            )

            worst_cost = -best_heap[0][0]

            if (
                len(best_heap) >= ef
                and current_cost > worst_cost
            ):
                break

            current_node = self._nodes[
                current_id
            ]

            for neighbour_id in current_node.neighbours.get(
                level,
                set(),
            ):
                if neighbour_id in visited:
                    continue

                visited.add(neighbour_id)

                score = calculate_metric(
                    query,
                    self._nodes[
                        neighbour_id
                    ].record.vector,
                    self.metric,
                )

                cost = self._score_to_cost(
                    score
                )

                worst_cost = -best_heap[0][0]

                if (
                    len(best_heap) < ef
                    or cost < worst_cost
                ):
                    heapq.heappush(
                        candidate_heap,
                        (
                            cost,
                            neighbour_id,
                            score,
                        ),
                    )

                    heapq.heappush(
                        best_heap,
                        (
                            -cost,
                            neighbour_id,
                            score,
                        ),
                    )

                    if len(best_heap) > ef:
                        heapq.heappop(
                            best_heap
                        )

        return [
            SearchResult(
                record=self._nodes[
                    node_id
                ].record,
                score=score,
            )
            for _, node_id, score in best_heap
        ]

    def _recalculate_entry_point(self) -> None:
        """Choose a node from the highest remaining graph level."""
        entry_id, entry_node = max(
            self._nodes.items(),
            key=lambda item: item[1].level,
        )

        self._entry_point = entry_id
        self._max_level = entry_node.level

    def _score_to_cost(
        self,
        score: float,
    ) -> float:
        """Convert metric scores into lower-is-better costs."""
        if self._is_similarity_metric():
            return -score

        return score

    def _is_better(
        self,
        candidate: float,
        current: float,
    ) -> bool:
        """Return whether a candidate score is better."""
        if self._is_similarity_metric():
            return candidate > current

        return candidate < current

    def _is_similarity_metric(self) -> bool:
        """Return whether larger metric scores are better."""
        return self.metric in {
            DistanceMetric.COSINE,
            DistanceMetric.DOT_PRODUCT,
        }