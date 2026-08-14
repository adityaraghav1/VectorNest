"""Collection configuration and its invariants."""

from dataclasses import dataclass

from vectornest.core.exceptions import ValidationError
from vectornest.core.types import DistanceMetric

MAX_COLLECTION_NAME_LENGTH = 128


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """Immutable schema settings for a homogeneous vector collection."""

    name: str
    dimension: int
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate collection settings immediately after construction."""
        if not isinstance(self.name, str):
            raise ValidationError("Collection name must be a string.")
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValidationError("Collection name cannot be empty.")
        if len(normalized_name) > MAX_COLLECTION_NAME_LENGTH:
            raise ValidationError(
                f"Collection name cannot exceed {MAX_COLLECTION_NAME_LENGTH} characters."
            )
        if not normalized_name.replace("_", "").replace("-", "").isalnum():
            raise ValidationError(
                "Collection name may contain only letters, digits, hyphens, and underscores."
            )
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise ValidationError("Collection dimension must be an integer.")
        if self.dimension <= 0:
            raise ValidationError("Collection dimension must be greater than zero.")
        if not isinstance(self.distance_metric, DistanceMetric):
            raise ValidationError("distance_metric must be a DistanceMetric value.")

        object.__setattr__(self, "name", normalized_name)
        if self.description is not None and not isinstance(self.description, str):
            raise ValidationError("Collection description must be a string or None.")
        if self.description is not None:
            object.__setattr__(self, "description", self.description.strip() or None)
