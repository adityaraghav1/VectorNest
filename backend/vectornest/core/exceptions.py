"""Domain-specific exceptions raised by VectorNest."""


class VectorNestError(Exception):
    """Base class for all expected VectorNest domain errors."""


class ValidationError(VectorNestError):
    """Raised when data violates a VectorNest domain invariant."""


class DimensionMismatchError(ValidationError):
    """Raised when a vector dimension does not match a collection."""


class ZeroVectorError(ValidationError):
    """Raised when cosine similarity receives a vector with zero magnitude."""


class CollectionNotFoundError(VectorNestError):
    """Raised when a requested collection does not exist."""


class DuplicateRecordError(VectorNestError):
    """Raised when a record ID already exists in a collection."""


class RecordNotFoundError(VectorNestError):
    """Raised when a requested record does not exist."""