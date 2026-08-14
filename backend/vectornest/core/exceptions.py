"""Domain-specific exceptions raised by VectorNest."""


class VectorNestError(Exception):
    """Base class for all expected VectorNest domain errors."""


class ValidationError(VectorNestError):
    """Raised when data violates a VectorNest domain invariant."""


class DimensionMismatchError(ValidationError):
    """Raised when a vector does not match its collection's dimension."""
