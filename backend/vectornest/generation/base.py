"""Generation provider abstractions for VectorNest."""

from abc import ABC, abstractmethod
from collections.abc import Iterator


class GenerationProvider(ABC):
    """Define the interface for text generation providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a complete response for a prompt."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream response chunks for a prompt."""
        raise NotImplementedError