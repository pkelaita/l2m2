from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """Abstract representation of a model's memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clears the model's memory."""
        pass  # pragma: no cover
