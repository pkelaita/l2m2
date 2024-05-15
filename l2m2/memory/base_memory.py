from enum import Enum
from abc import ABC, abstractmethod


class MemoryType(Enum):
    """The type of memory used by the model."""

    CHAT = "chat"
    EXTERNAL = "external"


class BaseMemory(ABC):
    """Abstract representation of a model's memory."""

    def __init__(self, memory_type: MemoryType) -> None:
        """Create a new BaseMemory object.

        Args:
            memory_type (MemoryType): The type of memory managed by the model.
        """
        self.memory_type = memory_type

    @abstractmethod
    def clear(self) -> None:
        """Clears the model's memory."""
        pass  # pragma: no cover
