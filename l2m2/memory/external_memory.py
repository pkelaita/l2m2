from enum import Enum

from l2m2.memory.base_memory import BaseMemory, MemoryType


class ExternalMemoryLoadingType(Enum):
    """Represents how the model should load external memory."""

    SYSTEM_PROMPT_APPEND = "system_prompt_append"
    USER_PROMPT_APPEND = "user_prompt_append"


class ExternalMemory(BaseMemory):
    """Represents custom memory that is managed completely externally to the model."""

    def __init__(
        self,
        contents: str = "",
        loading_type: ExternalMemoryLoadingType = ExternalMemoryLoadingType.SYSTEM_PROMPT_APPEND,
    ) -> None:
        """Create a new ExternalMemory object.

        Args:
            contents (str, optional): The memory to pre-load. Defaults to "".
            loading_type (LoadingType, optional): How the model should load the memory –
            either in the system prompt, inserted as a user prompt, or appended to the
            most recent user prompt. Defaults to LoadingType.SYSTEM_PROMPT.
        """

        super().__init__(MemoryType.EXTERNAL)
        self.contents: str = contents
        self.loading_type: ExternalMemoryLoadingType = loading_type

    def get_contents(self) -> str:
        """Get the contents of the memory object.

        Returns:
            str: The entire memory contents
        """

        return self.contents

    def set_contents(self, new_contents: str) -> None:
        """Load the contents into the memory object, replacing the existing contents.

        Args:
            new_contents (str): The new contents to load.
        """
        self.contents = new_contents

    def append_contents(self, new_contents: str) -> None:
        """Append new contents to the memory object.

        Args:
            new_contents (str): The new contents to append.
        """
        self.contents += new_contents

    def clear(self) -> None:
        """Clear the memory."""
        self.contents = ""
