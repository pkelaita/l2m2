from collections import deque
from typing import Deque, Iterator, List, Dict
from enum import Enum

from l2m2.memory.base_memory import BaseMemory
from l2m2.exceptions import L2M2UsageError

CHAT_MEMORY_DEFAULT_WINDOW_SIZE = 40


class ChatMemoryEntry:
    """Represents a message in a conversation memory."""

    class Role(Enum):
        USER = "user"
        AGENT = "agent"

    def __init__(self, text: str, role: Role):
        self.text = text
        self.role = role


class ChatMemory(BaseMemory):
    """Represents a sliding-window conversation memory between a user and an agent. `ChatMemory` is
    the most basic type of memory and is designed to be passed directly to chat-based models such
    as `llama-3-70b-instruct`.
    """

    def __init__(self, window_size: int = CHAT_MEMORY_DEFAULT_WINDOW_SIZE) -> None:
        """Create a new ChatMemory object.

        Args:
            window_size (int, optional): The maximum number of messages to store.
                Defaults to DEFAULT_WINDOW_SIZE.

        Raises:
            L2M2UsageError: If `window_size` is less than or equal to 0.
        """
        super().__init__()

        if not window_size > 0:
            raise L2M2UsageError("window_size must be a positive integer.")

        self.window_size: int = window_size
        self.mem_window: Deque[ChatMemoryEntry] = deque(maxlen=window_size)

    def add_user_message(self, text: str) -> None:
        """Adds a user message to the memory.

        Args:
            text (str): The user message to add.
        """
        self.mem_window.append(ChatMemoryEntry(text, ChatMemoryEntry.Role.USER))

    def add_agent_message(self, text: str) -> None:
        """Adds an agent message to the memory.

        Args:
            text (str): The agent message to add.
        """
        self.mem_window.append(ChatMemoryEntry(text, ChatMemoryEntry.Role.AGENT))

    def unpack(
        self, role_key: str, message_key: str, user_key: str, agent_key: str
    ) -> List[Dict[str, str]]:
        """Gets a representation of the memory as a list of objects designed to
        be passed directly into LLM provider APIs as JSON.

        For example, with the following memory:
        ```
        memory = ChatMemory()
        memory.add_user_message("Hello")
        memory.add_agent_message("Hi!")
        memory.add_user_message("How are you?")
        memory.add_agent_message("I'm good, how are you?")
        ```
        `memory.unpack("role", "content", "user", "assistant")` would return:
        ```
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good, how are you?"}
        ]
        ```

        Args:
            role_key (str): The key to use to denote role. For example, "role".
            message_key (str): The key to use to denote the message. For example, "content".
            user_key (str): The key to denote the user's message. For example, "user".
            agent_key (str): The key to denote the agent or model's message. For example, "assistant".

        Returns:
            List[Dict[str, str]]: The representation of the memory as a list of objects.
        """
        res = []
        for message in self.mem_window:
            role_value = (
                user_key if message.role == ChatMemoryEntry.Role.USER else agent_key
            )
            res.append({role_key: role_value, message_key: message.text})
        return res

    def clear(self) -> None:
        """Clears the memory."""
        self.mem_window.clear()

    def __len__(self) -> int:
        return len(self.mem_window)

    def __iter__(self) -> Iterator[ChatMemoryEntry]:
        return iter(self.mem_window)
