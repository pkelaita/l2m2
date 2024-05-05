from collections import deque
from typing import Deque, Iterator, List, Dict

DEFAULT_WINDOW_SIZE = 40


class MessagePair:
    def __init__(self, user: str, agent: str):
        self.user = user
        self.agent = agent


class ChatMemory:
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self.window_size: int = window_size
        self.memory: Deque[MessagePair] = deque(maxlen=window_size)

    def add(self, user: str, agent: str) -> None:
        self.memory.append(MessagePair(user, agent))

    def get(self, index: int) -> MessagePair:
        return self.memory[index]

    def unpack(
        self, role_key: str, message_key: str, user_key: str, agent_key: str
    ) -> List[Dict[str, str]]:
        res = []
        for pair in self.memory:
            res.append({role_key: user_key, message_key: pair.user})
            res.append({role_key: agent_key, message_key: pair.agent})
        return res

    def __len__(self) -> int:
        return len(self.memory)

    def __iter__(self) -> Iterator[MessagePair]:
        return iter(self.memory)
