import pytest
from l2m2.memory.chat_memory import ChatMemory, DEFAULT_WINDOW_SIZE, ChatMemoryEntry


def test_chat_memory():
    memory = ChatMemory()
    assert memory.window_size == DEFAULT_WINDOW_SIZE
    assert len(memory) == 0

    memory.add_user_message("A")
    memory.add_agent_message("B")
    it = iter(memory)
    e1 = next(it)
    e2 = next(it)
    assert len(memory) == 2
    assert e1.text == "A"
    assert e1.role == ChatMemoryEntry.Role.USER
    assert e2.text == "B"
    assert e2.role == ChatMemoryEntry.Role.AGENT

    memory.clear()
    assert len(memory) == 0


def test_unpack():
    memory = ChatMemory(window_size=10)
    memory.add_user_message("A")
    memory.add_agent_message("B")
    memory.add_user_message("C")
    memory.add_agent_message("D")
    memory.add_user_message("E")
    memory.add_agent_message("F")
    mem_arr = memory.unpack("role", "text", "user", "agent")
    assert len(mem_arr) == 6
    assert mem_arr[0] == {"role": "user", "text": "A"}
    assert mem_arr[1] == {"role": "agent", "text": "B"}
    assert mem_arr[2] == {"role": "user", "text": "C"}
    assert mem_arr[3] == {"role": "agent", "text": "D"}
    assert mem_arr[4] == {"role": "user", "text": "E"}
    assert mem_arr[5] == {"role": "agent", "text": "F"}


def test_sliding_window():
    memory = ChatMemory(window_size=3)
    memory.add_user_message("A")
    memory.add_agent_message("B")
    memory.add_user_message("C")
    memory.add_agent_message("D")
    memory.add_user_message("E")
    memory.add_agent_message("F")
    assert len(memory) == 3

    mem_arr = memory.unpack("role", "text", "user", "agent")
    assert len(mem_arr) == 3
    assert mem_arr[0] == {"role": "agent", "text": "D"}
    assert mem_arr[1] == {"role": "user", "text": "E"}
    assert mem_arr[2] == {"role": "agent", "text": "F"}

    memory.add_agent_message("G")

    mem_arr = memory.unpack("role", "text", "user", "agent")
    assert len(mem_arr) == 3
    assert mem_arr[0] == {"role": "user", "text": "E"}
    assert mem_arr[1] == {"role": "agent", "text": "F"}
    assert mem_arr[2] == {"role": "agent", "text": "G"}


def test_bad_window_size():
    with pytest.raises(ValueError):
        ChatMemory(window_size=-1)
