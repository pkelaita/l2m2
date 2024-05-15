from l2m2.memory.external_memory import ExternalMemory


def test_external_memory():
    memory = ExternalMemory()
    assert memory.get_contents() == ""

    memory.set_contents("A")
    assert memory.get_contents() == "A"

    memory.append_contents("B")
    assert memory.get_contents() == "AB"

    memory.clear()
    assert memory.get_contents() == ""
