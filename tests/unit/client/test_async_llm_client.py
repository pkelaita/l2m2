import pytest

from l2m2.client import AsyncLLMClient


@pytest.mark.asyncio
async def test_instantiation():
    async with AsyncLLMClient() as _:
        pass
