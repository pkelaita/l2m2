import pytest
from unittest.mock import patch

from l2m2.client import LLMClient
from l2m2.exceptions import L2M2UsageError


@pytest.fixture
def llm_client():
    return LLMClient()


@patch("l2m2.client.llm_client.asyncio.run", return_value="async_call_result")
def test_call(mock_asyncio_run, llm_client):
    result = llm_client.call(model="test-model", prompt="test prompt")

    mock_asyncio_run.assert_called_once()
    assert result == "async_call_result"


@pytest.mark.asyncio
async def test_sync_fn_wrapper(llm_client):
    async def dummy_fn(*args, **kwargs):
        return "dummy_result"

    original_client = llm_client.httpx_client

    with patch("httpx.AsyncClient", autospec=True) as MockAsyncClient:
        mock_temp_client = MockAsyncClient.return_value
        mock_temp_client.__aenter__.return_value = mock_temp_client

        result = await llm_client._sync_fn_wrapper(dummy_fn)

        MockAsyncClient.assert_called_once()
        assert result == "dummy_result"
        assert llm_client.httpx_client == original_client


@pytest.mark.asyncio
async def test_async_instantiation_raises_usage_error():
    with pytest.raises(L2M2UsageError):
        LLMClient()


@pytest.mark.asyncio
async def test_call_raises_usage_error(llm_client):
    with pytest.raises(L2M2UsageError):
        await llm_client.call(model="gpt-4o", prompt="foo")
