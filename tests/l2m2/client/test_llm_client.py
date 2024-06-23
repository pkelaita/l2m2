import pytest
from unittest.mock import patch

from l2m2.client import LLMClient


@pytest.fixture
def llm_client():
    return LLMClient()


@patch("l2m2.client.llm_client.asyncio.run", return_value="async_call_result")
def test_call(mock_asyncio_run, llm_client):
    result = llm_client.call(model="test-model", prompt="test prompt")

    mock_asyncio_run.assert_called_once()
    assert result == "async_call_result"


@patch("l2m2.client.llm_client.asyncio.run", return_value="async_call_custom_result")
def test_call_custom(mock_asyncio_run, llm_client):
    result = llm_client.call_custom(
        provider="test-provider", model_id="test-model-id", prompt="test prompt"
    )

    mock_asyncio_run.assert_called_once()
    assert result == "async_call_custom_result"


@pytest.mark.asyncio
async def test_sync_fn_wrapper():
    async def dummy_fn(*args, **kwargs):
        return "dummy_result"

    client = LLMClient()
    original_client = client.httpx_client

    with patch("httpx.AsyncClient", autospec=True) as MockAsyncClient:
        mock_temp_client = MockAsyncClient.return_value
        mock_temp_client.__aenter__.return_value = mock_temp_client

        result = await client._sync_fn_wrapper(dummy_fn)

        MockAsyncClient.assert_called_once()
        assert result == "dummy_result"
        assert client.httpx_client == original_client
