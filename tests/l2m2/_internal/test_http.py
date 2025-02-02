from l2m2._internal.http import (
    _get_headers,
    _handle_replicate_201,
    llm_post,
)
import pytest
import httpx
from l2m2.exceptions import LLMTimeoutError, LLMRateLimitError
from l2m2.model_info import API_KEY, HOSTED_PROVIDERS


def test_get_headers():
    # Test header generation with API key replacement
    test_api_key = "test_key_123"

    # Mock HOSTED_PROVIDERS for testing
    original_HOSTED_PROVIDERS = HOSTED_PROVIDERS.copy()
    HOSTED_PROVIDERS["openai"] = {
        "headers": {"Authorization": f"Bearer {API_KEY}"},
        "endpoint": "https://api.openai.com/v1/chat/completions",
    }
    HOSTED_PROVIDERS["replicate"] = {
        "headers": {"Authorization": f"Token {API_KEY}"},
        "endpoint": "https://api.replicate.com/v1/predictions",
    }

    try:
        headers = _get_headers("openai", test_api_key)
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {test_api_key}"

        headers_replicate = _get_headers("replicate", test_api_key)
        assert "Authorization" in headers_replicate
        assert headers_replicate["Authorization"] == f"Token {test_api_key}"
    finally:
        # Restore original HOSTED_PROVIDERS
        HOSTED_PROVIDERS.clear()
        HOSTED_PROVIDERS.update(original_HOSTED_PROVIDERS)


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses):
        self.responses = responses
        self.request_count = 0

    async def handle_async_request(self, request):
        if self.request_count >= len(self.responses):
            raise Exception("No more mock responses available")
        response = self.responses[self.request_count]
        self.request_count += 1
        if isinstance(response, Exception):
            raise response
        if isinstance(response, httpx.TimeoutException):
            raise response
        return response


@pytest.mark.asyncio
async def test_handle_replicate_201_success():
    responses = [
        httpx.Response(
            200,  # Changed from 201 to 200 for the status check
            json={
                "status": "succeeded",  # Changed from processing to succeeded
                "output": "test output",
            },
        ),
    ]

    async with httpx.AsyncClient(transport=MockTransport(responses)) as client:
        response = httpx.Response(
            201,
            json={
                "status": "processing",
                "urls": {"get": "https://api.replicate.com/status/1"},
            },
        )
        result = await _handle_replicate_201(client, response, "test_key")
        assert result["status"] == "succeeded"
        assert result["output"] == "test output"


@pytest.mark.asyncio
async def test_handle_replicate_201_failure():
    responses = [
        httpx.Response(
            200,
            json={
                "status": "failed",
                "error": "Something went wrong",
            },
        ),
    ]

    async with httpx.AsyncClient(transport=MockTransport(responses)) as client:
        response = httpx.Response(
            201,
            json={
                "status": "processing",
                "urls": {"get": "https://api.replicate.com/status/1"},
            },
        )
        with pytest.raises(Exception):
            await _handle_replicate_201(client, response, "test_key")


@pytest.mark.asyncio
async def test_handle_replicate_201_invalid_response():
    response = httpx.Response(201, json={"invalid": "response"})
    async with httpx.AsyncClient() as client:
        with pytest.raises(Exception):
            await _handle_replicate_201(client, response, "test_key")


@pytest.mark.asyncio
async def test_llm_post_success():
    responses = [
        httpx.Response(200, json={"result": "success"}),
    ]

    async with httpx.AsyncClient(transport=MockTransport(responses)) as client:
        result = await llm_post(
            client,
            "openai",
            "gpt-4",
            "test_key",
            {"prompt": "test"},
            timeout=10,
            extra_params={},
        )
        assert result == {"result": "success"}


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_param_value", ["bar", 123, 0.0])
async def test_llm_post_success_with_extra_params(extra_param_value):
    responses = [httpx.Response(200, json={"result": "success"})]

    async with httpx.AsyncClient(transport=MockTransport(responses)) as client:
        result = await llm_post(
            client,
            "openai",
            "gpt-4",
            "test_key",
            {"prompt": "test"},
            timeout=10,
            extra_params={"foo": extra_param_value},
        )
        assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_llm_post_timeout():
    responses = [httpx.ReadTimeout("Request timed out")]

    async with httpx.AsyncClient(transport=MockTransport(responses)) as client:
        with pytest.raises(LLMTimeoutError):
            await llm_post(
                client,
                "openai",
                "gpt-4",
                "test_key",
                {"prompt": "test"},
                timeout=10,
                extra_params={},
            )


@pytest.mark.asyncio
async def test_llm_post_rate_limit():
    responses = [
        httpx.Response(429, text="Rate limit exceeded"),
    ]

    async with httpx.AsyncClient(transport=MockTransport(responses)) as client:
        with pytest.raises(LLMRateLimitError):
            await llm_post(
                client,
                "openai",
                "gpt-4",
                "test_key",
                {"prompt": "test"},
                timeout=10,
                extra_params={},
            )


@pytest.mark.asyncio
async def test_llm_post_error():
    responses = [
        httpx.Response(400, text="Bad request"),
    ]

    async with httpx.AsyncClient(transport=MockTransport(responses)) as client:
        with pytest.raises(Exception) as exc_info:
            await llm_post(
                client,
                "openai",
                "gpt-4",
                "test_key",
                {"prompt": "test"},
                timeout=10,
                extra_params={},
            )
        assert str(exc_info.value) == "Bad request"


@pytest.mark.asyncio
async def test_llm_post_replicate_success():
    responses = [
        httpx.Response(
            201,
            json={
                "status": "processing",
                "urls": {"get": "https://api.replicate.com/status/1"},
            },
        ),
        httpx.Response(
            200,
            json={
                "status": "succeeded",
                "output": "test output",
            },
        ),
    ]

    async with httpx.AsyncClient(transport=MockTransport(responses)) as client:
        result = await llm_post(
            client,
            "replicate",
            "model123",
            "test_key",
            {"prompt": "test"},
            timeout=10,
            extra_params={},
        )
        assert result["status"] == "succeeded"
        assert result["output"] == "test output"


@pytest.mark.asyncio
async def test_handle_replicate_201_status_check_failure():
    """Test case where the status check request fails"""
    responses = [httpx.Response(400, text="Bad status check request")]

    async with httpx.AsyncClient(transport=MockTransport(responses)) as client:
        response = httpx.Response(
            201,
            json={
                "status": "processing",
                "urls": {"get": "https://api.replicate.com/status/1"},
            },
        )
        with pytest.raises(Exception) as exc_info:
            await _handle_replicate_201(client, response, "test_key")
        assert str(exc_info.value) == "Bad status check request"


@pytest.mark.asyncio
async def test_llm_post_with_api_key_in_endpoint():
    """Test case where API_KEY needs to be replaced in the endpoint"""
    responses = [
        httpx.Response(200, json={"result": "success"}),
    ]

    # Mock HOSTED_PROVIDERS with API_KEY in endpoint
    original_HOSTED_PROVIDERS = HOSTED_PROVIDERS.copy()
    HOSTED_PROVIDERS["test_provider"] = {
        "headers": {"Authorization": f"Bearer {API_KEY}"},
        "endpoint": f"https://api.test.com/{API_KEY}/v1/chat",
    }

    try:
        async with httpx.AsyncClient(transport=MockTransport(responses)) as client:
            result = await llm_post(
                client,
                "test_provider",
                None,
                "test_key_123",
                {"prompt": "test"},
                timeout=10,
                extra_params={},
            )
            assert result == {"result": "success"}
    finally:
        # Restore original HOSTED_PROVIDERS
        HOSTED_PROVIDERS.clear()
        HOSTED_PROVIDERS.update(original_HOSTED_PROVIDERS)
