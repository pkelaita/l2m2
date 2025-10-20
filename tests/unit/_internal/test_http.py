from l2m2._internal.http import (
    _get_headers,
    _handle_replicate_201,
    llm_post,
    local_llm_post,
)
import pytest
import asyncio
from l2m2.exceptions import LLMTimeoutError, LLMRateLimitError
from l2m2.model_info import API_KEY, HOSTED_PROVIDERS, LOCAL_PROVIDERS, SERVICE_BASE_URL


class FakeResponse:
    def __init__(self, status: int, json: dict | None = None, text: str | None = None):
        self.status = status
        self._json = json
        self._text = text or ""
        self.request = type("Req", (), {"headers": {}})()

    async def json(self):
        if self._json is None:
            raise ValueError("No JSON set on FakeResponse")
        return self._json

    async def text(self):
        return self._text


class FakeSession:
    def __init__(self, responses):
        self._responses = responses
        self._i = 0
        self._last_request_headers = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, endpoint, headers=None, json=None, timeout=None):
        if self._i >= len(self._responses):
            raise Exception("No more mock responses available")
        self._last_request_headers = headers or {}
        resp = self._responses[self._i]
        self._i += 1
        # attach headers to response.request for assertions
        resp.request.headers = self._last_request_headers
        return resp

    async def get(self, url, headers=None):
        if self._i >= len(self._responses):
            raise Exception("No more mock responses available")
        resp = self._responses[self._i]
        self._i += 1
        return resp


# -- Tests for headers -- #


def test_get_headers():
    test_api_key = "test_key_123"

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
        HOSTED_PROVIDERS.clear()
        HOSTED_PROVIDERS.update(original_HOSTED_PROVIDERS)


# -- Tests for replicate handling -- #


@pytest.mark.asyncio
async def test_handle_replicate_201_success():
    responses = [
        FakeResponse(200, json={"status": "succeeded", "output": "test output"}),
    ]

    async with FakeSession(responses) as client:
        response = FakeResponse(
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
        FakeResponse(200, json={"status": "failed", "error": "Something went wrong"}),
    ]

    async with FakeSession(responses) as client:
        response = FakeResponse(
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
    response = FakeResponse(201, json={"invalid": "response"})
    async with FakeSession([]) as client:
        with pytest.raises(Exception):
            await _handle_replicate_201(client, response, "test_key")


@pytest.mark.asyncio
async def test_handle_replicate_201_status_check_failure():
    """Test case where the status check request fails"""
    responses = [FakeResponse(400, text="Bad status check request")]

    async with FakeSession(responses) as client:
        response = FakeResponse(
            201,
            json={
                "status": "processing",
                "urls": {"get": "https://api.replicate.com/status/1"},
            },
        )
        with pytest.raises(Exception) as exc_info:
            await _handle_replicate_201(client, response, "test_key")
        assert str(exc_info.value) == "Bad status check request"


# -- Tests for hosted LLM posts -- #


@pytest.mark.asyncio
async def test_llm_post_success():
    responses = [FakeResponse(200, json={"result": "success"})]

    async with FakeSession(responses) as client:
        result = await llm_post(
            client,
            "openai",
            "gpt-4",
            "test_key",
            {"prompt": "test"},
            timeout=10,
            extra_params={},
            extra_headers={},
        )
        assert result == {"result": "success"}


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_param_value", ["bar", 123, 0.0])
async def test_llm_post_success_with_extra_params(extra_param_value):
    responses = [FakeResponse(200, json={"result": "success"})]

    async with FakeSession(responses) as client:
        result = await llm_post(
            client,
            "openai",
            "gpt-4",
            "test_key",
            {"prompt": "test"},
            timeout=10,
            extra_params={"foo": extra_param_value},
            extra_headers={},
        )
        assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_llm_post_timeout():
    class TimeoutSession(FakeSession):
        async def post(self, *args, **kwargs):
            raise asyncio.TimeoutError()

    async with TimeoutSession([]) as client:
        with pytest.raises(LLMTimeoutError):
            await llm_post(
                client,
                "openai",
                "gpt-4",
                "test_key",
                {"prompt": "test"},
                timeout=10,
                extra_params={},
                extra_headers={},
            )


@pytest.mark.asyncio
async def test_llm_post_rate_limit():
    responses = [FakeResponse(429, text="Rate limit exceeded")]

    async with FakeSession(responses) as client:
        with pytest.raises(LLMRateLimitError):
            await llm_post(
                client,
                "openai",
                "gpt-4",
                "test_key",
                {"prompt": "test"},
                timeout=10,
                extra_params={},
                extra_headers={},
            )


@pytest.mark.asyncio
async def test_llm_post_error():
    responses = [FakeResponse(400, text="Bad request")]

    async with FakeSession(responses) as client:
        with pytest.raises(Exception) as exc_info:
            await llm_post(
                client,
                "openai",
                "gpt-4",
                "test_key",
                {"prompt": "test"},
                timeout=10,
                extra_params={},
                extra_headers={},
            )
        assert str(exc_info.value) == "Bad request"


@pytest.mark.asyncio
async def test_llm_post_replicate_success():
    responses = [
        FakeResponse(
            201,
            json={
                "status": "processing",
                "urls": {"get": "https://api.replicate.com/status/1"},
            },
        ),
        FakeResponse(200, json={"status": "succeeded", "output": "test output"}),
    ]

    async with FakeSession(responses) as client:
        result = await llm_post(
            client,
            "replicate",
            "model123",
            "test_key",
            {"prompt": "test"},
            timeout=10,
            extra_params={},
            extra_headers={},
        )
        assert result["status"] == "succeeded"
        assert result["output"] == "test output"


@pytest.mark.asyncio
async def test_llm_post_with_api_key_in_endpoint():
    """Test case where API_KEY needs to be replaced in the endpoint"""
    responses = [FakeResponse(200, json={"result": "success"})]

    original_HOSTED_PROVIDERS = HOSTED_PROVIDERS.copy()
    HOSTED_PROVIDERS["test_provider"] = {
        "headers": {"Authorization": f"Bearer {API_KEY}"},
        "endpoint": f"https://api.test.com/{API_KEY}/v1/chat",
    }

    try:
        async with FakeSession(responses) as client:
            result = await llm_post(
                client,
                "test_provider",
                "test_model",
                "test_key_123",
                {"prompt": "test"},
                timeout=10,
                extra_params={},
                extra_headers={},
            )
            assert result == {"result": "success"}
    finally:
        HOSTED_PROVIDERS.clear()
        HOSTED_PROVIDERS.update(original_HOSTED_PROVIDERS)


@pytest.mark.asyncio
async def test_llm_post_with_extra_headers():
    """Test that extra headers are properly added to the request"""
    responses = [FakeResponse(200, json={"result": "success"})]

    async with FakeSession(responses) as client:
        await llm_post(
            client,
            "openai",
            "gpt-4",
            "test_key",
            {"prompt": "test"},
            timeout=10,
            extra_params={},
            extra_headers={"X-Custom-Header": "test-value"},
        )
        request = client._responses[0].request
        assert "X-Custom-Header" in request.headers
        assert request.headers["X-Custom-Header"] == "test-value"
        # Verify original headers are still present
        assert "Authorization" in request.headers
        assert request.headers["Authorization"] == "Bearer test_key"


# -- Tests for local LLM posts -- #


@pytest.mark.asyncio
async def test_local_llm_post_success():
    """Test successful local LLM post with default base URL"""
    responses = [FakeResponse(200, json={"result": "success"})]

    original_LOCAL_PROVIDERS = LOCAL_PROVIDERS.copy()
    LOCAL_PROVIDERS["local_provider"] = {
        "headers": {"Content-Type": "application/json"},
        "endpoint": f"{SERVICE_BASE_URL}/v1/completions",
        "default_base_url": "http://localhost:8000",
    }

    try:
        async with FakeSession(responses) as client:
            result = await local_llm_post(
                client,
                "local_provider",
                {"prompt": "test"},
                timeout=10,
                local_provider_overrides={},
                extra_params={},
                extra_headers={},
            )
            assert result == {"result": "success"}
    finally:
        LOCAL_PROVIDERS.clear()
        LOCAL_PROVIDERS.update(original_LOCAL_PROVIDERS)


@pytest.mark.asyncio
async def test_local_llm_post_with_override():
    """Test local LLM post with overridden base URL"""
    responses = [FakeResponse(200, json={"result": "success"})]

    original_LOCAL_PROVIDERS = LOCAL_PROVIDERS.copy()
    LOCAL_PROVIDERS["local_provider"] = {
        "headers": {"Content-Type": "application/json"},
        "endpoint": f"{SERVICE_BASE_URL}/v1/completions",
        "default_base_url": "http://localhost:8000",
    }

    try:
        async with FakeSession(responses) as client:
            result = await local_llm_post(
                client,
                "local_provider",
                {"prompt": "test"},
                timeout=10,
                local_provider_overrides={"local_provider": "http://custom:8080"},
                extra_params={"temperature": 0.7},
                extra_headers={},
            )
            assert result == {"result": "success"}
    finally:
        LOCAL_PROVIDERS.clear()
        LOCAL_PROVIDERS.update(original_LOCAL_PROVIDERS)


@pytest.mark.asyncio
async def test_local_llm_post_timeout():
    """Test timeout handling in local LLM post"""

    class TimeoutSession(FakeSession):
        async def post(self, *args, **kwargs):
            raise asyncio.TimeoutError()

    original_LOCAL_PROVIDERS = LOCAL_PROVIDERS.copy()
    LOCAL_PROVIDERS["local_provider"] = {
        "headers": {"Content-Type": "application/json"},
        "endpoint": f"{SERVICE_BASE_URL}/v1/completions",
        "default_base_url": "http://localhost:8000",
    }

    try:
        async with TimeoutSession([]) as client:
            with pytest.raises(LLMTimeoutError):
                await local_llm_post(
                    client,
                    "local_provider",
                    {"prompt": "test"},
                    timeout=10,
                    local_provider_overrides={},
                    extra_params={},
                    extra_headers={},
                )
    finally:
        LOCAL_PROVIDERS.clear()
        LOCAL_PROVIDERS.update(original_LOCAL_PROVIDERS)


@pytest.mark.asyncio
async def test_local_llm_post_error():
    """Test error handling in local LLM post"""
    responses = [FakeResponse(400, text="Bad request")]

    original_LOCAL_PROVIDERS = LOCAL_PROVIDERS.copy()
    LOCAL_PROVIDERS["local_provider"] = {
        "headers": {"Content-Type": "application/json"},
        "endpoint": f"{SERVICE_BASE_URL}/v1/completions",
        "default_base_url": "http://localhost:8000",
    }

    try:
        async with FakeSession(responses) as client:
            with pytest.raises(Exception) as exc_info:
                await local_llm_post(
                    client,
                    "local_provider",
                    {"prompt": "test"},
                    timeout=10,
                    local_provider_overrides={},
                    extra_params={},
                    extra_headers={},
                )
            assert str(exc_info.value) == "Bad request"
    finally:
        LOCAL_PROVIDERS.clear()
        LOCAL_PROVIDERS.update(original_LOCAL_PROVIDERS)


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_param_value", ["bar", 123, 0.7])
async def test_local_llm_post_with_extra_params(extra_param_value):
    """Test local LLM post with various extra parameters"""
    responses = [FakeResponse(200, json={"result": "success"})]

    original_LOCAL_PROVIDERS = LOCAL_PROVIDERS.copy()
    LOCAL_PROVIDERS["local_provider"] = {
        "headers": {"Content-Type": "application/json"},
        "endpoint": f"{SERVICE_BASE_URL}/v1/completions",
        "default_base_url": "http://localhost:8000",
    }

    try:
        async with FakeSession(responses) as client:
            result = await local_llm_post(
                client,
                "local_provider",
                {"prompt": "test"},
                timeout=10,
                local_provider_overrides={},
                extra_params={"param": extra_param_value},
                extra_headers={},
            )
            assert result == {"result": "success"}
    finally:
        LOCAL_PROVIDERS.clear()
        LOCAL_PROVIDERS.update(original_LOCAL_PROVIDERS)


@pytest.mark.asyncio
async def test_local_llm_post_with_extra_headers():
    """Test that extra headers are properly added to the request for local providers"""
    responses = [FakeResponse(200, json={"result": "success"})]

    original_LOCAL_PROVIDERS = LOCAL_PROVIDERS.copy()
    LOCAL_PROVIDERS["local_provider"] = {
        "headers": {"Content-Type": "application/json"},
        "endpoint": f"{SERVICE_BASE_URL}/v1/completions",
        "default_base_url": "http://localhost:8000",
    }

    try:
        async with FakeSession(responses) as client:
            await local_llm_post(
                client,
                "local_provider",
                {"prompt": "test"},
                timeout=10,
                local_provider_overrides={},
                extra_params={},
                extra_headers={"X-Custom-Header": "test-value"},
            )
            request = client._responses[0].request
            assert "X-Custom-Header" in request.headers
            assert request.headers["X-Custom-Header"] == "test-value"
            # Verify original headers are still present
            assert "Content-Type" in request.headers
            assert request.headers["Content-Type"] == "application/json"
    finally:
        LOCAL_PROVIDERS.clear()
        LOCAL_PROVIDERS.update(original_LOCAL_PROVIDERS)
