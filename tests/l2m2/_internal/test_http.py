import pytest
import httpx
import respx
from unittest.mock import patch

from l2m2.exceptions import LLMRateLimitError, LLMTimeoutError
from l2m2.model_info import API_KEY, MODEL_ID
from l2m2._internal.http import (
    _get_headers,
    _handle_replicate_201,
    llm_post,
)

PROVIDER_INFO_PATH = "l2m2._internal.http.PROVIDER_INFO"


MOCK_PROVIDER_INFO = {
    "test_provider": {
        "headers": {"Authorization": f"Bearer {API_KEY}"},
        "endpoint": "https://api.testprovider.com/v1",
    },
    "test_provider_model_id_in_url": {
        "headers": {"Authorization": f"Bearer {API_KEY}"},
        "endpoint": f"https://api.testprovider.com/v1/models/{MODEL_ID}",
    },
    "test_provider_api_key_in_url": {
        "headers": {"foo": "bar"},
        "endpoint": f"https://api.testprovider.com/v1?key={API_KEY}",
    },
    "test_provider_both_in_url": {
        "headers": {"foo": "bar"},
        "endpoint": f"https://api.testprovider.com/v1/models/{MODEL_ID}?key={API_KEY}",
    },
    # Need to test replicate's implementation separately
    "replicate": {
        "headers": {"Authorization": f"Token {API_KEY}"},
        "endpoint": "https://api.replicate.com/v1/predictions",
    },
}


@patch(PROVIDER_INFO_PATH, MOCK_PROVIDER_INFO)
def test_get_headers():
    provider = "test_provider"
    api_key = "test_api_key"
    expected_headers = {"Authorization": "Bearer test_api_key"}
    headers = _get_headers(provider, api_key)
    assert headers == expected_headers


@pytest.mark.asyncio
@respx.mock
@patch(PROVIDER_INFO_PATH, MOCK_PROVIDER_INFO)
async def test_handle_replicate_201_success():
    api_key = "test_api_key"
    resource_url = "https://api.replicate.com/v1/predictions/get"
    mock_resource = {
        "status": "succeeded",
        "urls": {"get": resource_url},
    }

    respx.get(resource_url).mock(return_value=httpx.Response(200, json=mock_resource))
    response = httpx.Response(
        201,
        json={
            "status": "starting",
            "urls": {"get": resource_url},
        },
    )

    async with httpx.AsyncClient() as client:
        result = await _handle_replicate_201(client, response, api_key)
        assert result == mock_resource


@pytest.mark.asyncio
@respx.mock
@patch(PROVIDER_INFO_PATH, MOCK_PROVIDER_INFO)
async def test_handle_replicate_201_failure():
    api_key = "test_api_key"
    resource_url = "https://api.replicate.com/v1/predictions/get"
    mock_resource = {
        "status": "failed",
        "urls": {"get": resource_url},
    }

    respx.get(resource_url).mock(return_value=httpx.Response(200, json=mock_resource))
    response = httpx.Response(
        201,
        json={
            "status": "starting",
            "urls": {"get": resource_url},
        },
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(Exception):
            await _handle_replicate_201(client, response, api_key)


@pytest.mark.asyncio
@respx.mock
@patch(PROVIDER_INFO_PATH, MOCK_PROVIDER_INFO)
async def test_llm_post_success():
    async def _test_generic_llm_post(provider):
        api_key = "test_api_key"
        data = {"input": "test input"}
        model_id = "test_model_id"

        endpoint = (
            MOCK_PROVIDER_INFO[provider]["endpoint"]
            .replace(API_KEY, api_key)
            .replace(MODEL_ID, model_id)
        )
        expected_response = {"result": "success"}

        respx.post(endpoint).mock(
            return_value=httpx.Response(200, json=expected_response)
        )
        async with httpx.AsyncClient() as client:
            result = await llm_post(
                client=client,
                provider=provider,
                model_id=model_id,
                api_key=api_key,
                data=data,
                timeout=10,
            )
            assert result == expected_response

    await _test_generic_llm_post("test_provider")
    await _test_generic_llm_post("test_provider_model_id_in_url")
    await _test_generic_llm_post("test_provider_api_key_in_url")
    await _test_generic_llm_post("test_provider_both_in_url")


@pytest.mark.asyncio
@respx.mock
@patch(PROVIDER_INFO_PATH, MOCK_PROVIDER_INFO)
async def test_llm_post_replicate():
    provider = "replicate"
    api_key = "test_api_key"
    data = {"input": "test input"}

    endpoint = MOCK_PROVIDER_INFO[provider]["endpoint"].replace(API_KEY, api_key)
    mock_initial_response = {
        "status": "starting",
        "urls": {"get": "https://api.replicate.com/v1/predictions/get"},
    }
    mock_success_response = {
        "status": "succeeded",
        "urls": {"get": "https://api.replicate.com/v1/predictions/get"},
    }

    respx.post(endpoint).mock(
        return_value=httpx.Response(201, json=mock_initial_response)
    )
    respx.get("https://api.replicate.com/v1/predictions/get").mock(
        return_value=httpx.Response(200, json=mock_success_response)
    )

    async with httpx.AsyncClient() as client:
        result = await llm_post(
            client=client,
            provider=provider,
            model_id="fake_model_id",
            api_key=api_key,
            data=data,
            timeout=10,
        )
        assert result == mock_success_response


@pytest.mark.asyncio
@patch(PROVIDER_INFO_PATH, MOCK_PROVIDER_INFO)
async def test_handle_replicate_201_missing_keys():
    api_key = "test_api_key"
    invalid_resource = {"some_other_key": "some_value"}

    response = httpx.Response(201, json=invalid_resource)

    async with httpx.AsyncClient() as client:
        with pytest.raises(Exception):
            await _handle_replicate_201(client, response, api_key)


@pytest.mark.asyncio
@respx.mock
@patch(PROVIDER_INFO_PATH, MOCK_PROVIDER_INFO)
async def test_handle_replicate_201_status_failed():
    api_key = "test_api_key"
    resource_url = "https://api.replicate.com/v1/predictions/get"
    failed_resource = {
        "status": "failed",
        "urls": {"get": resource_url},
    }

    respx.get(resource_url).mock(return_value=httpx.Response(200, json=failed_resource))
    response = httpx.Response(
        201,
        json={
            "status": "starting",
            "urls": {"get": resource_url},
        },
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(Exception):
            await _handle_replicate_201(client, response, api_key)


@pytest.mark.asyncio
@respx.mock
@patch(PROVIDER_INFO_PATH, MOCK_PROVIDER_INFO)
async def test_handle_replicate_201_status_code_not_200():
    api_key = "test_api_key"
    resource_url = "https://api.replicate.com/v1/predictions/get"

    respx.get(resource_url).mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )
    response = httpx.Response(
        201,
        json={
            "status": "starting",
            "urls": {"get": resource_url},
        },
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(Exception):
            await _handle_replicate_201(client, response, api_key)


@pytest.mark.asyncio
@respx.mock
@patch(PROVIDER_INFO_PATH, MOCK_PROVIDER_INFO)
async def test_llm_post_failure():
    provider = "test_provider"
    api_key = "test_api_key"
    data = {"input": "test input"}
    model_id = "test_model_id"

    endpoint = (
        MOCK_PROVIDER_INFO[provider]["endpoint"]
        .replace(API_KEY, api_key)
        .replace(MODEL_ID, model_id)
    )

    respx.post(endpoint).mock(return_value=httpx.Response(400, text="Bad Request"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(Exception):
            await llm_post(
                client=client,
                provider=provider,
                model_id=model_id,
                api_key=api_key,
                data=data,
                timeout=10,
            )


@pytest.mark.asyncio
@respx.mock
@patch(PROVIDER_INFO_PATH, MOCK_PROVIDER_INFO)
async def test_llm_post_timeout():
    provider = "test_provider"
    api_key = "test_api_key"
    data = {"input": "test input"}
    model_id = "test_model_id"
    timeout = 5

    endpoint = (
        MOCK_PROVIDER_INFO[provider]["endpoint"]
        .replace(API_KEY, api_key)
        .replace(MODEL_ID, model_id)
    )

    respx.post(endpoint).mock(side_effect=httpx.ReadTimeout)
    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMTimeoutError):
            await llm_post(
                client=client,
                provider=provider,
                model_id=model_id,
                api_key=api_key,
                data=data,
                timeout=timeout,
            )


@pytest.mark.asyncio
@respx.mock
@patch(PROVIDER_INFO_PATH, MOCK_PROVIDER_INFO)
async def test_llm_post_rate_limit_error():
    provider = "test_provider"
    api_key = "test_api_key"
    data = {"input": "test input"}
    model_id = "test_model_id"

    endpoint = (
        MOCK_PROVIDER_INFO[provider]["endpoint"]
        .replace(API_KEY, api_key)
        .replace(MODEL_ID, model_id)
    )

    respx.post(endpoint).mock(
        return_value=httpx.Response(429, text="Rate Limit Exceeded")
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(LLMRateLimitError):
            await llm_post(
                client=client,
                provider=provider,
                model_id=model_id,
                api_key=api_key,
                data=data,
                timeout=10,
            )
