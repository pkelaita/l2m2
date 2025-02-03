from typing import Optional, Dict, Any, Union
import httpx

from l2m2.exceptions import LLMTimeoutError, LLMRateLimitError
from l2m2.model_info import (
    API_KEY,
    MODEL_ID,
    SERVICE_BASE_URL,
    HOSTED_PROVIDERS,
    LOCAL_PROVIDERS,
)


def _get_headers(provider: str, api_key: str) -> Dict[str, str]:
    provider_info = HOSTED_PROVIDERS[provider]
    headers = provider_info["headers"].copy()
    return {key: value.replace(API_KEY, api_key) for key, value in headers.items()}


def _get_timeout_message(timeout: Optional[int]) -> str:
    return (
        f"Request timed out after {timeout} seconds. Try increasing the timeout by passing "
        + "the timeout parameter into call, or reducing the expected size of the output."
    )


async def _handle_replicate_201(
    client: httpx.AsyncClient,
    response: httpx.Response,
    api_key: str,
) -> Any:
    resource = response.json()
    if "status" in resource and "urls" in resource and "get" in resource["urls"]:
        while resource["status"] != "succeeded":
            if resource["status"] == "failed" or resource["status"] == "cancelled":
                raise Exception(resource)

            next_response = await client.get(
                resource["urls"]["get"],
                headers=_get_headers("replicate", api_key),
            )

            if next_response.status_code != 200:
                raise Exception(next_response.text)
            resource = next_response.json()

        return resource
    else:
        raise Exception(resource)


async def llm_post(
    client: httpx.AsyncClient,
    provider: str,
    model_id: str,
    api_key: str,
    data: Dict[str, Any],
    timeout: Optional[int],
    extra_params: Optional[Dict[str, Union[str, int, float]]],
    extra_headers: Optional[Dict[str, str]],
) -> Any:
    endpoint = HOSTED_PROVIDERS[provider]["endpoint"]
    if API_KEY in endpoint:
        endpoint = endpoint.replace(API_KEY, api_key)
    if MODEL_ID in endpoint and model_id is not None:
        endpoint = endpoint.replace(MODEL_ID, model_id)

    if extra_params:
        data.update(extra_params)

    headers = _get_headers(provider, api_key)
    if extra_headers:
        headers.update(extra_headers)

    try:
        response = await client.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=timeout,
        )
    except httpx.ReadTimeout:
        raise LLMTimeoutError(_get_timeout_message(timeout))

    if provider == "replicate" and response.status_code == 201:
        return await _handle_replicate_201(client, response, api_key)

    if response.status_code == 429:
        raise LLMRateLimitError(
            f"Reached rate limit for provider {provider} with model {model_id}."
        )

    elif response.status_code != 200:
        raise Exception(response.text)

    return response.json()


async def local_llm_post(
    client: httpx.AsyncClient,
    provider: str,
    data: Dict[str, Any],
    timeout: Optional[int],
    local_provider_overrides: Dict[str, str],
    extra_params: Optional[Dict[str, Union[str, int, float]]],
    extra_headers: Optional[Dict[str, str]],
) -> Any:
    provider_info = LOCAL_PROVIDERS[provider]

    endpoint = provider_info["endpoint"]
    base_url = local_provider_overrides.get(provider, provider_info["default_base_url"])

    if SERVICE_BASE_URL in endpoint:
        endpoint = endpoint.replace(SERVICE_BASE_URL, base_url)

    if extra_params:
        data.update(extra_params)

    data["stream"] = False

    headers = provider_info["headers"]
    if extra_headers:
        headers.update(extra_headers)

    try:
        response = await client.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=timeout,
        )
    except httpx.ReadTimeout:
        raise LLMTimeoutError(_get_timeout_message(timeout))

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()
