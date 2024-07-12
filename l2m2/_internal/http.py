from typing import Optional, Dict, Any
import httpx

from l2m2.exceptions import LLMTimeoutError, LLMRateLimitError
from l2m2.model_info import API_KEY, MODEL_ID, PROVIDER_INFO


def _get_headers(provider: str, api_key: str) -> Dict[str, str]:
    provider_info = PROVIDER_INFO[provider]
    headers = provider_info["headers"].copy()
    return {key: value.replace(API_KEY, api_key) for key, value in headers.items()}


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
    api_key: str,
    data: Dict[str, Any],
    timeout: Optional[int],
    model_id: Optional[str] = None,
) -> Any:
    endpoint = PROVIDER_INFO[provider]["endpoint"]
    if API_KEY in endpoint:
        endpoint = endpoint.replace(API_KEY, api_key)
    if MODEL_ID in endpoint and model_id is not None:
        endpoint = endpoint.replace(MODEL_ID, model_id)
    try:
        response = await client.post(
            endpoint,
            headers=_get_headers(provider, api_key),
            json=data,
            timeout=timeout,
        )
    except httpx.ReadTimeout:
        msg = (
            f"Request timed out after {timeout} seconds. Try increasing the timeout"
            + ", or reducing the size of the input."
        )
        raise LLMTimeoutError(msg)

    if provider == "replicate" and response.status_code == 201:
        return await _handle_replicate_201(client, response, api_key)

    if response.status_code == 429:
        raise LLMRateLimitError(
            f"Reached rate limit for provider {provider} with model {model_id}."
        )

    elif response.status_code != 200:
        raise Exception(response.text)

    return response.json()
