from typing import Optional, Dict, Any
import requests

from l2m2.model_info import API_KEY, MODEL_ID, PROVIDER_INFO


def _get_headers(provider: str, api_key: str) -> Dict[str, str]:
    provider_info = PROVIDER_INFO[provider]
    headers = provider_info["headers"].copy()
    return {key: value.replace(API_KEY, api_key) for key, value in headers.items()}


def _handle_replicate_201(response: requests.Response, api_key: str) -> Any:
    # See https://replicate.com/docs/reference/http#models.versions.get
    resource = response.json()
    if "status" in resource and "urls" in resource and "get" in resource["urls"]:

        while resource["status"] != "succeeded":
            if resource["status"] == "failed" or resource["status"] == "cancelled":
                raise Exception(resource)

            next_response = requests.get(
                resource["urls"]["get"],
                headers=_get_headers("replicate", api_key),
            )

            if next_response.status_code != 200:
                raise Exception(next_response.text)

            resource = next_response.json()

        return resource

    else:
        raise Exception(resource)


def llm_post(
    provider: str,
    api_key: str,
    data: Dict[str, Any],
    model_id: Optional[str] = None,
) -> Any:
    endpoint = PROVIDER_INFO[provider]["endpoint"]
    endpoint = endpoint.replace(API_KEY, api_key)
    if model_id is not None:
        endpoint = endpoint.replace(MODEL_ID, model_id)

    response = requests.post(
        endpoint,
        headers=_get_headers(provider, api_key),
        json=data,
    )

    if provider == "replicate" and response.status_code == 201:
        return _handle_replicate_201(response, api_key)

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()
