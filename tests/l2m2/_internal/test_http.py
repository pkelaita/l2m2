import pytest
import requests
import requests_mock
from unittest.mock import patch
import json

from l2m2.model_info import API_KEY, MODEL_ID
from l2m2._internal.http import _get_headers, _handle_replicate_201, llm_post


MOCK_PROVIDER_INFO = {
    "test_provider": {
        "headers": {"Authorization": f"Bearer {API_KEY}"},
        "endpoint": "https://api.testprovider.com/v1/models/" + MODEL_ID,
    },
    "replicate": {
        "headers": {"Authorization": f"Token {API_KEY}"},
        "endpoint": "https://api.replicate.com/v1/predictions",
    },
}


@pytest.fixture
def mock_provider_info():
    with patch("l2m2._internal.http.PROVIDER_INFO", MOCK_PROVIDER_INFO):
        yield


def test_get_headers(mock_provider_info):
    provider = "test_provider"
    api_key = "test_api_key"
    expected_headers = {"Authorization": "Bearer test_api_key"}
    headers = _get_headers(provider, api_key)
    assert headers == expected_headers


def test_handle_replicate_201_success(mock_provider_info):
    api_key = "test_api_key"
    resource_url = "https://api.replicate.com/v1/predictions/get"
    mock_resource = {
        "status": "succeeded",
        "urls": {"get": resource_url},
    }

    with requests_mock.Mocker() as m:
        m.get(resource_url, json=mock_resource)
        response = requests.Response()
        response.status_code = 201
        response._content = str.encode(
            '{"status": "starting", "urls": {"get": "%s"}}' % resource_url
        )

        result = _handle_replicate_201(response, api_key)
        assert result == mock_resource


def test_handle_replicate_201_failure(mock_provider_info):
    api_key = "test_api_key"
    resource_url = "https://api.replicate.com/v1/predictions/get"
    mock_resource = {
        "status": "failed",
        "urls": {"get": resource_url},
    }

    with requests_mock.Mocker() as m:
        m.get(resource_url, json=mock_resource)
        response = requests.Response()
        response.status_code = 201
        response._content = str.encode(
            '{"status": "starting", "urls": {"get": "%s"}}' % resource_url
        )

        with pytest.raises(Exception):
            _handle_replicate_201(response, api_key)


def test_llm_post_success(mock_provider_info):
    provider = "test_provider"
    api_key = "test_api_key"
    data = {"input": "test input"}
    model_id = "test_model_id"

    endpoint = (
        MOCK_PROVIDER_INFO[provider]["endpoint"]
        .replace(API_KEY, api_key)
        .replace(MODEL_ID, model_id)
    )
    expected_response = {"result": "success"}

    with requests_mock.Mocker() as m:
        m.post(endpoint, json=expected_response, status_code=200)
        result = llm_post(provider, api_key, data, model_id)
        assert result == expected_response


def test_llm_post_replicate(mock_provider_info):
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

    with requests_mock.Mocker() as m:
        m.post(endpoint, json=mock_initial_response, status_code=201)
        m.get(
            "https://api.replicate.com/v1/predictions/get", json=mock_success_response
        )
        result = llm_post(provider, api_key, data)
        assert result == mock_success_response


def test_handle_replicate_201_missing_keys(mock_provider_info):
    api_key = "test_api_key"
    invalid_resource = {"some_other_key": "some_value"}

    response = requests.Response()
    response.status_code = 201
    response._content = str.encode(json.dumps(invalid_resource))

    with pytest.raises(Exception):
        _handle_replicate_201(response, api_key)


def test_handle_replicate_201_status_failed(mock_provider_info):
    api_key = "test_api_key"
    resource_url = "https://api.replicate.com/v1/predictions/get"
    failed_resource = {
        "status": "failed",
        "urls": {"get": resource_url},
    }

    with requests_mock.Mocker() as m:
        m.get(resource_url, json=failed_resource)
        response = requests.Response()
        response.status_code = 201
        response._content = str.encode(
            '{"status": "starting", "urls": {"get": "%s"}}' % resource_url
        )

        with pytest.raises(Exception):
            _handle_replicate_201(response, api_key)


def test_handle_replicate_201_status_code_not_200(mock_provider_info):
    api_key = "test_api_key"
    resource_url = "https://api.replicate.com/v1/predictions/get"

    with requests_mock.Mocker() as m:
        m.get(resource_url, status_code=500, text="Internal Server Error")
        response = requests.Response()
        response.status_code = 201
        response._content = str.encode(
            '{"status": "starting", "urls": {"get": "%s"}}' % resource_url
        )

        with pytest.raises(Exception):
            _handle_replicate_201(response, api_key)


def test_llm_post_failure(mock_provider_info):
    provider = "test_provider"
    api_key = "test_api_key"
    data = {"input": "test input"}
    model_id = "test_model_id"

    endpoint = (
        MOCK_PROVIDER_INFO[provider]["endpoint"]
        .replace(API_KEY, api_key)
        .replace(MODEL_ID, model_id)
    )

    with requests_mock.Mocker() as m:
        m.post(endpoint, status_code=400, text="Bad Request")
        with pytest.raises(Exception):
            llm_post(provider, api_key, data, model_id)
