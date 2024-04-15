import pytest
from unittest.mock import patch, Mock

# These aren't used, but are imported to ensure they are available
from openai import OpenAI  # noqa: F401
from cohere import Client as CohereClient  # noqa: F401
from anthropic import Anthropic  # noqa: F401
from groq import Groq  # noqa: F401
import google.generativeai as google  # noqa: F401

from test_utils.llm_mock import (
    construct_mock_from_path,
    get_nested_attribute,
)
from l2m2.llm_client import LLMClient


@pytest.fixture
def llm_client():
    """Fixture to provide a clean LLMManager instance for each test."""
    return LLMClient()


# -- Tests for initialization and provider management -- #


def test_init(llm_client):
    assert llm_client.API_KEYS == {}
    assert llm_client.active_providers == set()
    assert llm_client.active_models == set()


def test_with_providers():
    llm_client = LLMClient.with_providers(
        {"openai": "test-key-openai", "cohere": "test-key-cohere"}
    )
    assert llm_client.API_KEYS == {
        "openai": "test-key-openai",
        "cohere": "test-key-cohere",
    }
    assert llm_client.active_providers == {"openai", "cohere"}
    assert "gpt-4-turbo" in llm_client.active_models
    assert "command-r" in llm_client.active_models
    assert "claude-3-opus" not in llm_client.active_models


def test_with_providers_invalid():
    with pytest.raises(ValueError):
        LLMClient.with_providers(
            {"invalid_provider": "some-key", "openai": "test-key-openai"}
        )


def test_getters(llm_client):
    llm_client.add_provider("openai", "test-key-openai")
    llm_client.add_provider("cohere", "test-key-cohere")
    assert llm_client.get_active_providers() == {"openai", "cohere"}
    active_models = llm_client.get_active_models()
    assert "gpt-4-turbo" in active_models
    assert "command-r" in active_models
    assert "claude-3-opus" not in active_models


def test_add_provider(llm_client):
    llm_client.add_provider("openai", "test-key-openai")
    assert "openai" in llm_client.active_providers
    assert "gpt-4-turbo" in llm_client.active_models


def test_add_provider_invalid(llm_client):
    with pytest.raises(ValueError):
        llm_client.add_provider("invalid_provider", "some-key")


def test_remove_provider(llm_client):
    with patch.object(
        LLMClient,
        "get_available_providers",
        return_value={"openai"},
    ):
        llm_client.add_provider("openai", "test-key-openai")
        llm_client.remove_provider("openai")
        assert "openai" not in llm_client.active_providers
        assert "gpt-4-turbo" not in llm_client.active_models


def test_remove_provider_not_active(llm_client):
    with pytest.raises(ValueError):
        llm_client.remove_provider("openai")


# -- Tests for call -- #


def _generic_test_call(
    llm_client,
    provider_patch_path,
    call_path,
    response_path,
    provider_key,
    model_name,
):
    with patch(provider_patch_path) as mock_provider:
        mock_client = Mock()

        # Dynamically get the mock call and response objects based on the delimited paths
        mock_call = get_nested_attribute(mock_client, call_path)
        mock_response = construct_mock_from_path(response_path)
        mock_call.return_value = mock_response

        mock_provider.return_value = mock_client

        llm_client.add_provider(provider_key, "fake-api-key")
        response_default = llm_client.call(prompt="Hello", model=model_name)
        response_custom = llm_client.call(
            prompt="Hello",
            model=model_name,
            system_prompt="System prompt",
            temperature=0.5,
            max_tokens=100,
        )

        assert response_default == "response"
        assert response_custom == "response"


def test_call_openai(llm_client):
    _generic_test_call(
        llm_client=llm_client,
        provider_patch_path="l2m2.llm_client.OpenAI",
        call_path="chat.completions.create",
        response_path="choices[0].message.content",
        provider_key="openai",
        model_name="gpt-4-turbo",
    )


def test_call_anthropic(llm_client):
    _generic_test_call(
        llm_client=llm_client,
        provider_patch_path="l2m2.llm_client.Anthropic",
        call_path="messages.create",
        response_path="content[0].text",
        provider_key="anthropic",
        model_name="claude-3-opus",
    )


def test_call_cohere(llm_client):
    _generic_test_call(
        llm_client=llm_client,
        provider_patch_path="l2m2.llm_client.CohereClient",
        call_path="chat",
        response_path="text",
        provider_key="cohere",
        model_name="command-r",
    )


def test_call_groq(llm_client):
    _generic_test_call(
        llm_client=llm_client,
        provider_patch_path="l2m2.llm_client.Groq",
        call_path="chat.completions.create",
        response_path="choices[0].message.content",
        provider_key="groq",
        model_name="llama2-70b",
    )


# Need to test gemini 1.0 and 1.5 separately because 1.0 doesn't support system prompts
def test_call_google_1_5(llm_client):
    _generic_test_call(
        llm_client=llm_client,
        provider_patch_path="l2m2.llm_client.google.GenerativeModel",
        call_path="generate_content",
        response_path="candidates[0].content.parts[0].text",
        provider_key="google",
        model_name="gemini-1.5-pro",
    )


def test_call_google_1_0(llm_client):
    _generic_test_call(
        llm_client=llm_client,
        provider_patch_path="l2m2.llm_client.google.GenerativeModel",
        call_path="generate_content",
        response_path="candidates[0].content.parts[0].text",
        provider_key="google",
        model_name="gemini-1.0-pro",
    )


def test_call_valid_model_not_active(llm_client):
    with pytest.raises(ValueError):
        llm_client.call(prompt="Hello", model="gpt-4-turbo")


def test_call_invalid_model(llm_client):
    with pytest.raises(ValueError):
        llm_client.call(prompt="Hello", model="unknown-model")


def test_call_tokens_too_large(llm_client):
    llm_client.add_provider("openai", "fake-api-key")
    with pytest.raises(ValueError):
        llm_client.call(prompt="Hello", model="gpt-4-turbo", max_tokens=float("inf"))


def test_call_temperature_too_high(llm_client):
    llm_client.add_provider("openai", "fake-api-key")
    with pytest.raises(ValueError):
        llm_client.call(prompt="Hello", model="gpt-4-turbo", temperature=3.0)


# -- Tests for call_custom -- #


def test_call_custom(llm_client):
    with patch("l2m2.llm_client.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_call = mock_client.chat.completions.create
        mock_response = construct_mock_from_path("choices[0].message.content")
        mock_call.return_value = mock_response
        mock_openai.return_value = mock_client

        llm_client.add_provider("openai", "fake-api-key")
        response_default = llm_client.call_custom(
            provider="openai",
            prompt="Hello",
            model="custom-model-xyz",
        )
        response_custom = llm_client.call_custom(
            provider="openai",
            prompt="Hello",
            model="custom-model-xyz",
            system_prompt="System prompt",
            temperature=0.5,
        )

        assert response_default == "response"
        assert response_custom == "response"


def test_call_custom_invalid_provider(llm_client):
    with pytest.raises(ValueError):
        llm_client.call_custom(
            provider="invalid_provider",
            prompt="Hello",
            model="custom-model-xyz",
        )


def test_call_custom_not_active(llm_client):
    with pytest.raises(ValueError):
        llm_client.call_custom(
            provider="openai",
            prompt="Hello",
            model="custom-model-xyz",
        )
