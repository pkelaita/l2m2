import pytest
from unittest.mock import patch, Mock

from test_utils.llm_mock import (
    construct_mock_from_path,
    get_nested_attribute,
)
from l2m2.client import LLMClient

MODULE_PATH = "l2m2.client.llm_client"


# Make sure all the providers are available
def test_provider_imports():
    from openai import OpenAI  # noqa: F401
    from cohere import Client as CohereClient  # noqa: F401
    from anthropic import Anthropic  # noqa: F401
    from groq import Groq  # noqa: F401
    import google.generativeai as google  # noqa: F401
    import replicate  # noqa: F401


@pytest.fixture
def llm_client():
    """Fixture to provide a clean LLMManager instance for each test."""
    return LLMClient()


# -- Tests for initialization and provider management -- #


def test_init(llm_client):
    assert llm_client.api_keys == {}
    assert llm_client.active_providers == set()
    assert llm_client.active_models == set()


def test_init_with_providers():
    llm_client = LLMClient({"openai": "test-key-openai", "cohere": "test-key-cohere"})
    assert llm_client.api_keys == {
        "openai": "test-key-openai",
        "cohere": "test-key-cohere",
    }
    assert llm_client.active_providers == {"openai", "cohere"}
    assert "gpt-4-turbo" in llm_client.active_models
    assert "command-r" in llm_client.active_models
    assert "claude-3-opus" not in llm_client.active_models


def test_init_with_providers_invalid():
    with pytest.raises(ValueError):
        LLMClient({"invalid_provider": "some-key", "openai": "test-key-openai"})


def test_getters(llm_client):
    llm_client.add_provider("openai", "test-key-openai")
    llm_client.add_provider("cohere", "test-key-cohere")
    assert llm_client.get_active_providers() == {"openai", "cohere"}

    active_models = llm_client.get_active_models()
    assert "gpt-4-turbo" in active_models
    assert "command-r" in active_models
    assert "claude-3-opus" not in active_models

    available_providers = LLMClient.get_available_providers()
    assert llm_client.active_providers.issubset(available_providers)
    assert len(available_providers) > len(llm_client.active_providers)

    available_models = LLMClient.get_available_models()
    assert llm_client.active_models.issubset(available_models)
    assert len(available_models) > len(llm_client.active_models)


def test_add_provider(llm_client):
    llm_client.add_provider("openai", "test-key-openai")
    assert "openai" in llm_client.active_providers
    assert "gpt-4-turbo" in llm_client.active_models


def test_add_provider_invalid(llm_client):
    with pytest.raises(ValueError):
        llm_client.add_provider("invalid_provider", "some-key")


def test_remove_provider(llm_client):
    llm_client.add_provider("openai", "test-key-openai")
    llm_client.add_provider("anthropic", "test-key-anthropic")
    llm_client.remove_provider("openai")

    assert "openai" not in llm_client.active_providers
    assert "anthropic" in llm_client.active_providers
    assert "gpt-4-turbo" not in llm_client.active_models
    assert "claude-3-opus" in llm_client.active_models


def test_remove_provider_overlapping_model(llm_client):
    llm_client.add_provider("groq", "test-key-groq")
    llm_client.add_provider("replicate", "test-key-replicate")
    assert "llama3-8b" in llm_client.active_models

    llm_client.remove_provider("groq")
    assert "llama3-8b" in llm_client.active_models


def test_remove_provider_not_active(llm_client):
    with pytest.raises(ValueError):
        llm_client.remove_provider("openai")


def test_set_preferred_provider(llm_client):
    llm_client.set_preferred_providers({"llama3-8b": "groq", "llama3-70b": "replicate"})
    assert llm_client.preferred_providers == {
        "llama3-8b": "groq",
        "llama3-70b": "replicate",
    }
    llm_client.set_preferred_providers({"llama3-8b": "replicate"})
    assert llm_client.preferred_providers == {
        "llama3-8b": "replicate",
        "llama3-70b": "replicate",
    }
    llm_client.set_preferred_providers({"llama3-8b": None})
    assert llm_client.preferred_providers == {
        "llama3-70b": "replicate",
        "llama3-8b": None,
    }


def test_set_preferred_provider_invalid(llm_client):
    with pytest.raises(ValueError):  # Invalid provider
        llm_client.set_preferred_providers({"llama3-8b": "invalid_provider"})

    with pytest.raises(ValueError):  # Invalid model
        llm_client.set_preferred_providers({"invalid_model": "groq"})

    with pytest.raises(ValueError):  # Mismatched model and provider
        llm_client.set_preferred_providers({"llama3-70b": "openai"})


# -- Tests for call -- #


def _generic_test_call(
    llm_client,
    mock_provider,
    call_path,
    response_path,
    provider_key,
    model_name,
):
    mock_client = Mock()

    # Dynamically get the mock call and response objects based on the delimited paths
    mock_call = get_nested_attribute(mock_client, call_path)
    if response_path == "":
        # Stopgap for replicate, TODO fix this!
        mock_call.return_value = ["response"]
    else:
        mock_response = construct_mock_from_path(response_path, "response")
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


@patch(f"{MODULE_PATH}.OpenAI")
def test_call_openai(mock_openai, llm_client):
    _generic_test_call(
        llm_client=llm_client,
        mock_provider=mock_openai,
        call_path="chat.completions.create",
        response_path="choices[0].message.content",
        provider_key="openai",
        model_name="gpt-4-turbo",
    )


@patch(f"{MODULE_PATH}.Anthropic")
def test_call_anthropic(mock_anthropic, llm_client):
    _generic_test_call(
        llm_client=llm_client,
        mock_provider=mock_anthropic,
        call_path="messages.create",
        response_path="content[0].text",
        provider_key="anthropic",
        model_name="claude-3-opus",
    )


@patch(f"{MODULE_PATH}.CohereClient")
def test_call_cohere(mock_cohere, llm_client):
    _generic_test_call(
        llm_client=llm_client,
        mock_provider=mock_cohere,
        call_path="chat",
        response_path="text",
        provider_key="cohere",
        model_name="command-r",
    )


@patch(f"{MODULE_PATH}.Groq")
def test_call_groq(mock_groq, llm_client):
    _generic_test_call(
        llm_client=llm_client,
        mock_provider=mock_groq,
        call_path="chat.completions.create",
        response_path="choices[0].message.content",
        provider_key="groq",
        model_name="llama3-70b",
    )


# Need to test gemini 1.0 and 1.5 separately because 1.0 doesn't support system prompts
@patch(f"{MODULE_PATH}.google.GenerativeModel")
def test_call_google_1_5(mock_google, llm_client):
    _generic_test_call(
        llm_client=llm_client,
        mock_provider=mock_google,
        call_path="generate_content",
        response_path="candidates[0].content.parts[0].text",
        provider_key="google",
        model_name="gemini-1.5-pro",
    )


@patch(f"{MODULE_PATH}.google.GenerativeModel")
def test_call_google_1_0(mock_google, llm_client):
    _generic_test_call(
        llm_client=llm_client,
        mock_provider=mock_google,
        call_path="generate_content",
        response_path="candidates[0].content.parts[0].text",
        provider_key="google",
        model_name="gemini-1.0-pro",
    )


@patch(f"{MODULE_PATH}.replicate.Client")
def test_call_replicate(mock_replicate, llm_client):
    _generic_test_call(
        llm_client=llm_client,
        mock_provider=mock_replicate,
        call_path="run",
        response_path="",
        provider_key="replicate",
        model_name="llama3-8b",
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


@patch(f"{MODULE_PATH}.OpenAI")
def test_call_custom(mock_openai, llm_client):
    mock_client = Mock()
    mock_call = mock_client.chat.completions.create
    mock_response = construct_mock_from_path("choices[0].message.content")
    mock_call.return_value = mock_response
    mock_openai.return_value = mock_client

    llm_client.add_provider("openai", "fake-api-key")
    response_default = llm_client.call_custom(
        provider="openai",
        prompt="Hello",
        model_id="custom-model-xyz",
    )
    response_custom = llm_client.call_custom(
        provider="openai",
        prompt="Hello",
        model_id="custom-model-xyz",
        system_prompt="System prompt",
        temperature=0.5,
        max_tokens=100,
    )

    assert response_default == "response"
    assert response_custom == "response"


def test_call_custom_invalid_provider(llm_client):
    with pytest.raises(ValueError):
        llm_client.call_custom(
            provider="invalid_provider",
            prompt="Hello",
            model_id="custom-model-xyz",
        )


def test_call_custom_not_active(llm_client):
    with pytest.raises(ValueError):
        llm_client.call_custom(
            provider="openai",
            prompt="Hello",
            model_id="custom-model-xyz",
        )


# -- Tests for multi provider -- #


@patch(f"{MODULE_PATH}.Groq")
@patch(f"{MODULE_PATH}.replicate.Client")
def test_multi_provider(mock_replicate, mock_groq, llm_client):
    mock_client_groq = Mock()
    mock_call_groq = mock_client_groq.chat.completions.create
    mock_call_groq.return_value = construct_mock_from_path(
        "choices[0].message.content", final_response="hello from groq"
    )
    mock_groq.return_value = mock_client_groq

    mock_client_replicate = Mock()
    mock_call_replicate = mock_client_replicate.run
    mock_call_replicate.return_value = ["hello from replicate"]
    mock_replicate.return_value = mock_client_replicate

    llm_client.add_provider("groq", "test-key-groq")
    llm_client.add_provider("replicate", "test-key-replicate")
    kwargs = {"prompt": "Hello", "model": "llama3-70b"}
    response_groq = llm_client.call(**kwargs, prefer_provider="groq")
    response_replicate = llm_client.call(**kwargs, prefer_provider="replicate")

    assert response_groq == "hello from groq"
    assert response_replicate == "hello from replicate"


@patch(f"{MODULE_PATH}.Groq")
@patch(f"{MODULE_PATH}.replicate.Client")
def test_multi_provider_with_defaults(mock_replicate, mock_groq, llm_client):
    mock_client_groq = Mock()
    mock_call_groq = mock_client_groq.chat.completions.create
    mock_call_groq.return_value = construct_mock_from_path(
        "choices[0].message.content", final_response="hello from groq"
    )
    mock_groq.return_value = mock_client_groq

    mock_client_replicate = Mock()
    mock_call_replicate = mock_client_replicate.run
    mock_call_replicate.return_value = ["hello from replicate"]
    mock_replicate.return_value = mock_client_replicate

    llm_client.add_provider("groq", "test-key-groq")
    llm_client.add_provider("replicate", "test-key-replicate")
    llm_client.set_preferred_providers({"llama3-70b": "replicate", "llama3-8b": "groq"})

    response_groq = llm_client.call(prompt="Hello", model="llama3-8b")
    response_replicate = llm_client.call(prompt="Hello", model="llama3-70b")

    assert response_groq == "hello from groq"
    assert response_replicate == "hello from replicate"


@patch(f"{MODULE_PATH}.Groq")
def test_multi_provider_one_active(mock_groq, llm_client):
    mock_client_groq = Mock()
    mock_call_groq = mock_client_groq.chat.completions.create
    mock_call_groq.return_value = construct_mock_from_path(
        "choices[0].message.content", final_response="hello from groq"
    )
    mock_groq.return_value = mock_client_groq

    llm_client.add_provider("groq", "test-key-groq")
    response = llm_client.call(prompt="Hello", model="llama3-8b")
    assert response == "hello from groq"


@patch(f"{MODULE_PATH}.Groq")
def test_multi_provider_pref_missing(_, llm_client):
    llm_client.add_provider("groq", "test-key-groq")
    llm_client.add_provider("replicate", "test-key-replicate")

    # Shouldn't raise an error
    llm_client.call(prompt="Hello", model="mixtral-8x7b")

    with pytest.raises(ValueError):
        llm_client.call(prompt="Hello", model="llama3-70b")


def test_multi_provider_pref_inactive(llm_client):
    llm_client.add_provider("groq", "test-key-groq")
    llm_client.add_provider("replicate", "test-key-replicate")
    with pytest.raises(ValueError):
        llm_client.call(prompt="Hello", model="llama3-70b", prefer_provider="openai")
