import pytest
from unittest.mock import patch, Mock, MagicMock
from l2m2.llm_client import LLMClient

# This ensures all providers are available for import.
from openai import OpenAI  # noqa: F401
from cohere import Client as CohereClient  # noqa: F401
from anthropic import Anthropic  # noqa: F401
from groq import Groq  # noqa: F401


@pytest.fixture
def llm_client():
    """Fixture to provide a clean LLMManager instance for each test."""
    return LLMClient()


# -- Tests for initialization and provider management -- #


def test_initialization(llm_client):
    assert llm_client.API_KEYS == {}
    assert llm_client.active_providers == set()
    assert llm_client.active_models == set()


def test_add_provider_valid(llm_client):
    with patch.object(
        LLMClient, "get_available_providers", return_value={"openai", "cohere"}
    ):
        llm_client.add_provider("openai", "test-key-openai")
        assert "openai" in llm_client.active_providers
        assert "gpt-4-turbo" in llm_client.active_models


def test_add_provider_invalid(llm_client):
    with pytest.raises(ValueError):
        llm_client.add_provider("invalid_provider", "some-key")


def test_remove_provider(llm_client):
    with patch.object(LLMClient, "get_available_providers", return_value={"openai"}):
        llm_client.add_provider("openai", "test-key-openai")
        llm_client.remove_provider("openai")
        assert "openai" not in llm_client.active_providers
        assert "gpt-4-turbo" not in llm_client.active_models


def test_remove_provider_not_active(llm_client):
    with pytest.raises(ValueError):
        llm_client.remove_provider("openai")


# -- Tests for call -- #


def test_call_with_openai(llm_client):
    with patch("l2m2.llm_client.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="response"))]
        )

        llm_client.API_KEYS = {"openai": "fake-key"}
        llm_client.active_models = {"gpt-4-turbo"}

        response = llm_client.call(prompt="Hello", model="gpt-4-turbo")
        assert response == "response"


def test_call_with_cohere(llm_client):
    with patch("l2m2.llm_client.CohereClient") as mock_cohere:
        mock_client = Mock()
        mock_cohere.return_value = mock_client
        mock_client.chat.return_value = Mock(text="response")

        llm_client.API_KEYS = {"cohere": "fake-key"}
        llm_client.active_models = {"command-r"}

        response = llm_client.call(prompt="Hello", model="command-r")
        assert response == "response"


def test_call_with_google_1_5(llm_client):
    with patch("l2m2.llm_client.google.GenerativeModel") as mock_model:
        mock_instance = mock_model.return_value
        mock_response = MagicMock()
        mock_response.candidates = [
            MagicMock(content=MagicMock(parts=[MagicMock(text="response")]))
        ]
        mock_instance.generate_content.return_value = mock_response

        llm_client.API_KEYS = {"google": "fake-key"}
        llm_client.active_models = {"gemini-1.5-pro"}

        response = llm_client.call(
            prompt="Hello",
            model="gemini-1.5-pro",
            system_prompt="Respond as if you were a pirate.",
        )
        assert response == "response"


def test_call_with_google_1_0(llm_client):
    with patch("l2m2.llm_client.google.GenerativeModel") as mock_model:
        mock_instance = mock_model.return_value
        mock_response = MagicMock()
        mock_response.candidates = [
            MagicMock(content=MagicMock(parts=[MagicMock(text="response")]))
        ]
        mock_instance.generate_content.return_value = mock_response

        llm_client.API_KEYS = {"google": "fake-key"}
        llm_client.active_models = {"gemini-1.0-pro"}

        response = llm_client.call(
            prompt="Hello",
            model="gemini-1.0-pro",
            system_prompt="Respond as if you were a pirate.",
        )
        assert response == "response"


def test_call_invalid_model(llm_client):
    with pytest.raises(ValueError):
        llm_client.call(prompt="Hello", model="unknown-model")
