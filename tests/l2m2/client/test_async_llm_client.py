import pytest
import asyncio
from unittest.mock import patch, Mock

from l2m2.tools.json_mode_strategies import JsonModeStrategy
from test_utils.llm_mock import (
    construct_mock_from_path,
    get_nested_attribute,
)
from l2m2.client import AsyncLLMClient, LLMClient

BASE_MODULE_PATH = "l2m2.client.llm_client"

MOCK_PROVIDER_KEYS = {
    "openai": "test-key-openai",
    "anthropic": "test-key-anthropic",
    "cohere": "test-key-cohere",
    "groq": "test-key-groq",
    "google": "test-key-google",
}


@pytest.fixture
def async_llm_client():
    """Fixture to provide a clean LLMManager instance for each test."""
    return AsyncLLMClient()


# -- Tests for initialization and provider management -- #


def test_init(async_llm_client):
    assert async_llm_client.api_keys == {}
    assert async_llm_client.active_providers == set()
    assert async_llm_client.active_models == set()


def test_init_with_providers():
    async_llm_client = AsyncLLMClient(MOCK_PROVIDER_KEYS)
    assert async_llm_client.api_keys == MOCK_PROVIDER_KEYS
    assert async_llm_client.active_providers == set(MOCK_PROVIDER_KEYS.keys())
    assert "gpt-4-turbo" in async_llm_client.active_models
    assert "command-r" in async_llm_client.active_models
    assert "claude-3-opus" in async_llm_client.active_models


def test_init_with_providers_invalid():
    with pytest.raises(ValueError):
        AsyncLLMClient({"invalid_provider": "some-key", "openai": "test-key-openai"})


def test_init_from_llm_client():
    llm_client = LLMClient(MOCK_PROVIDER_KEYS)
    async_llm_client = AsyncLLMClient.from_client(llm_client)
    assert async_llm_client.api_keys == MOCK_PROVIDER_KEYS
    assert async_llm_client.active_providers == set(MOCK_PROVIDER_KEYS.keys())
    assert "gpt-4-turbo" in async_llm_client.active_models
    assert "command-r" in async_llm_client.active_models
    assert "claude-3-opus" in async_llm_client.active_models


def test_invalid_init():
    with pytest.raises(ValueError):
        AsyncLLMClient(
            {
                "openai": "test-key-openai",
                "cohere": "test-key-cohere",
                "invalid_provider": "some-key",
            }
        )


def test_getters(async_llm_client):
    async_llm_client.add_provider("openai", "test-key-openai")
    async_llm_client.add_provider("cohere", "test-key-cohere")
    assert async_llm_client.get_active_providers() == {"openai", "cohere"}
    active_models = async_llm_client.get_active_models()
    assert "gpt-4-turbo" in active_models
    assert "command-r" in active_models
    assert "claude-3-opus" not in active_models


# -- Tests for call_async -- #


def _get_mock_client(call_path, response_path, response_value="response"):
    mock_client = Mock()
    mock_call = get_nested_attribute(mock_client, call_path)
    mock_response = construct_mock_from_path(response_path, response_value)
    mock_call.return_value = mock_response
    return mock_client


def _prepare_three_mock_clients(mock_openai, mock_anthropic, mock_cohere, responses):
    mock_openai_client = _get_mock_client(
        "chat.completions.create",
        "choices[0].message.content",
        responses[0],
    )
    mock_anthropic_client = _get_mock_client(
        "messages.create",
        "content[0].text",
        responses[1],
    )
    mock_cohere_client = _get_mock_client(
        "chat",
        "text",
        responses[2],
    )
    mock_openai.return_value = mock_openai_client
    mock_anthropic.return_value = mock_anthropic_client
    mock_cohere.return_value = mock_cohere_client


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.OpenAI")
async def test_call_async(mock_openai, async_llm_client):
    mock_openai.return_value = _get_mock_client(
        "chat.completions.create",
        "choices[0].message.content",
        "response",
    )

    async_llm_client.add_provider("openai", "fake-api-key")
    response_default = await async_llm_client.call_async(
        prompt="Hello", model="gpt-4-turbo"
    )
    response_custom = await async_llm_client.call_async(
        prompt="Hello",
        model="gpt-4-turbo",
        system_prompt="System prompt",
        temperature=0.5,
        max_tokens=100,
    )

    assert response_default == "response"
    assert response_custom == "response"


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.OpenAI")
@patch(f"{BASE_MODULE_PATH}.Anthropic")
@patch(f"{BASE_MODULE_PATH}.CohereClient")
async def test_call_async_gather(
    mock_cohere,
    mock_anthropic,
    mock_openai,
    async_llm_client,
):
    _prepare_three_mock_clients(
        mock_openai,
        mock_anthropic,
        mock_cohere,
        ["hello from openai", "hello from anthropic", "hello from cohere"],
    )

    async_llm_client.add_provider("openai", "fake-api-key")
    async_llm_client.add_provider("anthropic", "fake-api-key")
    async_llm_client.add_provider("cohere", "fake-api-key")

    t1 = lambda: async_llm_client.call_async(model="gpt-4-turbo", prompt="Hello")
    t2 = lambda: async_llm_client.call_async(model="claude-3-opus", prompt="Hello")
    t3 = lambda: async_llm_client.call_async(model="command-r", prompt="Hello")

    response1, response2, response3 = await asyncio.gather(t1(), t2(), t3())

    assert response1 == "hello from openai"
    assert response2 == "hello from anthropic"
    assert response3 == "hello from cohere"


# -- Tests for call_concurrent -- #


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.OpenAI")
@patch(f"{BASE_MODULE_PATH}.Anthropic")
@patch(f"{BASE_MODULE_PATH}.CohereClient")
async def test_call_concurrent(
    mock_cohere,
    mock_anthropic,
    mock_openai,
    async_llm_client,
):
    _prepare_three_mock_clients(
        mock_openai,
        mock_anthropic,
        mock_cohere,
        ["hello from openai", "hello from anthropic", "hello from cohere"],
    )

    async_llm_client.add_provider("openai", "fake-api-key")
    async_llm_client.add_provider("anthropic", "fake-api-key")
    async_llm_client.add_provider("cohere", "fake-api-key")

    response = await async_llm_client.call_concurrent(
        n=3,
        models=["gpt-4-turbo", "claude-3-opus", "command-r"],
        prompts=["Hello", "Hello", "Hello"],
    )

    assert response == [
        "hello from openai",
        "hello from anthropic",
        "hello from cohere",
    ]


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.OpenAI")
@patch(f"{BASE_MODULE_PATH}.Anthropic")
@patch(f"{BASE_MODULE_PATH}.CohereClient")
async def test_call_concurrent_1_to_3(
    mock_cohere,
    mock_anthropic,
    mock_openai,
    async_llm_client,
):
    _prepare_three_mock_clients(
        mock_openai,
        mock_anthropic,
        mock_cohere,
        ["hello from openai", "hello from anthropic", "hello from cohere"],
    )

    async_llm_client.add_provider("openai", "fake-api-key")
    async_llm_client.add_provider("anthropic", "fake-api-key")
    async_llm_client.add_provider("cohere", "fake-api-key")

    response = await async_llm_client.call_concurrent(
        n=3,
        models=["gpt-4-turbo"],
        prompts=["Hello", "Hello", "Hello"],
    )

    assert response == ["hello from openai"] * 3


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.OpenAI")
@patch(f"{BASE_MODULE_PATH}.Anthropic")
@patch(f"{BASE_MODULE_PATH}.CohereClient")
async def test_call_concurrent_3_to_1(
    mock_cohere,
    mock_anthropic,
    mock_openai,
    async_llm_client,
):
    _prepare_three_mock_clients(
        mock_openai,
        mock_anthropic,
        mock_cohere,
        ["hello from openai", "hello from anthropic", "hello from cohere"],
    )

    async_llm_client.add_provider("openai", "fake-api-key")
    async_llm_client.add_provider("anthropic", "fake-api-key")
    async_llm_client.add_provider("cohere", "fake-api-key")

    response = await async_llm_client.call_concurrent(
        n=3,
        models=["gpt-4-turbo", "claude-3-opus", "command-r"],
        prompts=["Hello"],
    )

    assert response == [
        "hello from openai",
        "hello from anthropic",
        "hello from cohere",
    ]


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.OpenAI")
@patch(f"{BASE_MODULE_PATH}.Anthropic")
@patch(f"{BASE_MODULE_PATH}.CohereClient")
async def test_call_concurrent_custom_params(
    mock_cohere,
    mock_anthropic,
    mock_openai,
    async_llm_client,
):
    _prepare_three_mock_clients(
        mock_openai,
        mock_anthropic,
        mock_cohere,
        ["hello from openai", "hello from anthropic", "hello from cohere"],
    )

    async_llm_client.add_provider("openai", "fake-api-key")
    async_llm_client.add_provider("anthropic", "fake-api-key")
    async_llm_client.add_provider("cohere", "fake-api-key")

    response = await async_llm_client.call_concurrent(
        n=3,
        models=["gpt-4-turbo", "claude-3-opus", "command-r"],
        prompts=["Hello a", "Hello b", "Hello c"],
        system_prompts=["System a", "System b", "System c"],
        temperatures=[0.1, 0.2, 0.3],
        max_tokens=[10, 20, 30],
    )

    assert response == [
        "hello from openai",
        "hello from anthropic",
        "hello from cohere",
    ]


@pytest.mark.asyncio
async def test_call_concurrent_bad_n(async_llm_client):
    with pytest.raises(ValueError):
        await async_llm_client.call_concurrent(
            n=0,
            models=["gpt-4-turbo", "claude-3-opus", "command-r"],
            prompts=["Hello", "Hello", "Hello"],
        )


@pytest.mark.asyncio
async def test_call_concurrent_wrong_n(async_llm_client):
    with pytest.raises(ValueError):
        await async_llm_client.call_concurrent(
            n=4,
            models=["gpt-4-turbo", "claude-3-opus", "command-r"],
            prompts=["Hello", "Hello", "Hello"],
        )


@pytest.mark.asyncio
async def test_call_concurrent_array_mismatch(async_llm_client):
    with pytest.raises(ValueError):
        await async_llm_client.call_concurrent(
            n=3,
            models=["gpt-4-turbo", "claude-3-opus", "command-r"],
            prompts=["Hello", "Hello"],
        )


# -- Tests for call_custom_async and call_custon_concurrent -- #


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.OpenAI")
async def test_call_custom_async(mock_openai, async_llm_client):
    mock_openai.return_value = _get_mock_client(
        "chat.completions.create",
        "choices[0].message.content",
        "response",
    )

    async_llm_client.add_provider("openai", "fake-api-key")
    response_default = await async_llm_client.call_custom_async(
        provider="openai",
        prompt="Hello",
        model_id="custom-model-xyz",
    )
    response_custom = await async_llm_client.call_custom_async(
        provider="openai",
        prompt="Hello",
        model_id="custom-model-xyz",
        system_prompt="System prompt",
        temperature=0.5,
        max_tokens=100,
    )

    assert response_default == "response"
    assert response_custom == "response"


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.OpenAI")
@patch(f"{BASE_MODULE_PATH}.Anthropic")
@patch(f"{BASE_MODULE_PATH}.CohereClient")
async def test_call_custom_concurrent(
    mock_cohere,
    mock_anthropic,
    mock_openai,
    async_llm_client,
):
    _prepare_three_mock_clients(
        mock_openai,
        mock_anthropic,
        mock_cohere,
        ["hello from openai", "hello from anthropic", "hello from cohere"],
    )

    async_llm_client.add_provider("openai", "fake-api-key")
    async_llm_client.add_provider("anthropic", "fake-api-key")
    async_llm_client.add_provider("cohere", "fake-api-key")

    response = await async_llm_client.call_custom_concurrent(
        n=3,
        providers=["openai", "anthropic", "cohere"],
        model_ids=["custom-x", "custom-y", "custom-z"],
        prompts=["Hello a", "Hello b", "Hello c"],
        system_prompts=["System a", "System b", "System c"],
        temperatures=[0.1, 0.2, 0.3],
        max_tokens=[10, 20, 30],
    )

    assert response == [
        "hello from openai",
        "hello from anthropic",
        "hello from cohere",
    ]


# -- Tests for JSON mode -- #


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.OpenAI")
async def test_json_mode_native(mock_openai, async_llm_client):
    mock_openai.return_value = _get_mock_client(
        "chat.completions.create",
        "choices[0].message.content",
        "response",
    )

    async_llm_client.add_provider("openai", "fake-api-key")
    response = await async_llm_client.call_async(
        prompt="Hello",
        model="gpt-4-turbo",
        json_mode=True,
    )

    assert response == "response"


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.Anthropic")
async def test_json_mode_strategy_strip(mock_anthropic, async_llm_client):
    mock_anthropic.return_value = _get_mock_client(
        "messages.create",
        "content[0].text",
        "--{response}--",
    )

    async_llm_client.add_provider("anthropic", "fake-api-key")
    response = await async_llm_client.call_async(
        prompt="Hello",
        model="claude-3-opus",
        json_mode=True,
        json_mode_strategy=JsonModeStrategy.strip(),
    )

    assert response == "{response}"


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.Anthropic")
async def test_json_mode_strategy_prepend(mock_anthropic, async_llm_client):
    mock_anthropic.return_value = _get_mock_client(
        "messages.create",
        "content[0].text",
        "response",
    )

    async_llm_client.add_provider("anthropic", "fake-api-key")
    response = await async_llm_client.call_async(
        prompt="Hello",
        model="claude-3-opus",
        json_mode=True,
        json_mode_strategy=JsonModeStrategy.prepend(),
    )

    assert response == "{response"


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.Anthropic")
@patch(f"{BASE_MODULE_PATH}.CohereClient")
async def test_json_mode_call_concurrent_default_strategy(
    mock_cohere,
    mock_anthropic,
    async_llm_client,
):
    _prepare_three_mock_clients(
        mock_anthropic,
        mock_anthropic,
        mock_cohere,
        ["--{response}--", "--{response}--", "--{response}--"],
    )

    async_llm_client.add_provider("anthropic", "fake-api-key")
    async_llm_client.add_provider("cohere", "fake-api-key")

    response = await async_llm_client.call_concurrent(
        n=3,
        models=["claude-3-opus", "claude-3-opus", "command-r"],
        prompts=["Hello"],
        json_modes=[True, False, True],
    )

    assert response == ["{response}", "--{response}--", "{response}"]


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.Anthropic")
@patch(f"{BASE_MODULE_PATH}.CohereClient")
async def test_json_mode_call_concurrent_prepend_strategy(
    mock_cohere,
    mock_anthropic,
    async_llm_client,
):
    _prepare_three_mock_clients(
        mock_anthropic,
        mock_anthropic,
        mock_cohere,
        ["response", "response", "response"],
    )

    async_llm_client.add_provider("anthropic", "fake-api-key")
    async_llm_client.add_provider("cohere", "fake-api-key")

    response = await async_llm_client.call_concurrent(
        n=3,
        models=["claude-3-opus", "claude-3-opus", "command-r"],
        prompts=["Hello", "Hello", "Hello"],
        json_modes=[True, False, True],
        json_mode_strategies=[JsonModeStrategy.prepend()],
    )

    assert response == ["{response", "response", "{response"]


@pytest.mark.asyncio
@patch(f"{BASE_MODULE_PATH}.Anthropic")
@patch(f"{BASE_MODULE_PATH}.CohereClient")
async def test_json_mode_call_concurrent_mixed_strategies(
    mock_cohere,
    mock_anthropic,
    async_llm_client,
):
    _prepare_three_mock_clients(
        mock_anthropic,
        mock_anthropic,
        mock_cohere,
        ["--{response}--", "--{response}--", "--{response}--"],
    )

    async_llm_client.add_provider("anthropic", "fake-api-key")
    async_llm_client.add_provider("cohere", "fake-api-key")

    response = await async_llm_client.call_concurrent(
        n=3,
        models=["claude-3-opus", "claude-3-opus", "command-r"],
        prompts=["Hello", "Hello", "Hello"],
        json_modes=[True, True, True],
        json_mode_strategies=[
            JsonModeStrategy.strip(),
            JsonModeStrategy.prepend(),
            JsonModeStrategy.strip(),
        ],
    )

    assert response == ["{response}", "{--{response}--", "{response}"]
