import pytest
import pytest_asyncio
from unittest.mock import patch

from l2m2.memory import (
    ChatMemory,
    ExternalMemory,
    ExternalMemoryLoadingType,
)
from l2m2.client.base_llm_client import BaseLLMClient
from l2m2.tools import JsonModeStrategy
from l2m2.exceptions import LLMOperationError

LLM_POST_PATH = "l2m2.client.base_llm_client.llm_post"
GET_EXTRA_MESSAGE_PATH = "l2m2.client.base_llm_client.get_extra_message"
CALL_BASE_PATH = "l2m2.client.base_llm_client.BaseLLMClient._call_"

# Model/provider pairs which don't support ChatMemory
CHAT_MEMORY_UNSUPPORTED_MODELS = {
    "replicate": "llama-3-8b",  # Applies to all models via Replicate
}


# -- Fixtures -- #


@pytest_asyncio.fixture
async def llm_client():
    async with BaseLLMClient() as b:
        yield b


@pytest_asyncio.fixture
async def llm_client_mem_chat():
    async with BaseLLMClient(memory=ChatMemory()) as b:
        yield b


@pytest_asyncio.fixture
async def llm_client_mem_ext_sys():
    async with BaseLLMClient(
        memory=ExternalMemory(),  # Default is SYSTEM_PROMPT_APPEND
    ) as b:
        yield b


@pytest_asyncio.fixture
async def llm_client_mem_ext_usr():
    async with BaseLLMClient(
        memory=ExternalMemory(
            loading_type=ExternalMemoryLoadingType.USER_PROMPT_APPEND
        ),
    ) as b:
        yield b


# -- Tests for initialization and provider management -- #


def test_init(llm_client):
    assert llm_client.api_keys == {}
    assert llm_client.active_providers == set()
    assert llm_client.active_models == set()


@pytest.mark.asyncio
async def test_init_with_providers():
    async with BaseLLMClient(
        {"openai": "test-key-openai", "cohere": "test-key-cohere"}
    ) as llm_client:
        assert llm_client.api_keys == {
            "openai": "test-key-openai",
            "cohere": "test-key-cohere",
        }
        assert llm_client.active_providers == {"openai", "cohere"}
        assert "gpt-4-turbo" in llm_client.active_models
        assert "command-r" in llm_client.active_models
        assert "claude-3-opus" not in llm_client.active_models


@pytest.mark.asyncio
@patch.dict(
    "os.environ", {"OPENAI_API_KEY": "test-key-openai", "CO_API_KEY": "test-key-cohere"}
)
async def test_init_with_env_providers():
    async with BaseLLMClient() as llm_client:
        assert llm_client.api_keys == {
            "openai": "test-key-openai",
            "cohere": "test-key-cohere",
        }
        assert llm_client.active_providers == {"openai", "cohere"}
        assert "gpt-4-turbo" in llm_client.active_models
        assert "command-r" in llm_client.active_models
        assert "claude-3-opus" not in llm_client.active_models


@pytest.mark.asyncio
@patch.dict(
    "os.environ", {"OPENAI_API_KEY": "env-key-openai", "CO_API_KEY": "env-key-cohere"}
)
async def test_init_with_env_providers_override():
    async with BaseLLMClient(
        {
            "openai": "override-key-openai",
            "anthropic": "new-key-anthropic",
        }
    ) as llm_client:
        assert llm_client.api_keys == {
            "openai": "override-key-openai",
            "cohere": "env-key-cohere",
            "anthropic": "new-key-anthropic",
        }
        assert llm_client.active_providers == {"openai", "cohere", "anthropic"}
        assert "gpt-4-turbo" in llm_client.active_models
        assert "command-r" in llm_client.active_models
        assert "claude-3-opus" in llm_client.active_models


def test_init_with_providers_invalid():
    with pytest.raises(ValueError):
        BaseLLMClient({"invalid_provider": "some-key", "openai": "test-key-openai"})


def test_getters(llm_client):
    llm_client.add_provider("openai", "test-key-openai")
    llm_client.add_provider("cohere", "test-key-cohere")
    assert llm_client.get_active_providers() == {"openai", "cohere"}

    active_models = llm_client.get_active_models()
    assert "gpt-4-turbo" in active_models
    assert "command-r" in active_models
    assert "claude-3-opus" not in active_models

    available_providers = BaseLLMClient.get_available_providers()
    assert llm_client.active_providers.issubset(available_providers)
    assert len(available_providers) > len(llm_client.active_providers)

    available_models = BaseLLMClient.get_available_models()
    assert llm_client.active_models.issubset(available_models)
    assert len(available_models) > len(llm_client.active_models)


def test_add_provider(llm_client):
    llm_client.add_provider("openai", "test-key-openai")
    assert "openai" in llm_client.active_providers
    assert "gpt-4-turbo" in llm_client.active_models


def test_add_provider_invalid(llm_client):
    with pytest.raises(ValueError):
        llm_client.add_provider("invalid_provider", "some-key")


def test_add_provider_bad_key(llm_client):
    with pytest.raises(ValueError):
        llm_client.add_provider("openai", None)
    with pytest.raises(ValueError):
        llm_client.add_provider("openai", 123)


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
    assert "llama-3-8b" in llm_client.active_models

    llm_client.remove_provider("groq")
    assert "llama-3-8b" in llm_client.active_models


def test_remove_provider_not_active(llm_client):
    with pytest.raises(ValueError):
        llm_client.remove_provider("openai")


def test_set_preferred_provider(llm_client):
    llm_client.set_preferred_providers(
        {"llama-3-8b": "groq", "llama-3-70b": "replicate"}
    )
    assert llm_client.preferred_providers == {
        "llama-3-8b": "groq",
        "llama-3-70b": "replicate",
    }
    llm_client.set_preferred_providers({"llama-3-8b": "replicate"})
    assert llm_client.preferred_providers == {
        "llama-3-8b": "replicate",
        "llama-3-70b": "replicate",
    }
    llm_client.set_preferred_providers({"llama-3-8b": None})
    assert llm_client.preferred_providers == {
        "llama-3-70b": "replicate",
        "llama-3-8b": None,
    }


def test_set_preferred_provider_invalid(llm_client):
    with pytest.raises(ValueError):  # Invalid provider
        llm_client.set_preferred_providers({"llama-3-8b": "invalid_provider"})

    with pytest.raises(ValueError):  # Invalid model
        llm_client.set_preferred_providers({"invalid_model": "groq"})

    with pytest.raises(ValueError):  # Mismatched model and provider
        llm_client.set_preferred_providers({"llama-3-70b": "openai"})


# -- Tests for call -- #


async def _generic_test_call(
    llm_client,
    provider_key,
    model_name,
    response="response",
):
    if provider_key != "replicate":
        # ChatMemory behaves differently with each provider (except replicate where it's
        # not supported), so load it here to test each separate implementation.
        llm_client.load_memory(ChatMemory())

    llm_client.add_provider(provider_key, "fake-api-key")
    response_default = await llm_client.call(prompt="Hello", model=model_name)
    response_custom = await llm_client.call(
        prompt="Hello",
        model=model_name,
        system_prompt="System prompt",
        temperature=0.5,
        max_tokens=100,
        json_mode=True,
        # Just passing this in here to make sure nothing breaks - we'll test the actual
        # functionality later on
        json_mode_strategy=JsonModeStrategy.strip(),
    )

    assert response_default == response
    assert response_custom == response


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_openai(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"choices": [{"message": {"content": "response"}}]}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "openai", "gpt-4-turbo")


# Need to test gemini 1.0 and 1.5 separately because of different system prompt handling
@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_google_1_5(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"candidates": [{"content": {"parts": [{"text": "response"}]}}]}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "google", "gemini-1.5-pro")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_google_1_0(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"candidates": [{"content": {"parts": [{"text": "response"}]}}]}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "google", "gemini-1.0-pro")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_anthropic(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"content": [{"text": "response"}]}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "anthropic", "claude-3-opus")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_cohere(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"text": "response"}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "cohere", "command-r")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_mistral(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"choices": [{"message": {"content": "response"}}]}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "mistral", "mistral-large")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_groq(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"choices": [{"message": {"content": "response"}}]}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "groq", "llama-3-70b")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_replicate(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"output": ["response"]}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "replicate", "llama-3-8b")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_cerebras(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"choices": [{"message": {"content": "response"}}]}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "cerebras", "llama-3.1-70b")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_call_google_gemini_fails(mock_llm_post, llm_client):
    llm_client.add_provider("google", "fake-api-key")
    mock_return_value = {"candidates": [{"error": "123"}]}
    mock_llm_post.return_value = mock_return_value
    response = await llm_client.call(prompt="Hello", model="gemini-1.5-pro")
    assert response == "{'error': '123'}"


@pytest.mark.asyncio
async def test_call_valid_model_not_active(llm_client):
    with pytest.raises(ValueError):
        await llm_client.call(prompt="Hello", model="gpt-4-turbo")


@pytest.mark.asyncio
async def test_call_invalid_model(llm_client):
    with pytest.raises(ValueError):
        await llm_client.call(prompt="Hello", model="unknown-model")


@pytest.mark.asyncio
async def test_call_tokens_too_large(llm_client):
    llm_client.add_provider("openai", "fake-api-key")
    with pytest.raises(ValueError):
        await llm_client.call(
            prompt="Hello", model="gpt-4-turbo", max_tokens=float("inf")
        )


@pytest.mark.asyncio
async def test_call_temperature_too_high(llm_client):
    llm_client.add_provider("openai", "fake-api-key")
    with pytest.raises(ValueError):
        await llm_client.call(prompt="Hello", model="gpt-4-turbo", temperature=3.0)


# -- Tests for call_custom -- #


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_call_custom(mock_call_openai, llm_client):
    mock_call_openai.return_value = {"choices": [{"message": {"content": "response"}}]}
    llm_client.add_provider("openai", "fake-api-key")
    response_default = await llm_client.call_custom(
        provider="openai",
        prompt="Hello",
        model_id="custom-model-xyz",
    )
    response_custom = await llm_client.call_custom(
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
async def test_call_custom_invalid_provider(llm_client):
    with pytest.raises(ValueError):
        await llm_client.call_custom(
            provider="invalid_provider",
            prompt="Hello",
            model_id="custom-model-xyz",
        )


@pytest.mark.asyncio
async def test_call_custom_not_active(llm_client):
    with pytest.raises(ValueError):
        await llm_client.call_custom(
            provider="openai",
            prompt="Hello",
            model_id="custom-model-xyz",
        )


# -- Tests for multi provider -- #


@pytest.mark.asyncio
@patch(f"{CALL_BASE_PATH}groq")
@patch(f"{CALL_BASE_PATH}replicate")
async def test_multi_provider(mock_call_replicate, mock_call_groq, llm_client):
    mock_call_groq.return_value = "hello from groq"
    mock_call_replicate.return_value = "hello from replicate"

    llm_client.add_provider("groq", "test-key-groq")
    llm_client.add_provider("replicate", "test-key-replicate")

    kwargs = {"prompt": "Hello", "model": "llama-3-70b"}
    response_groq = await llm_client.call(**kwargs, prefer_provider="groq")
    response_replicate = await llm_client.call(**kwargs, prefer_provider="replicate")
    assert response_groq == "hello from groq"
    assert response_replicate == "hello from replicate"


@pytest.mark.asyncio
@patch(f"{CALL_BASE_PATH}groq")
@patch(f"{CALL_BASE_PATH}replicate")
async def test_multi_provider_with_defaults(
    mock_call_replicate, mock_call_groq, llm_client
):
    mock_call_groq.return_value = "hello from groq"
    mock_call_replicate.return_value = "hello from replicate"

    llm_client.add_provider("groq", "test-key-groq")
    llm_client.add_provider("replicate", "test-key-replicate")
    llm_client.set_preferred_providers({"llama-3-8b": "groq"})
    llm_client.set_preferred_providers({"llama-3-70b": "replicate"})

    response_groq = await llm_client.call(prompt="Hello", model="llama-3-8b")
    response_replicate = await llm_client.call(prompt="Hello", model="llama-3-70b")
    assert response_groq == "hello from groq"
    assert response_replicate == "hello from replicate"


@pytest.mark.asyncio
@patch(f"{CALL_BASE_PATH}groq")
async def test_multi_provider_one_active(mock_call_groq, llm_client):
    mock_call_groq.return_value = "hello from groq"
    llm_client.add_provider("groq", "test-key-groq")
    response = await llm_client.call(prompt="Hello", model="llama-3-8b")
    assert response == "hello from groq"


@pytest.mark.asyncio
@patch(f"{CALL_BASE_PATH}groq")
async def test_multi_provider_pref_missing(_, llm_client):
    llm_client.add_provider("groq", "test-key-groq")
    llm_client.add_provider("replicate", "test-key-replicate")

    # Shouldn't raise an error
    await llm_client.call(prompt="Hello", model="mixtral-8x7b")

    with pytest.raises(ValueError):
        await llm_client.call(prompt="Hello", model="llama-3-70b")


@pytest.mark.asyncio
async def test_multi_provider_pref_inactive(llm_client):
    llm_client.add_provider("groq", "test-key-groq")
    llm_client.add_provider("replicate", "test-key-replicate")
    with pytest.raises(ValueError):
        await llm_client.call(
            prompt="Hello", model="llama-3-70b", prefer_provider="openai"
        )


# -- Tests for memory -- #


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_chat_memory(mock_call_openai, llm_client_mem_chat):
    mock_call_openai.return_value = {"choices": [{"message": {"content": "response"}}]}

    llm_client_mem_chat.add_provider("openai", "fake-api-key")

    memory = llm_client_mem_chat.get_memory()
    assert isinstance(memory, ChatMemory)

    memory.add_user_message("A")
    memory.add_agent_message("B")

    response = await llm_client_mem_chat.call(prompt="C", model="gpt-4-turbo")
    assert response == "response"
    assert memory.unpack("role", "content", "user", "assistant") == [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "response"},
    ]

    llm_client_mem_chat.clear_memory()
    assert (
        llm_client_mem_chat.get_memory().unpack("role", "content", "user", "assistant")
        == []
    )


def test_chat_memory_errors(llm_client):
    with pytest.raises(ValueError):
        llm_client.get_memory()

    with pytest.raises(ValueError):
        llm_client.clear_memory()


@pytest.mark.asyncio
async def test_chat_memory_unsupported_provider(llm_client_mem_chat):
    for provider, model in CHAT_MEMORY_UNSUPPORTED_MODELS.items():
        llm_client_mem_chat.add_provider(provider, "fake-api-key")
        with pytest.raises(LLMOperationError):
            await llm_client_mem_chat.call(prompt="Hello", model=model)


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_external_memory_system_prompt(mock_call_openai, llm_client_mem_ext_sys):
    mock_call_openai.return_value = {"choices": [{"message": {"content": "response"}}]}
    llm_client_mem_ext_sys.add_provider("openai", "fake-api-key")

    memory = llm_client_mem_ext_sys.get_memory()
    assert isinstance(memory, ExternalMemory)

    memory.set_contents("stuff")

    await llm_client_mem_ext_sys.call(prompt="Hello", model="gpt-4-turbo")
    assert mock_call_openai.call_args.kwargs["data"]["messages"] == [
        {"role": "system", "content": "stuff"},
        {"role": "user", "content": "Hello"},
    ]

    await llm_client_mem_ext_sys.call(
        system_prompt="system-123", prompt="Hello", model="gpt-4-turbo"
    )
    assert mock_call_openai.call_args.kwargs["data"]["messages"] == [
        {"role": "system", "content": "system-123\nstuff"},
        {"role": "user", "content": "Hello"},
    ]


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_external_memory_user_prompt(mock_call_openai, llm_client_mem_ext_usr):
    mock_call_openai.return_value = {"choices": [{"message": {"content": "response"}}]}
    llm_client_mem_ext_usr.add_provider("openai", "fake-api-key")

    memory = llm_client_mem_ext_usr.get_memory()
    assert isinstance(memory, ExternalMemory)

    memory.set_contents("stuff")

    await llm_client_mem_ext_usr.call(prompt="Hello", model="gpt-4-turbo")
    assert mock_call_openai.call_args.kwargs["data"]["messages"] == [
        {"role": "user", "content": "Hello\nstuff"},
    ]

    await llm_client_mem_ext_usr.call(
        system_prompt="system-123", prompt="Hello", model="gpt-4-turbo"
    )
    assert mock_call_openai.call_args.kwargs["data"]["messages"] == [
        {"role": "system", "content": "system-123"},
        {"role": "user", "content": "Hello\nstuff"},
    ]


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_bypass_memory(mock_call_openai, llm_client_mem_chat):
    mock_call_openai.return_value = {"choices": [{"message": {"content": "response"}}]}
    llm_client_mem_chat.add_provider("openai", "fake-api-key")
    llm_client_mem_chat.get_memory().add_user_message("A")
    llm_client_mem_chat.get_memory().add_agent_message("B")

    await llm_client_mem_chat.call(prompt="Hello", model="gpt-4o", bypass_memory=True)
    assert mock_call_openai.call_args.kwargs["data"]["messages"] == [
        {"role": "user", "content": "Hello"},
    ]
    assert llm_client_mem_chat.get_memory().unpack(
        "role", "content", "user", "assistant"
    ) == [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]

    await llm_client_mem_chat.call(prompt="Hello", model="gpt-4o")
    assert mock_call_openai.call_args.kwargs["data"]["messages"] == [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "Hello"},
    ]
    assert llm_client_mem_chat.get_memory().unpack(
        "role", "content", "user", "assistant"
    ) == [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "response"},
    ]


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_alt_memory(mock_call_openai, llm_client):
    mock_call_openai.return_value = {"choices": [{"message": {"content": "response"}}]}
    llm_client.add_provider("openai", "fake-api-key")

    m1 = ChatMemory()
    m2 = ChatMemory()
    llm_client.load_memory(ChatMemory())

    await llm_client.call(prompt="A", model="gpt-4-turbo", alt_memory=m1)
    await llm_client.call(prompt="X", model="gpt-4-turbo", alt_memory=m2)
    await llm_client.call(prompt="B", model="gpt-4-turbo", alt_memory=m1)
    await llm_client.call(prompt="Y", model="gpt-4-turbo", alt_memory=m2)
    await llm_client.call(prompt="C", model="gpt-4-turbo", alt_memory=m1)
    await llm_client.call(prompt="Z", model="gpt-4-turbo", alt_memory=m2)

    assert m1.unpack("role", "content", "user", "assistant") == [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "response"},
    ]

    assert m2.unpack("role", "content", "user", "assistant") == [
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "Y"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "Z"},
        {"role": "assistant", "content": "response"},
    ]

    assert llm_client.get_memory().unpack("role", "content", "user", "assistant") == []


# -- Test for non-native JSON mode default strategy (strip for all but Anthropic) -- #


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_json_mode_default_strategy_strip(mock_call, llm_client):

    # Cohere
    mock_call.return_value = {"text": "--{response}--"}
    llm_client.add_provider("cohere", "fake-api-key")
    response = await llm_client.call(
        prompt="Hello",
        model="command-r",
        json_mode=True,
    )
    assert response == "{response}"

    # Groq
    mock_call.return_value = {"choices": [{"message": {"content": "--{response}--"}}]}
    llm_client.add_provider("groq", "fake-api-key")
    response = await llm_client.call(
        prompt="Hello",
        model="llama-3-70b",
        json_mode=True,
    )
    assert response == "{response}"


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_json_mode_default_strategy_prepend(mock_call_anthropic, llm_client):
    mock_call_anthropic.return_value = {"content": [{"text": "response"}]}
    llm_client.add_provider("anthropic", "fake-api-key")
    response = await llm_client.call(
        prompt="Hello",
        model="claude-3-opus",
        json_mode=True,
    )
    assert response == "{response"


# -- Tests for non-native JSON mode strategy STRIP -- #


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_json_mode_strategy_strip(mock_call_anthropic, llm_client):
    mock_call_anthropic.return_value = {"content": [{"text": "--{response}--"}]}
    llm_client.add_provider("anthropic", "fake-api-key")
    response = await llm_client.call(
        prompt="Hello",
        model="claude-3-opus",
        json_mode=True,
        json_mode_strategy=JsonModeStrategy.strip(),
    )

    assert response == "{response}"


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_json_mode_strategy_strip_invalid_aborts(mock_call_anthropic, llm_client):
    mock_call_anthropic.return_value = {"content": [{"text": "--}response{--"}]}
    llm_client.add_provider("anthropic", "fake-api-key")
    response = await llm_client.call(
        prompt="Hello",
        model="claude-3-opus",
        json_mode=True,
        json_mode_strategy=JsonModeStrategy.strip(),
    )

    assert response == "--}response{--"


# -- Tests for non-native JSON mode strategy PREPEND -- #

# Note: We need to test each applicable provider separately because prepend's
# implementation is provider-specific


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_json_mode_strategy_prepend_anthropic(mock_call_anthropic, llm_client):
    mock_call_anthropic.return_value = {"content": [{"text": "response"}]}
    llm_client.add_provider("anthropic", "fake-api-key")
    response = await llm_client.call(
        prompt="Hello",
        model="claude-3-opus",
        json_mode=True,
        json_mode_strategy=JsonModeStrategy.prepend(),
    )

    assert response == "{response"
    assert (
        mock_call_anthropic.call_args.kwargs["data"]["messages"][-1]["content"]
        == "Here is the JSON response: {"
    )


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_json_mode_strategy_prepend_cohere(mock_call_cohere, llm_client):
    mock_call_cohere.return_value = {"text": "response"}
    llm_client.add_provider("cohere", "fake-api-key")
    response = await llm_client.call(
        prompt="Hello",
        model="command-r",
        json_mode=True,
        json_mode_strategy=JsonModeStrategy.prepend(),
    )

    assert response == "{response"
    assert (
        mock_call_cohere.call_args.kwargs["data"]["chat_history"][-1]["message"]
        == "Here is the JSON response: {"
    )


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_json_mode_strategy_prepend_groq(mock_call_groq, llm_client):
    mock_call_groq.return_value = {"choices": [{"message": {"content": "response"}}]}
    llm_client.add_provider("groq", "fake-api-key")
    response = await llm_client.call(
        prompt="Hello",
        model="llama-3-70b",
        json_mode=True,
        json_mode_strategy=JsonModeStrategy.prepend(),
    )

    assert response == "{response"
    assert (
        mock_call_groq.call_args.kwargs["data"]["messages"][-1]["content"]
        == "Here is the JSON response: {"
    )


@pytest.mark.asyncio
async def test_json_mode_strategy_prepend_replicate_throws_error(llm_client):
    llm_client.add_provider("replicate", "fake-api-key")
    with pytest.raises(LLMOperationError):
        await llm_client.call(
            prompt="Hello",
            model="llama-3-8b",
            json_mode=True,
            json_mode_strategy=JsonModeStrategy.prepend(),
        )


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_json_mode_strategy_prepend_custom_prefix_anthropic(
    mock_call_anthropic, llm_client
):
    mock_call_anthropic.return_value = {"content": [{"text": "response"}]}
    llm_client.add_provider("anthropic", "fake-api-key")
    response = await llm_client.call(
        prompt="Hello",
        model="claude-3-opus",
        json_mode=True,
        json_mode_strategy=JsonModeStrategy.prepend("custom-prefix-123"),
    )

    assert response == "{response"
    assert (
        mock_call_anthropic.call_args.kwargs["data"]["messages"][-1]["content"]
        == "custom-prefix-123{"
    )
