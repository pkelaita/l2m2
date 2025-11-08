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
from l2m2.exceptions import LLMOperationError, L2M2UsageError

LLM_POST_PATH = "l2m2.client.base_llm_client.llm_post"
LOCAL_LLM_POST_PATH = "l2m2.client.base_llm_client.local_llm_post"
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
    assert llm_client.active_hosted_providers == set()
    assert llm_client.active_hosted_models == set()


@pytest.mark.asyncio
async def test_init_with_api_keys_passed_in():
    async with BaseLLMClient(
        {"openai": "test-key-openai", "cohere": "test-key-cohere"}
    ) as llm_client:
        assert llm_client.api_keys == {
            "openai": "test-key-openai",
            "cohere": "test-key-cohere",
        }
        assert llm_client.active_hosted_providers == {"openai", "cohere"}
        assert "gpt-4o" in llm_client.active_hosted_models
        assert "command-r" in llm_client.active_hosted_models
        assert "claude-3-opus" not in llm_client.active_hosted_models


@pytest.mark.asyncio
async def test_init_with_api_keys_in_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-openai")
    monkeypatch.setenv("CO_API_KEY", "test-key-cohere")
    async with BaseLLMClient() as llm_client:
        assert llm_client.api_keys == {
            "openai": "test-key-openai",
            "cohere": "test-key-cohere",
        }
        assert llm_client.active_hosted_providers == {"openai", "cohere"}
        assert "gpt-4o" in llm_client.active_hosted_models
        assert "command-r" in llm_client.active_hosted_models
        assert "claude-3-opus" not in llm_client.active_hosted_models


@pytest.mark.asyncio
async def test_init_with_api_keys_overridden(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-openai")
    monkeypatch.setenv("CO_API_KEY", "env-key-cohere")
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
        assert llm_client.active_hosted_providers == {"openai", "cohere", "anthropic"}
        assert "gpt-4o" in llm_client.active_hosted_models
        assert "command-r" in llm_client.active_hosted_models
        assert "claude-3-opus" in llm_client.active_hosted_models


def test_init_with_invalid_provider():
    with pytest.raises(L2M2UsageError):
        BaseLLMClient({"invalid_provider": "some-key", "openai": "test-key-openai"})


def test_getters(llm_client):
    llm_client.add_provider("openai", "test-key-openai")
    llm_client.add_provider("cohere", "test-key-cohere")
    assert llm_client.get_active_providers() == {"openai", "cohere"}

    active_models = llm_client.get_active_models()
    assert "gpt-4o" in active_models
    assert "command-r" in active_models
    assert "claude-3-opus" not in active_models

    available_providers = BaseLLMClient.get_available_providers()
    assert llm_client.get_active_providers().issubset(available_providers)
    assert len(available_providers) > len(llm_client.get_active_providers())


def test_add_provider(llm_client):
    llm_client.add_provider("openai", "test-key-openai")
    assert "openai" in llm_client.get_active_providers()
    assert "gpt-4o" in llm_client.get_active_models()


def test_add_provider_invalid(llm_client):
    with pytest.raises(L2M2UsageError):
        llm_client.add_provider("invalid_provider", "some-key")


def test_remove_provider(llm_client):
    llm_client.add_provider("openai", "test-key-openai")
    llm_client.add_provider("anthropic", "test-key-anthropic")
    llm_client.remove_provider("openai")

    assert "openai" not in llm_client.active_hosted_providers
    assert "anthropic" in llm_client.active_hosted_providers
    assert "gpt-4o" not in llm_client.active_hosted_models
    assert "claude-3-opus" in llm_client.active_hosted_models


def test_remove_provider_overlapping_model(llm_client):
    llm_client.add_provider("groq", "test-key-groq")
    llm_client.add_provider("replicate", "test-key-replicate")
    assert "llama-3-8b" in llm_client.active_hosted_models

    llm_client.remove_provider("groq")
    assert "llama-3-8b" in llm_client.active_hosted_models


def test_remove_provider_not_active(llm_client):
    with pytest.raises(L2M2UsageError):
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
    with pytest.raises(L2M2UsageError):  # Invalid provider
        llm_client.set_preferred_providers({"llama-3-8b": "invalid_provider"})

    with pytest.raises(L2M2UsageError):  # Invalid model
        llm_client.set_preferred_providers({"invalid_model": "groq"})

    with pytest.raises(L2M2UsageError):  # Mismatched model and provider
        llm_client.set_preferred_providers({"llama-3-70b": "openai"})


# -- Tests for local model management -- #


def test_add_local_model(llm_client):
    llm_client.add_local_model("phi3", "ollama")
    assert ("phi3", "ollama") in llm_client.local_model_pairings


def test_add_local_model_invalid(llm_client):
    with pytest.raises(L2M2UsageError):
        llm_client.add_local_model("phi3", "invalid-provider")


def test_remove_local_model(llm_client):
    llm_client.add_local_model("phi3", "ollama")
    llm_client.add_local_model("phi4", "ollama")
    llm_client.remove_local_model("phi3", "ollama")
    assert ("phi4", "ollama") in llm_client.local_model_pairings
    assert ("phi3", "ollama") not in llm_client.local_model_pairings


def test_override_local_base_url(llm_client):
    llm_client.override_local_base_url("ollama", "http://abc:123")
    assert "ollama" in llm_client.local_provider_overrides
    assert llm_client.local_provider_overrides.get("ollama") == "http://abc:123"


def test_override_local_base_url_invalid(llm_client):
    with pytest.raises(L2M2UsageError):
        llm_client.override_local_base_url("invalid-provider", "http://abc:123")


def test_reset_local_base_url(llm_client):
    llm_client.override_local_base_url("ollama", "http://localhost:11435")
    llm_client.reset_local_base_url("ollama")
    assert "ollama" not in llm_client.local_provider_overrides


def test_reset_local_base_url_invalid(llm_client):
    with pytest.raises(L2M2UsageError):
        llm_client.reset_local_base_url("invalid-provider")


def test_reset_local_base_url_not_overridden(llm_client):
    llm_client.reset_local_base_url("ollama")  # Should not raise an error


def test_remove_local_model_invalid(llm_client):
    with pytest.raises(L2M2UsageError):
        llm_client.remove_local_model("phi3", "ollama")


def test_set_preferred_providers_local(llm_client):
    llm_client.add_local_model("phi3", "ollama")
    llm_client.set_preferred_providers({"phi3": "ollama"})
    assert llm_client.preferred_providers == {"phi3": "ollama"}


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
    mock_return_value = {
        "output": [{"type": "message", "content": [{"text": "response"}]}]
    }
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "openai", "gpt-5")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_google(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {
        "candidates": [
            {"content": {"parts": [{"text": "response"}]}, "finishReason": "STOP"}
        ]
    }
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "google", "gemini-2.0-flash")


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
    mock_return_value = {"message": {"content": [{"text": "response"}]}}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "cohere", "command-r")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_cohere_reasoning(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"message": {"content": [{"type": "text", "text": "response"}]}}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "cohere", "command-a-reasoning")


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
async def test_call_mistral_reasoning(
    mock_get_extra_message, mock_llm_post, llm_client
):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {
        "choices": [{"message": {"content": [{"type": "text", "text": "response"}]}}]
    }
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "mistral", "magistral-medium")


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
    await _generic_test_call(llm_client, "cerebras", "llama-3.3-70b")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_moonshot(mock_get_extra_message, mock_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"choices": [{"message": {"content": "response"}}]}
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "moonshot", "kimi-k2-turbo")


@pytest.mark.asyncio
@patch(LOCAL_LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_ollama(mock_get_extra_message, mock_local_llm_post, llm_client):
    mock_get_extra_message.return_value = "extra message"
    llm_client.add_local_model("phi3", "ollama")
    mock_return_value = {"message": {"content": "response"}}
    mock_local_llm_post.return_value = mock_return_value
    await _generic_test_call(llm_client, "ollama", "phi3")


# Special case for claude 3.7+ thinking: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_anthropic_thinking(
    mock_get_extra_message, mock_llm_post, llm_client
):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {
        "content": [
            {
                "type": "thinking",
                "thinking": "To approach this, let's think about what we know about prime numbers...",
                "signature": "zbbJhbGciOiJFU8zI1NiIsImtakcjsu38219c0.eyJoYXNoIjoiYWJjMTIzIiwiaWFxxxjoxNjE0NTM0NTY3fQ....",
            },
            {"type": "text", "text": "thinking response"},
        ]
    }
    mock_llm_post.return_value = mock_return_value
    await _generic_test_call(
        llm_client, "anthropic", "claude-3.7-sonnet", "thinking response"
    )


# -- Tests for call errors -- #


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_openai_bad_response(
    mock_get_extra_message, mock_llm_post, llm_client
):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"output": []}
    mock_llm_post.return_value = mock_return_value
    with pytest.raises(LLMOperationError):
        await _generic_test_call(llm_client, "openai", "gpt-5")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_call_google_bad_response(mock_llm_post, llm_client):
    llm_client.add_provider("google", "fake-api-key")
    mock_return_value = {"candidates": [{"error": "123"}]}
    mock_llm_post.return_value = mock_return_value
    response = await llm_client.call(prompt="Hello", model="gemini-2.5-pro")
    assert response == "{'error': '123'}"


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_cohere_bad_response(
    mock_get_extra_message, mock_llm_post, llm_client
):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"message": {"content": []}}
    mock_llm_post.return_value = mock_return_value
    with pytest.raises(LLMOperationError):
        await _generic_test_call(llm_client, "cohere", "command-a-reasoning")


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
@patch(GET_EXTRA_MESSAGE_PATH)
async def test_call_mistral_bad_response(
    mock_get_extra_message, mock_llm_post, llm_client
):
    mock_get_extra_message.return_value = "extra message"
    mock_return_value = {"choices": [{"message": {"content": []}}]}
    mock_llm_post.return_value = mock_return_value
    with pytest.raises(LLMOperationError):
        await _generic_test_call(llm_client, "mistral", "magistral-medium")


@pytest.mark.asyncio
async def test_call_valid_model_not_active(llm_client):
    with pytest.raises(L2M2UsageError):
        await llm_client.call(prompt="Hello", model="gpt-5")


@pytest.mark.asyncio
async def test_call_invalid_model(llm_client):
    with pytest.raises(L2M2UsageError):
        await llm_client.call(prompt="Hello", model="unknown-model")


@pytest.mark.asyncio
async def test_call_tokens_too_large(llm_client):
    llm_client.add_provider("openai", "fake-api-key")
    with pytest.raises(L2M2UsageError):
        await llm_client.call(prompt="Hello", model="gpt-5", max_tokens=float("inf"))


@pytest.mark.asyncio
async def test_call_temperature_too_high(llm_client):
    llm_client.add_provider("openai", "fake-api-key")
    with pytest.raises(L2M2UsageError):
        await llm_client.call(prompt="Hello", model="gpt-5", temperature=3.0)


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
    await llm_client.call(prompt="Hello", model="qwen-qwq-32b")

    with pytest.raises(L2M2UsageError):
        await llm_client.call(prompt="Hello", model="llama-3-70b")


@pytest.mark.asyncio
async def test_multi_provider_pref_inactive(llm_client):
    llm_client.add_provider("groq", "test-key-groq")
    llm_client.add_provider("replicate", "test-key-replicate")
    with pytest.raises(L2M2UsageError):
        await llm_client.call(
            prompt="Hello", model="llama-3-70b", prefer_provider="openai"
        )


# -- Tests for memory -- #


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_chat_memory(mock_call_openai, llm_client_mem_chat):
    mock_call_openai.return_value = {
        "output": [{"type": "message", "content": [{"text": "response"}]}]
    }

    llm_client_mem_chat.add_provider("openai", "fake-api-key")

    memory = llm_client_mem_chat.get_memory()
    assert isinstance(memory, ChatMemory)

    memory.add_user_message("A")
    memory.add_agent_message("B")

    response = await llm_client_mem_chat.call(prompt="C", model="gpt-5")
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
    with pytest.raises(L2M2UsageError):
        llm_client.get_memory()

    with pytest.raises(L2M2UsageError):
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
    mock_call_openai.return_value = {
        "output": [{"type": "message", "content": [{"text": "response"}]}]
    }
    llm_client_mem_ext_sys.add_provider("openai", "fake-api-key")

    memory = llm_client_mem_ext_sys.get_memory()
    assert isinstance(memory, ExternalMemory)

    memory.set_contents("stuff")

    await llm_client_mem_ext_sys.call(prompt="Hello", model="gpt-5")
    assert mock_call_openai.call_args.kwargs["data"]["input"] == [
        {"role": "developer", "content": "stuff"},
        {"role": "user", "content": "Hello"},
    ]

    await llm_client_mem_ext_sys.call(
        system_prompt="system-123", prompt="Hello", model="gpt-5"
    )
    assert mock_call_openai.call_args.kwargs["data"]["input"] == [
        {"role": "developer", "content": "system-123\nstuff"},
        {"role": "user", "content": "Hello"},
    ]


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_external_memory_user_prompt(mock_call_openai, llm_client_mem_ext_usr):
    mock_call_openai.return_value = {
        "output": [{"type": "message", "content": [{"text": "response"}]}]
    }
    llm_client_mem_ext_usr.add_provider("openai", "fake-api-key")

    memory = llm_client_mem_ext_usr.get_memory()
    assert isinstance(memory, ExternalMemory)

    memory.set_contents("stuff")

    await llm_client_mem_ext_usr.call(prompt="Hello", model="gpt-5")
    assert mock_call_openai.call_args.kwargs["data"]["input"] == [
        {"role": "user", "content": "Hello\nstuff"},
    ]

    await llm_client_mem_ext_usr.call(
        system_prompt="system-123", prompt="Hello", model="gpt-5"
    )
    assert mock_call_openai.call_args.kwargs["data"]["input"] == [
        {"role": "developer", "content": "system-123"},
        {"role": "user", "content": "Hello\nstuff"},
    ]


@pytest.mark.asyncio
@patch(LLM_POST_PATH)
async def test_bypass_memory(mock_call_openai, llm_client_mem_chat):
    mock_call_openai.return_value = {
        "output": [{"type": "message", "content": [{"text": "response"}]}]
    }
    llm_client_mem_chat.add_provider("openai", "fake-api-key")
    llm_client_mem_chat.get_memory().add_user_message("A")
    llm_client_mem_chat.get_memory().add_agent_message("B")

    await llm_client_mem_chat.call(prompt="Hello", model="gpt-5", bypass_memory=True)
    assert mock_call_openai.call_args.kwargs["data"]["input"] == [
        {"role": "user", "content": "Hello"},
    ]
    assert llm_client_mem_chat.get_memory().unpack(
        "role", "content", "user", "assistant"
    ) == [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]

    await llm_client_mem_chat.call(prompt="Hello", model="gpt-5")
    assert mock_call_openai.call_args.kwargs["data"]["input"] == [
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
    mock_call_openai.return_value = {
        "output": [{"type": "message", "content": [{"text": "response"}]}]
    }
    llm_client.add_provider("openai", "fake-api-key")

    m1 = ChatMemory()
    m2 = ChatMemory()
    llm_client.load_memory(ChatMemory())

    await llm_client.call(prompt="A", model="gpt-5", alt_memory=m1)
    await llm_client.call(prompt="X", model="gpt-5", alt_memory=m2)
    await llm_client.call(prompt="B", model="gpt-5", alt_memory=m1)
    await llm_client.call(prompt="Y", model="gpt-5", alt_memory=m2)
    await llm_client.call(prompt="C", model="gpt-5", alt_memory=m1)
    await llm_client.call(prompt="Z", model="gpt-5", alt_memory=m2)

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
    mock_call.return_value = {"message": {"content": [{"text": "--{response}--"}]}}
    llm_client.add_provider("cohere", "fake-api-key")
    response = await llm_client.call(
        prompt="Hello",
        model="command-r",
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
    mock_call_cohere.return_value = {"message": {"content": [{"text": "response"}]}}
    llm_client.add_provider("cohere", "fake-api-key")
    response = await llm_client.call(
        prompt="Hello",
        model="command-r",
        json_mode=True,
        json_mode_strategy=JsonModeStrategy.prepend(),
    )

    assert response == "{response"
    assert (
        mock_call_cohere.call_args.kwargs["data"]["messages"][-1]["content"]
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
