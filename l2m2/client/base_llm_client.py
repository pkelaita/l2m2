from typing import Any, List, Set, Dict, Optional, Tuple, Union
import httpx
import os

from l2m2.model_info import (
    MODEL_INFO,
    HOSTED_PROVIDERS,
    LOCAL_PROVIDERS,
    PROVIDER_DEFAULT,
    ModelEntry,
    ModelParams,
    ParamName,
    get_id,
)
from l2m2.memory import (
    ChatMemory,
    ExternalMemory,
    ExternalMemoryLoadingType,
    BaseMemory,
)
from l2m2.tools.json_mode_strategies import (
    JsonModeStrategy,
    StrategyName,
    get_extra_message,
    run_json_strats_out,
)
from l2m2.exceptions import LLMOperationError, L2M2UsageError
from l2m2._internal.http import llm_post, local_llm_post
from l2m2.warnings import deprecated

DEFAULT_TIMEOUT_SECONDS = 25

DEFAULT_PROVIDER_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "CO_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
    "mistral": "MISTRAL_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}


class BaseLLMClient:
    def __init__(
        self,
        api_keys: Optional[Dict[str, str]] = None,
        memory: Optional[BaseMemory] = None,
    ) -> None:
        """Initializes the LLM Client.

        Args:
            providers ([Dict[str, str]], optional): Mapping from provider name to API key.
                For example::

                    {
                        "openai": "openai-api
                        "anthropic": "anthropic-api-key",
                        "google": "google-api-key",
                    }

                Defaults to `None`.
            memory (BaseMemory, optional): The memory object to use. Defaults to `None`, in which
                case memory is not enabled.

        Raises:
            L2M2UsageError: If an invalid provider is specified in `providers`.
        """
        # Hosted models and providers state
        self.api_keys: Dict[str, str] = {}
        self.active_hosted_providers: Set[str] = set()
        self.active_hosted_models: Set[str] = set()

        # Local models and providers state
        self.local_model_pairings: Set[Tuple[str, str]] = set()  # (model, provider)
        self.local_provider_overrides: Dict[str, str] = {}  # provider -> base url

        # Misc state
        self.preferred_providers: Dict[str, str] = {}  # model -> provider
        self.memory = memory
        self.httpx_client = httpx.AsyncClient()

        if api_keys is not None:
            for provider, api_key in api_keys.items():
                self.add_provider(provider, api_key)

        for provider, env_var in DEFAULT_PROVIDER_ENVS.items():
            if (
                provider not in self.active_hosted_providers
                and (default_api_key := os.getenv(env_var)) is not None
            ):
                self.add_provider(provider, default_api_key)

    async def __aenter__(self) -> "BaseLLMClient":
        await self.httpx_client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.httpx_client.__aexit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def get_available_providers() -> Set[str]:
        """Get the exhaustive set of L2M2's available model providers. This set includes
        all providers, regardless of whether they are currently active, and includes both hosted
        and local providers.

        Returns:
            Set[str]: A set of available providers.
        """
        return set(HOSTED_PROVIDERS.keys()) | set(LOCAL_PROVIDERS.keys())

    @staticmethod
    @deprecated(
        "This set is no longer meaningful since L2M2 supports local models as of 0.0.41."
    )
    def get_available_models() -> Set[str]:  # pragma: no cover
        """The set of L2M2's supported models. This set includes all models, regardless of
        whether they are currently active.

        Returns:
            Set[str]: A set of available models.
        """
        return set(MODEL_INFO.keys())

    def get_active_providers(self) -> Set[str]:
        """Get the set of currently active providers. Active providers are either hosted providers
        for which an API key has been set, or local providers for which a model has been added.

        Returns:
            Set[str]: A set of active providers.
        """
        return set(self.active_hosted_providers) | self._get_active_local_providers()

    def get_active_models(self) -> Set[str]:
        """Get the set of currently active models. Active models are those that are available and
        have a provider that is active.

        Returns:
            Set[str]: A set of active models.
        """
        return set(self.active_hosted_models) | self._get_active_local_models()

    def add_provider(self, provider: str, api_key: str) -> None:
        """Add a provider to the LLMClient, making its models available for use.

        Args:
            provider (str): The provider name. Must be one of the available providers.
            api_key (str): The API key for the provider.

        Raises:
            L2M2UsageError: If the provider is not one of the available providers.
        """
        if provider not in (providers := self.get_available_providers()):
            raise L2M2UsageError(
                f"Invalid provider: {provider}. Available providers: {providers}"
            )

        self.api_keys[provider] = api_key
        self.active_hosted_providers.add(provider)
        self.active_hosted_models.update(
            model for model in MODEL_INFO.keys() if provider in MODEL_INFO[model].keys()
        )

    def remove_provider(self, provider_to_remove: str) -> None:
        """Remove a provider from the LLMClient, making its models unavailable for use.

        Args:
            provider (str): The active provider to remove.

        Raises:
            L2M2UsageError: If the given provider is not active.
        """
        if provider_to_remove not in self.active_hosted_providers:
            raise L2M2UsageError(f"Provider not active: {provider_to_remove}")

        del self.api_keys[provider_to_remove]
        self.active_hosted_providers.remove(provider_to_remove)

        self.active_hosted_models = {
            model
            for model in self.active_hosted_models
            if not MODEL_INFO[model].keys().isdisjoint(self.active_hosted_providers)
        }

    def add_local_model(self, model: str, local_provider: str) -> None:
        """Add a local model to the LLMClient.

        Args:
            model (str): The model name.
            local_provider (str): The local provider name (Currently, only "ollama" is supported).

        Raises:
            L2M2UsageError: If the local provider is invalid.
        """
        if local_provider not in LOCAL_PROVIDERS:
            raise L2M2UsageError(
                f"Local provider must be one of {LOCAL_PROVIDERS.keys()}"
            )

        self.local_model_pairings.add((model, local_provider))

    def remove_local_model(self, model: str, local_provider: str) -> None:
        """Remove a local model from the LLMClient.

        Args:
            model (str): The model name.
            local_provider (str): The local provider name (Currently, only "ollama" is supported).

        Raises:
            L2M2UsageError: If the local model is not active.
        """
        if (model, local_provider) not in self.local_model_pairings:
            raise L2M2UsageError(f"Local {model} via {local_provider} is not active.")

        self.local_model_pairings.remove((model, local_provider))

    def override_local_base_url(self, local_provider: str, base_url: str) -> None:
        """Overrides the default base URL for a local provider. For example, ollama defaults to
        http://localhost:11434, but you can override it to use a remote instance, a different port,
        etc.

        Note - If you're hosting your own models on a remote server, you'll probably want to ensure
        that you inject the appropriate headers to authenticate your requests. In order to do that,
        you can use the `extra_headers` argument in the `call` method.

        Args:
            local_provider (str): The local provider name (Currently, only "ollama" is supported).
            new_base_url (str): The new base URL to use for the local provider.

        Raises:
            L2M2UsageError: If the local provider is invalid.
        """
        if local_provider not in LOCAL_PROVIDERS:
            raise L2M2UsageError(
                f"Local provider must be one of {LOCAL_PROVIDERS.keys()}"
            )

        self.local_provider_overrides[local_provider] = base_url

    def reset_local_base_url(self, local_provider: str) -> None:
        """Resets the base URL for a local provider to the default.

        Args:
            local_provider (str): The local provider name (Currently, only "ollama" is supported).

        Raises:
            L2M2UsageError: If the local provider is invalid.
        """
        if local_provider not in LOCAL_PROVIDERS:
            raise L2M2UsageError(
                f"Local provider must be one of {LOCAL_PROVIDERS.keys()}"
            )

        self.local_provider_overrides.pop(local_provider, None)

    def set_preferred_providers(self, preferred_providers: Dict[str, str]) -> None:
        """Set the preferred provider for each model. If a model is available from multiple active
        providers, the preferred provider will be used.

        Args:
            preferred_providers (Dict[str, str]): A mapping from model name to preferred provider.
                For example::

                    {
                        "llama-3-8b": "groq",
                        "llama-3-70b": "replicate",
                    }

                If you'd like to remove a preferred provider, set it to `None`, e.g.::

                    {
                        "llama-3-8b": None,
                    }

        Raises:
            L2M2UsageError: If an invalid model or provider is specified in `preferred_providers`.
            L2M2UsageError: If the given provider is hosted and the given model is not available from it.
        """

        for model, provider in preferred_providers.items():
            if provider is not None:
                if provider not in self.get_available_providers():
                    raise L2M2UsageError(f"Invalid provider: {provider}")

                if provider in HOSTED_PROVIDERS and (
                    model not in MODEL_INFO or provider not in MODEL_INFO[model].keys()
                ):
                    raise L2M2UsageError(
                        f"Model {model} is not available from provider {provider}."
                    )

        self.preferred_providers.update(preferred_providers)

    def get_memory(self) -> BaseMemory:
        """Get the memory object, if memory is enabled.

        Returns:
            BaseMemory: The memory object.

        Raises:
            L2M2UsageError: If memory is not enabled.
        """
        if self.memory is None:
            raise L2M2UsageError("Memory is not enabled.")

        return self.memory

    def clear_memory(self) -> None:
        """Clear the memory, if memory is enabled.

        Raises:
            L2M2UsageError: If memory is not enabled.
        """
        if self.memory is None:
            raise L2M2UsageError("Memory is not enabled.")

        self.memory.clear()

    def load_memory(self, memory_object: BaseMemory) -> None:
        """Loads memory into the LLM client. If the client already has memory enabled, the existing
        memory is replaced with the new memory.

        Args:
            memory_object (BaseMemory): The memory to load.

        """
        self.memory = memory_object

    async def call(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        prefer_provider: Optional[str] = None,
        json_mode: bool = False,
        json_mode_strategy: Optional[JsonModeStrategy] = None,
        timeout: Optional[int] = DEFAULT_TIMEOUT_SECONDS,
        bypass_memory: bool = False,
        alt_memory: Optional[BaseMemory] = None,
        extra_params: Optional[Dict[str, Union[str, int, float]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Performs inference on any active model.

        Args:
            model (str): The active model to call.
            prompt (str): The user prompt for which to generate a completion.
            system_prompt (str, optional): The system prompt to send to the model. If the specified
                model does not support system prompts, it is prepended to the user prompt. Defaults
                to None.
            temperature (float, optional): The sampling temperature for the model. If not specified,
                the provider's default value for the model is used. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. If not specified,
                the provider's default value for the model is used. Defaults to None.
            prefer_provider (str, optional): The preferred provider to use for the model, if the
                model is available from multiple active providers. Defaults to None.
            json_mode (bool, optional): Whether to return the response in JSON format. Defaults to False.
            json_mode_strategy (JsonModeStrategy, optional): The strategy to use to enforce JSON outputs
                when `json_mode` is True. If `None`, the default strategy will be used:
                `JsonModeStrategy.prepend()` for Anthropic, and `JsonModeStrategy.strip()` for all other
                providers. Defaults to `None`.
            timeout (int, optional): The timeout in seconds for the LLM request. Can be set to `None`,
                in which case the request will be allowed to run indefinitely. Defaults to `10`.
            bypass_memory (bool, optional): Whether to bypass memory when calling the model. If `True`, the
                model will not read from or write to memory during the call if memory is enabled. Defaults
                to `False`.
            alt_memory (BaseMemory, optional): An alternative memory object to use for this call only. This
                is very useful for asynchronous workflows where you want to keep track of multiple memory
                streams in parallel without risking race conditions. Defaults to `None`.
            extra_params (Dict[str, Union[str, int, float]], optional): Extra parameters to pass to the model.
                Defaults to `None`.
            extra_headers (Dict[str, str], optional): Extra HTTP headers to pass in the request to the service
                hosting the model. Defaults to `None`.

        Raises:
            L2M2UsageError: If the provided model is not active and/or not available.
            L2M2UsageError: If the model is available from multiple active providers neither `prefer_provider`
                nor a default provider is specified.
            L2M2UsageError: If `prefer_provider` is specified but not active.

        Returns:
            str: The model's completion for the prompt, or an error message if the model is
                unable to generate a completion.
        """
        if model not in self.get_active_models():
            raise L2M2UsageError(f"Invalid or non-active model: {model}")

        if (
            prefer_provider is not None
            and prefer_provider not in self.get_active_providers()
        ):
            raise L2M2UsageError(
                "Argument prefer_provider must either be None or an active provider."
                + f" Active providers are {', '.join(self.get_active_providers())}"
            )

        hosted_providers = (
            set(MODEL_INFO.get(model, {}).keys()) & self.active_hosted_providers
        )
        local_providers = self._get_local_providers_for_model(model)
        providers = hosted_providers | local_providers

        if len(providers) == 1:
            provider = next(iter(providers))

        elif prefer_provider is not None:
            provider = prefer_provider

        elif self.preferred_providers.get(model) is not None:
            provider = self.preferred_providers[model]

        else:
            raise L2M2UsageError(
                f"Model {model} is available from multiple active providers: {', '.join(providers)}."
                + " Please specify a preferred provider with the argument prefer_provider, or set a"
                + " default provider for the model with set_preferred_providers."
            )

        model_entry = (
            MODEL_INFO[model][provider]
            if provider in HOSTED_PROVIDERS
            else _get_local_model_entry(provider, model)
        )

        return await self._call_impl(
            model_entry,
            provider,
            prompt,
            system_prompt,
            temperature,
            max_tokens,
            json_mode,
            json_mode_strategy,
            timeout,
            bypass_memory,
            alt_memory,
            extra_params,
            extra_headers,
        )

    async def _call_impl(
        self,
        model_entry: ModelEntry,
        provider: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        json_mode: bool,
        json_mode_strategy: Optional[JsonModeStrategy],
        timeout: Optional[int],
        bypass_memory: bool,
        alt_memory: Optional[BaseMemory],
        extra_params: Optional[Dict[str, Union[str, int, float]]],
        extra_headers: Optional[Dict[str, str]],
    ) -> str:
        # Prepare memory
        memory = alt_memory if alt_memory is not None else self.memory
        if bypass_memory:
            memory = None

        # Prepare JSON mode strategy
        if json_mode_strategy is None:
            json_mode_strategy = (
                JsonModeStrategy.strip()
                if provider != "anthropic"
                else JsonModeStrategy.prepend()
            )

        # Prepare params
        params: Dict[str, Any] = {}
        _add_param(params, model_entry["params"], "temperature", temperature)
        _add_param(params, model_entry["params"], "max_tokens", max_tokens)

        # Handle native JSON mode
        has_native_json_mode = "json_mode_arg" in model_entry["extras"]
        if json_mode and has_native_json_mode:
            arg = model_entry["extras"]["json_mode_arg"]
            key, value = next(iter(arg.items()))
            params[key] = value

        # Update prompts if we're using external memory
        if isinstance(memory, ExternalMemory):
            system_prompt, prompt = _get_external_memory_prompts(
                memory, system_prompt, prompt
            )

        # Run the LLM
        call_fn = getattr(self, f"_call_{provider}")
        result = await call_fn(
            model_entry["model_id"],
            prompt,
            system_prompt,
            params,
            timeout,
            memory,
            extra_params,
            extra_headers,
            # Args below here are not always used
            json_mode,
            json_mode_strategy,
            model_entry["extras"],
        )

        # Handle JSON mode strategies for the output (but only if we don't have native support)
        if json_mode and not has_native_json_mode:
            result = run_json_strats_out(json_mode_strategy, result)

        # Lastly, update chat memory if applicable
        if isinstance(memory, ChatMemory):
            memory.add_user_message(prompt)
            memory.add_agent_message(result)

        return str(result)

    async def _call_openai(
        self,
        *args: Any,
    ) -> str:
        if args[2] and args[0] in [
            get_id("openai", "o1-mini"),
            get_id("openai", "o1-preview"),
        ]:
            raise LLMOperationError(
                "OpenAI o1-mini and o1-preview do not support system prompts. Try using "
                + "o1, which supports them, or appending the system prompt to the user prompt. "
                + "For discussion on this issue, see "
                + "https://community.openai.com/t/o1-supports-system-role-o1-mini-does-not/1071954/3"
            )
        return await self._generic_openai_spec_call("openai", *args)

    async def _call_google(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
        timeout: Optional[int],
        memory: Optional[BaseMemory],
        extra_params: Optional[Dict[str, Union[str, int, float]]],
        extra_headers: Optional[Dict[str, str]],
        *_: Any,  # json_mode and json_mode_strategy, and extras are not used here
    ) -> str:
        data: Dict[str, Any] = {}

        if system_prompt is not None:
            data["system_instruction"] = {"parts": {"text": system_prompt}}

        messages: List[Dict[str, Any]] = []
        if isinstance(memory, ChatMemory):
            mem_items = memory.unpack("role", "parts", "user", "model")
            # Need to do this wrap – see https://ai.google.dev/api/rest/v1beta/cachedContents#Part
            messages.extend([{**m, "parts": {"text": m["parts"]}} for m in mem_items])

        messages.append({"role": "user", "parts": {"text": prompt}})

        data["contents"] = messages
        data["generation_config"] = params

        result = await llm_post(
            client=self.httpx_client,
            provider="google",
            model_id=model_id,
            api_key=self.api_keys["google"],
            data=data,
            timeout=timeout,
            extra_params=extra_params,
            extra_headers=extra_headers,
        )
        result = result["candidates"][0]

        # Will sometimes fail due to safety filters
        if "content" in result:
            return str(result["content"]["parts"][0]["text"])
        else:
            return str(result)

    async def _call_anthropic(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
        timeout: Optional[int],
        memory: Optional[BaseMemory],
        extra_params: Optional[Dict[str, Union[str, int, float]]],
        extra_headers: Optional[Dict[str, str]],
        json_mode: bool,
        json_mode_strategy: JsonModeStrategy,
        _: Dict[str, Any],  # extras is not used here
    ) -> str:
        if system_prompt is not None:
            params["system"] = system_prompt
        messages = []
        if isinstance(memory, ChatMemory):
            messages.extend(memory.unpack("role", "content", "user", "assistant"))
        messages.append({"role": "user", "content": prompt})

        if json_mode:
            append_msg = get_extra_message(json_mode_strategy)
            if append_msg:
                messages.append({"role": "assistant", "content": append_msg})

        result = await llm_post(
            client=self.httpx_client,
            provider="anthropic",
            model_id=model_id,
            api_key=self.api_keys["anthropic"],
            data={"model": model_id, "messages": messages, **params},
            timeout=timeout,
            extra_params=extra_params,
            extra_headers=extra_headers,
        )
        if "text" in result["content"][0]:
            return str(result["content"][0]["text"])
        else:
            # Account for thinking with claude 3.7+: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
            return str(result["content"][1]["text"])

    async def _call_cohere(
        self,
        *args: Any,
    ) -> str:
        return await self._generic_openai_spec_call("cohere", *args)

    async def _call_groq(
        self,
        *args: Any,
    ) -> str:
        return await self._generic_openai_spec_call("groq", *args)

    async def _call_mistral(
        self,
        *args: Any,
    ) -> str:
        return await self._generic_openai_spec_call("mistral", *args)

    async def _call_replicate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
        timeout: Optional[int],
        memory: Optional[BaseMemory],
        extra_params: Optional[Dict[str, Union[str, int, float]]],
        extra_headers: Optional[Dict[str, str]],
        _: bool,  # json_mode is not used here
        json_mode_strategy: JsonModeStrategy,
        __: Dict[str, Any],  # extras is not used here
    ) -> str:
        if isinstance(memory, ChatMemory):
            raise LLMOperationError(
                "ChatMemory is not supported with Replicate."
                + " Try using Groq, or using ExternalMemory instead."
            )
        if json_mode_strategy.strategy_name == StrategyName.PREPEND:
            raise LLMOperationError(
                "JsonModeStrategy.prepend() is not supported with Replicate."
                + " Try using Groq, or using JsonModeStrategy.strip() instead."
            )

        if system_prompt is not None:
            params["system_prompt"] = system_prompt

        result = await llm_post(
            client=self.httpx_client,
            provider="replicate",
            model_id=model_id,
            api_key=self.api_keys["replicate"],
            data={"input": {"prompt": prompt, **params}},
            timeout=timeout,
            extra_params=extra_params,
            extra_headers=extra_headers,
        )
        return "".join(result["output"])

    async def _call_cerebras(
        self,
        *args: Any,
    ) -> str:
        return await self._generic_openai_spec_call("cerebras", *args)

    async def _call_ollama(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
        timeout: Optional[int],
        memory: Optional[BaseMemory],
        extra_params: Optional[Dict[str, Union[str, int, float]]],
        extra_headers: Optional[Dict[str, str]],
        json_mode: bool,
        json_mode_strategy: JsonModeStrategy,
        _: Dict[str, Any],  # extras is not used here
    ) -> str:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        if isinstance(memory, ChatMemory):
            messages.extend(memory.unpack("role", "content", "user", "assistant"))
        messages.append({"role": "user", "content": prompt})

        result = await local_llm_post(
            client=self.httpx_client,
            provider="ollama",
            data={"model": model_id, "messages": messages, **params},
            timeout=timeout,
            local_provider_overrides=self.local_provider_overrides,
            extra_params=extra_params,
            extra_headers=extra_headers,
        )
        return str(result["message"]["content"])

    async def _generic_openai_spec_call(
        self,
        provider: str,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
        timeout: Optional[int],
        memory: Optional[BaseMemory],
        extra_params: Optional[Dict[str, Union[str, int, float]]],
        extra_headers: Optional[Dict[str, str]],
        json_mode: bool,
        json_mode_strategy: JsonModeStrategy,
        extras: Dict[str, Any],
    ) -> str:
        """Generic call method for providers who follow the OpenAI API spec."""
        supports_native_json_mode = "json_mode_arg" in extras

        # For o1 and newer, use "developer" messages instead of "system"
        system_key = "system"
        if provider == "openai" and model_id in [
            get_id("openai", "o1"),
            get_id("openai", "o3-mini"),
        ]:
            system_key = "developer"

        messages = []
        if system_prompt is not None:
            messages.append({"role": system_key, "content": system_prompt})
        if isinstance(memory, ChatMemory):
            messages.extend(memory.unpack("role", "content", "user", "assistant"))
        messages.append({"role": "user", "content": prompt})

        if json_mode and not supports_native_json_mode:
            append_msg = get_extra_message(json_mode_strategy)
            if append_msg:
                messages.append({"role": "assistant", "content": append_msg})

        result = await llm_post(
            client=self.httpx_client,
            provider=provider,
            model_id=model_id,
            api_key=self.api_keys[provider],
            data={"model": model_id, "messages": messages, **params},
            timeout=timeout,
            extra_params=extra_params,
            extra_headers=extra_headers,
        )

        # Cohere API v2 uses OpenAI spec, but not the same response format for some reason...
        if provider == "cohere":
            return str(result["message"]["content"][0]["text"])

        return str(result["choices"][0]["message"]["content"])

    # State-dependent helper methods

    def _get_local_providers_for_model(self, model: str) -> Set[str]:
        return {
            provider_i
            for model_i, provider_i in self.local_model_pairings  # comment to preserve formatting
            if model_i == model
        }

    def _get_active_local_models(self) -> Set[str]:
        return {model_i for model_i, _ in self.local_model_pairings}

    def _get_active_local_providers(self) -> Set[str]:
        return {provider_i for _, provider_i in self.local_model_pairings}


# Non-state-dependent helper methods


def _get_local_model_entry(provider: str, model_id: str) -> ModelEntry:
    generic_model_entry = LOCAL_PROVIDERS[provider]["model_entry"]
    return {**generic_model_entry, "model_id": model_id}


def _get_external_memory_prompts(
    memory: ExternalMemory, system_prompt: Optional[str], prompt: str
) -> Tuple[Optional[str], str]:
    if memory.loading_type == ExternalMemoryLoadingType.SYSTEM_PROMPT_APPEND:
        contents = memory.get_contents()
        if system_prompt is not None:
            system_prompt += "\n" + contents
        else:
            system_prompt = contents

    elif memory.loading_type == ExternalMemoryLoadingType.USER_PROMPT_APPEND:
        prompt += "\n" + memory.get_contents()

    return system_prompt, prompt


def _add_param(
    params: Dict[str, Any],
    param_info: ModelParams,
    name: ParamName,
    value: Any,
) -> None:
    if value is not None and value > (max_val := param_info[name]["max"]):
        msg = f"Parameter {name} exceeds max value {max_val}"
        raise L2M2UsageError(msg)

    key = str(name) if (key := param_info[name].get("custom_key")) is None else key

    if value is not None:
        params[key] = value
    elif (default := param_info[name]["default"]) != PROVIDER_DEFAULT:
        params[key] = default
