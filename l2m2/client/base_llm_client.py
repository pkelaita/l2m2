from typing import Any, List, Set, Dict, Optional, Tuple
import httpx
import os

from l2m2.model_info import (
    MODEL_INFO,
    PROVIDER_INFO,
    PROVIDER_DEFAULT,
    ModelEntry,
    ModelParams,
    ParamName,
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
from l2m2.exceptions import LLMOperationError
from l2m2._internal.http import llm_post


DEFAULT_TIMEOUT_SECONDS = 10

DEFAULT_PROVIDER_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "CO_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
    "mistral": "MISTRAL_API_KEY",
}


class BaseLLMClient:
    def __init__(
        self,
        providers: Optional[Dict[str, str]] = None,
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
            ValueError: If an invalid provider is specified in `providers`.
        """
        self.api_keys: Dict[str, str] = {}
        self.active_providers: Set[str] = set()
        self.active_models: Set[str] = set()
        self.preferred_providers: Dict[str, str] = {}
        self.memory: Optional[BaseMemory] = None

        if providers is not None:
            for provider, api_key in providers.items():
                self.add_provider(provider, api_key)

        for provider, env_var in DEFAULT_PROVIDER_ENVS.items():
            if (
                provider not in self.active_providers
                and (default_api_key := os.getenv(env_var)) is not None
            ):
                self.add_provider(provider, default_api_key)

        self.memory = memory
        self.httpx_client = httpx.AsyncClient()

    async def __aenter__(self) -> "BaseLLMClient":
        await self.httpx_client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.httpx_client.__aexit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def get_available_providers() -> Set[str]:
        """Get the exhaustive set of L2M2's available model providers. This set includes
        all providers, regardless of whether they are currently active. L2M2 does not currently
        support adding custom providers.

        Returns:
            Set[str]: A set of available providers.
        """
        return set(PROVIDER_INFO.keys())

    @staticmethod
    def get_available_models() -> Set[str]:
        """The set of L2M2's supported models. This set includes all models, regardless of
        whether they are currently active. L2M2 allows users to call non-available models
        from available and active providers, but does not guarantee correctness for such calls.

        Returns:
            Set[str]: A set of available models.
        """
        return set(MODEL_INFO.keys())

    def get_active_providers(self) -> Set[str]:
        """Get the set of currently active providers. Active providers are those for which an API
        key has been set.

        Returns:
            Set[str]: A set of active providers.
        """
        return set(self.active_providers)

    def get_active_models(self) -> Set[str]:
        """Get the set of currently active models. Active models are those that are available and
        have a provider that is active.

        Returns:
            Set[str]: A set of active models.
        """
        return set(self.active_models)

    def add_provider(self, provider: str, api_key: str) -> None:
        """Add a provider to the LLMClient, making its models available for use.

        Args:
            provider (str): The provider name. Must be one of the available providers.
            api_key (str): The API key for the provider.

        Raises:
            ValueError: If the provider is not one of the available providers.
            ValueError: If the API key is not a string.
        """
        if provider not in (providers := self.get_available_providers()):
            raise ValueError(
                f"Invalid provider: {provider}. Available providers: {providers}"
            )
        if not isinstance(api_key, str):
            raise ValueError(f"API key for provider {provider} must be a string.")

        self.api_keys[provider] = api_key
        self.active_providers.add(provider)
        self.active_models.update(
            model for model in MODEL_INFO.keys() if provider in MODEL_INFO[model].keys()
        )

    def remove_provider(self, provider_to_remove: str) -> None:
        """Remove a provider from the LLMClient, making its models unavailable for use.

        Args:
            provider (str): The active provider to remove.

        Raises:
            ValueError: If the given provider is not active.
        """
        if provider_to_remove not in self.active_providers:
            raise ValueError(f"Provider not active: {provider_to_remove}")

        del self.api_keys[provider_to_remove]
        self.active_providers.remove(provider_to_remove)

        self.active_models = {
            model
            for model in self.active_models
            if not MODEL_INFO[model].keys().isdisjoint(self.active_providers)
        }

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
            ValueError: If an invalid model or provider is specified in `preferred_providers`.
            ValueError: If the given model does not correlate to the given provider.
        """
        for model, provider in preferred_providers.items():
            if model not in self.get_available_models():
                raise ValueError(f"Invalid model: {model}")
            if provider is not None:
                if provider not in self.get_available_providers():
                    raise ValueError(f"Invalid provider: {provider}")
                if provider not in MODEL_INFO[model].keys():
                    raise ValueError(
                        f"Model {model} is not available from provider {provider}."
                    )

        self.preferred_providers.update(preferred_providers)

    def get_memory(self) -> BaseMemory:
        """Get the memory object, if memory is enabled.

        Returns:
            BaseMemory: The memory object.

        Raises:
            ValueError: If memory is not enabled.
        """
        if self.memory is None:
            raise ValueError("Memory is not enabled.")

        return self.memory

    def clear_memory(self) -> None:
        """Clear the memory, if memory is enabled.

        Raises:
            ValueError: If memory is not enabled.
        """
        if self.memory is None:
            raise ValueError("Memory is not enabled.")

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

        Raises:
            ValueError: If the provided model is not active and/or not available.
            ValueError: If the model is available from multiple active providers neither `prefer_provider`
                nor a default provider is specified.
            ValueError: If `prefer_provider` is specified but not active.

        Returns:
            str: The model's completion for the prompt, or an error message if the model is
                unable to generate a completion.
        """
        if model not in self.active_models:
            if model in self.get_available_models():
                available_providers = ", ".join(MODEL_INFO[model].keys())
                msg = (
                    f"Model {model} is available, but not active."
                    + f" Please add any of ({available_providers}) to activate it."
                )
                raise ValueError(msg)
            else:
                raise ValueError(f"Invalid model: {model}")

        if prefer_provider is not None and prefer_provider not in self.active_providers:
            raise ValueError(
                "Argument prefer_provider must either be None or an active provider."
                + f" Active providers are {', '.join(self.active_providers)}"
            )

        providers = set(MODEL_INFO[model].keys()) & self.active_providers
        if len(providers) == 1:
            provider = next(iter(providers))

        elif prefer_provider is not None:
            provider = prefer_provider

        elif self.preferred_providers.get(model) is not None:
            provider = self.preferred_providers[model]

        else:
            raise ValueError(
                f"Model {model} is available from multiple active providers: {', '.join(providers)}."
                + " Please specify a preferred provider with the argument prefer_provider, or set a"
                + " default provider for the model with set_preferred_providers."
            )

        return await self._call_impl(
            MODEL_INFO[model][provider],
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
        )

    async def call_custom(
        self,
        *,
        provider: str,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        json_mode_strategy: Optional[JsonModeStrategy] = None,
        timeout: Optional[int] = DEFAULT_TIMEOUT_SECONDS,
        bypass_memory: bool = False,
        alt_memory: Optional[BaseMemory] = None,
    ) -> str:
        """Performs inference on any model from an active provider that is not officially supported
        by L2M2. This method does not guarantee correctness.

        Args:
            provider (str): The provider to use. Must be one of the active providers.
            model_id (str): The ID of model to call. Must be the exact match to how you would call
                it with the provider's API. For example, `gpt-3.5-turbo-0125` can be used to call
                a legacy model from OpenAI as per the OpenAI API docs.
                (https://platform.openai.com/docs/api-reference/chat)
            prompt (str): The user prompt for which to generate a completion.
            system_prompt (str, optional): The system prompt to send to the model. Defaults to None.
            temperature (float, optional): The sampling temperature for the model. If not specified,
                the provider's default value for the model is used. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. If not specified,
                the provider's default value for the model is used. Defaults to None.
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

        Raises:
            ValueError: If the provided model is not active and/or not available.

        Returns:
            str: The model's completion for the prompt (correctness not guaranteed).
        """
        if provider not in self.get_available_providers():
            raise ValueError(f"Invalid provider: {provider}")
        if provider not in self.active_providers:
            raise ValueError(f"Provider not active: {provider}")

        # Get the param info from the first model where the provider matches.
        # Not ideal, but the best we can do for user-provided models.
        model_info: ModelEntry = {
            "model_id": model_id,
            "params": MODEL_INFO[
                next(
                    model
                    for model in self.get_available_models()
                    if provider in MODEL_INFO[model].keys()
                )
            ][provider]["params"],
            "extras": {},
        }

        return await self._call_impl(
            model_info,
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
        )

    async def _call_impl(
        self,
        model_info: ModelEntry,
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
        _add_param(params, model_info["params"], "temperature", temperature)
        _add_param(params, model_info["params"], "max_tokens", max_tokens)

        # Handle native JSON mode
        has_native_json_mode = "json_mode_arg" in model_info["extras"]
        if json_mode and has_native_json_mode:
            arg = model_info["extras"]["json_mode_arg"]
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
            model_info["model_id"],
            prompt,
            system_prompt,
            params,
            timeout,
            memory,
            json_mode,
            json_mode_strategy,
            model_info["extras"],
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
        return await self._generic_openai_spec_call("openai", *args)

    async def _call_google(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
        timeout: Optional[int],
        memory: Optional[BaseMemory],
        *_: Any,  # json_mode and json_mode_strategy, and extras are not used here
    ) -> str:
        data: Dict[str, Any] = {}

        if system_prompt is not None:
            # Earlier models don't support system prompts, so prepend it to the prompt
            if model_id not in ["gemini-1.5-pro"]:
                prompt = f"{system_prompt}\n{prompt}"
            else:
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
        )
        return str(result["content"][0]["text"])

    async def _call_cohere(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
        timeout: Optional[int],
        memory: Optional[BaseMemory],
        json_mode: bool,
        json_mode_strategy: JsonModeStrategy,
        _: Dict[str, Any],  # extras is not used here
    ) -> str:
        if system_prompt is not None:
            params["preamble"] = system_prompt
        if isinstance(memory, ChatMemory):
            params["chat_history"] = memory.unpack("role", "message", "USER", "CHATBOT")

        if json_mode:
            append_msg = get_extra_message(json_mode_strategy)
            if append_msg:
                entry = {"role": "CHATBOT", "message": append_msg}
                params.setdefault("chat_history", []).append(entry)

        result = await llm_post(
            client=self.httpx_client,
            provider="cohere",
            model_id=model_id,
            api_key=self.api_keys["cohere"],
            data={"model": model_id, "message": prompt, **params},
            timeout=timeout,
        )
        return str(result["text"])

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
        _: bool,  # json_mode is not used here
        json_mode_strategy: JsonModeStrategy,
        __: Dict[str, Any],  # extras is not used here
    ) -> str:
        if isinstance(memory, ChatMemory):
            raise LLMOperationError(
                "Chat memory is not supported with Replicate."
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
        )
        return "".join(result["output"])

    async def _generic_openai_spec_call(
        self,
        provider: str,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
        timeout: Optional[int],
        memory: Optional[BaseMemory],
        json_mode: bool,
        json_mode_strategy: JsonModeStrategy,
        extras: Dict[str, Any],
    ) -> str:
        """Generic call method for providers who follow the OpenAI API spec."""
        supports_native_json_mode = "json_mode_arg" in extras

        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
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
        )
        return str(result["choices"][0]["message"]["content"])


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
        raise ValueError(msg)

    key = str(name) if (key := param_info[name].get("custom_key")) is None else key

    if value is not None:
        params[key] = value
    elif (default := param_info[name]["default"]) != PROVIDER_DEFAULT:
        params[key] = default
