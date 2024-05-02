from typing import Any, Set, Dict, Optional

import google.generativeai as google
from cohere import Client as CohereClient
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
import replicate

from l2m2.model_info import (
    MODEL_INFO,
    PROVIDER_INFO,
    PROVIDER_DEFAULT,
)


class LLMClient:
    """A high-level interface for interacting with L2M2's supported language models."""

    def __init__(self, providers: Optional[Dict[str, str]] = None) -> None:
        """Initialize the LLMClient, optionally with active providers.

        Args:
            providers ([Dict[str, str]], optional): Mapping from provider name to API key.
                For example::

                    {
                        "openai": "openai-api
                        "anthropic": "anthropic-api-key",
                        "google": "google-api-key",
                    }

                Defaults to None.

        Raises:
            ValueError: If an invalid provider is specified in `providers`.
        """
        self.api_keys: Dict[str, str] = {}
        self.active_providers: Set[str] = set()
        self.active_models: Set[str] = set()
        self.preferred_providers: Dict[str, str] = {}

        if providers is not None:
            for provider, api_key in providers.items():
                self.add_provider(provider, api_key)

    @staticmethod
    def get_available_providers() -> Set[str]:
        """Get the exhaustive set of L2M2's available model providers. This set includes
        all providers, regardless of whether they are currently active. L2M2 does not currently
        support adding custom providers.

        Returns:
            Set[str]: A set of available providers.
        """
        # return set([str(info["provider"]) for info in MODEL_INFO.values()])
        return set(PROVIDER_INFO.keys())

    @staticmethod
    def get_available_models() -> Set[str]:
        """The set of L2M2's supported models. This set includes all models, regardless of
        whether they are currently active. L2M2 allows users to call non-available models
        from available and active providers, but does not guarantee correctness for such calls.

        Returns:
            Set[str]: A set of available models.
        """
        # return set(MODEL_INFO.keys())
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
        """
        if provider not in (providers := self.get_available_providers()):
            raise ValueError(
                f"Invalid provider: {provider}. Available providers: {providers}"
            )

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
                        "llama3-8b": "groq",
                        "llama3-70b": "replicate",
                    }
                If you'd like to remove a preferred provider, set it to `None`, e.g.::
                    {
                        "llama3-8b": None,
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

    def call(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        prefer_provider: Optional[str] = None,
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

        result = self._call_impl(
            MODEL_INFO[model][provider],
            provider,
            prompt,
            system_prompt,
            temperature,
            max_tokens,
        )
        return result

    def call_custom(
        self,
        *,
        provider: str,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
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
        model_info = {
            "model_id": model_id,
            "params": MODEL_INFO[
                next(
                    model
                    for model in self.get_available_models()
                    if provider in MODEL_INFO[model].keys()
                )
            ][provider]["params"],
        }

        result = self._call_impl(
            model_info, provider, prompt, system_prompt, temperature, max_tokens
        )
        return result

    def _call_impl(
        self,
        model_info: Dict[str, Any],
        provider: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        param_info = model_info["params"]
        params = {}

        def add_param(name: str, value: Any) -> None:
            if value is not None and value > (max_val := param_info[name]["max"]):
                msg = f"Parameter {name} exceeds max value {max_val}"
                raise ValueError(msg)

            key = name
            if "custom_key" in param_info[name]:
                key = param_info[name]["custom_key"]

            if value is not None:
                params[key] = value
            elif (default := param_info[name]["default"]) != PROVIDER_DEFAULT:
                params[key] = default

        add_param("temperature", temperature)
        add_param("max_tokens", max_tokens)

        result = getattr(self, f"_call_{provider}")(
            model_info["model_id"], prompt, system_prompt, params
        )
        assert isinstance(result, str), "This should never happen."
        return result

    def _call_openai(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
    ) -> str:
        oai = OpenAI(api_key=self.api_keys["openai"])
        messages = [{"role": "user", "content": prompt}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        result = oai.chat.completions.create(
            model=model_id,
            messages=messages,  # type: ignore
            **params,
        )
        return str(result.choices[0].message.content)

    def _call_anthropic(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
    ) -> str:
        anthr = Anthropic(api_key=self.api_keys["anthropic"])
        if system_prompt is not None:
            params["system"] = system_prompt
        result = anthr.messages.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            **params,
        )
        return str(result.content[0].text)

    def _call_cohere(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
    ) -> str:
        cohere = CohereClient(api_key=self.api_keys["cohere"])
        if system_prompt is not None:
            params["preamble"] = system_prompt
        result = cohere.chat(
            model=model_id,
            message=prompt,
            **params,
        )
        return str(result.text)

    def _call_groq(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
    ) -> str:
        groq = Groq(api_key=self.api_keys["groq"])
        messages = [{"role": "user", "content": prompt}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        result = groq.chat.completions.create(
            model=model_id,
            messages=messages,  # type: ignore
            **params,
        )
        return str(result.choices[0].message.content)

    def _call_google(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
    ) -> str:
        google.configure(api_key=self.api_keys["google"])

        model_params = {"model_name": model_id}
        if system_prompt is not None:
            # Earlier versions don't support system prompts, so prepend it to the prompt
            if model_id not in ["gemini-1.5-pro-latest"]:
                prompt = f"{system_prompt}\n{prompt}"
            else:
                model_params["system_instruction"] = system_prompt
        model = google.GenerativeModel(**model_params)

        result = model.generate_content(prompt, generation_config=params)
        result = result.candidates[0]

        # Will sometimes fail due to safety filters
        if result.content:
            return str(result.content.parts[0].text)
        else:
            return str(result)

    def _call_replicate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
    ) -> str:
        client = replicate.Client(api_token=self.api_keys["replicate"])
        if system_prompt is not None:
            params["system_prompt"] = system_prompt
        result = client.run(
            model_id,
            input={
                "prompt": prompt,
                **params,
            },
        )
        return "".join(result)
