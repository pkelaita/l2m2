from typing import Any, Set, Dict, Optional
import pydash as py_

import google.generativeai as google
from cohere import Client as CohereClient
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq


ModelInfo = Dict[str, Any]

PROVIDER_DEFAULT = "<<PROVIDER_DEFAULT>>"

_MODEL_INFO: Dict[str, ModelInfo] = {
    "gpt-4-turbo": {
        "provider": "openai",
        "model_id": "gpt-4-turbo-2024-04-09",
        "provider_homepage": "https://openai.com/product",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": 2.0,
            },
            "max_tokens": {
                "default": PROVIDER_DEFAULT,
                "max": 4096,
            },
        },
    },
    "gpt-4-turbo-0125": {
        "provider": "openai",
        "model_id": "gpt-4-0125-preview",
        "provider_homepage": "https://openai.com/product",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": 2.0,
            },
            "max_tokens": {
                "default": PROVIDER_DEFAULT,
                "max": 4096,
            },
        },
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "model_id": "gemini-1.5-pro-latest",
        "provider_homepage": "https://ai.google.dev/",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": 2.0,
            },
            "max_tokens": {
                "default": PROVIDER_DEFAULT,
                # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models
                "max": 8192,
            },
        },
    },
    "gemini-1.0-pro": {
        "provider": "google",
        "model_id": "gemini-1.0-pro-latest",
        "provider_homepage": "https://ai.google.dev/",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": 2.0,
            },
            "max_tokens": {
                "default": PROVIDER_DEFAULT,
                # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models
                "max": 8192,
            },
        },
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "model_id": "claude-3-opus-20240229",
        "provider_homepage": "https://www.anthropic.com/api",
        "params": {
            "temperature": {
                "default": 0.0,
                "max": 1.0,
            },
            "max_tokens": {
                "default": 1000,  # L2M2 default, field is required
                "max": 4096,
            },
        },
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-sonnet-20240229",
        "provider_homepage": "https://www.anthropic.com/api",
        "params": {
            "temperature": {
                "default": 0.0,
                "max": 1.0,
            },
            "max_tokens": {
                "default": 1000,  # L2M2 default, field is required
                "max": 4096,
            },
        },
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "provider_homepage": "https://www.anthropic.com/api",
        "params": {
            "temperature": {
                "default": 0.0,
                "max": 1.0,
            },
            "max_tokens": {
                "default": 1000,  # L2M2 default, field is required
                "max": 4096,
            },
        },
    },
    "command-r": {
        "provider": "cohere",
        "model_id": "command-r",
        "provider_homepage": "https://docs.cohere.com/",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": 1.0,
            },
            "max_tokens": {
                "default": PROVIDER_DEFAULT,
                "max": 4000,
            },
        },
    },
    "command-r-plus": {
        "provider": "cohere",
        "model_id": "command-r-plus",
        "provider_homepage": "https://docs.cohere.com/",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": 1.0,
            },
            "max_tokens": {
                "default": PROVIDER_DEFAULT,
                "max": 4000,
            },
        },
    },
    "llama2-70b": {
        "provider": "groq",
        "model_id": "llama2-70b-4096",
        "provider_homepage": "https://wow.groq.com/",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": 2.0,
            },
            "max_tokens": {
                "default": PROVIDER_DEFAULT,
                "max": 2**16 - 1,
            },
        },
    },
    "mixtral-8x7b": {
        "provider": "groq",
        "model_id": "mixtral-8x7b-32768",
        "provider_homepage": "https://wow.groq.com/",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": 2.0,
            },
            "max_tokens": {
                "default": PROVIDER_DEFAULT,
                "max": 2**16 - 1,
            },
        },
    },
    "gemma-7b": {
        "provider": "groq",
        "model_id": "gemma-7b-it",
        "provider_homepage": "https://wow.groq.com/",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": 2.0,
            },
            "max_tokens": {
                "default": PROVIDER_DEFAULT,
                "max": 2**16 - 1,
            },
        },
    },
}


class LLMClient:
    def __init__(self) -> None:
        self.API_KEYS: Dict[str, str] = {}
        self.active_providers: Set[str] = set()
        self.active_models: Set[str] = set()

    @classmethod
    def with_providers(cls, providers: Dict[str, str]) -> "LLMClient":
        obj = cls()
        for provider, api_key in providers.items():
            obj.add_provider(provider, api_key)
        return obj

    @staticmethod
    def get_available_providers() -> Set[str]:
        return set([str(info["provider"]) for info in _MODEL_INFO.values()])

    @staticmethod
    def get_available_models() -> Set[str]:
        return set(_MODEL_INFO.keys())

    def get_active_providers(self) -> Set[str]:
        return set(self.active_providers)

    def get_active_models(self) -> Set[str]:
        return set(self.active_models)

    def add_provider(self, provider: str, api_key: str) -> None:
        if provider not in (providers := self.get_available_providers()):
            raise ValueError(
                f"Invalid provider: {provider}. Available providers: {providers}"
            )

        self.API_KEYS[provider] = api_key
        self.active_providers.add(provider)
        self.active_models.update(
            model for model, info in _MODEL_INFO.items() if info["provider"] == provider
        )

    def remove_provider(self, provider: str) -> None:
        if provider not in self.active_providers:
            raise ValueError(f"Provider not active: {provider}")

        del self.API_KEYS[provider]
        self.active_providers.remove(provider)
        self.active_models.difference_update(
            model for model, info in _MODEL_INFO.items() if info["provider"] == provider
        )

    def call(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if model not in self.active_models:
            if model in self.get_available_models():
                provider = _MODEL_INFO[model]["provider"]
                msg = (
                    f"Model {model} is available, but not active."
                    + f" Please add provider {provider} to activate it."
                )
                raise ValueError(msg)
            else:
                raise ValueError(f"Invalid model: {model}")

        result = self._call_impl(
            _MODEL_INFO[model], prompt, system_prompt, temperature, max_tokens
        )
        return result

    def call_custom(
        self,
        *,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if provider not in self.get_available_providers():
            raise ValueError(f"Invalid provider: {provider}")
        if provider not in self.active_providers:
            raise ValueError(f"Provider not active: {provider}")

        model_info = {
            "provider": provider,
            "model_id": model,
            # Get the param info from the first model where the provider matches.
            # Not ideal, but the best we can do for user-provided models.
            **py_.pick(
                py_.find(list(_MODEL_INFO.values()), {"provider": provider}),
                "params",
            ),
        }

        result = self._call_impl(
            model_info, prompt, system_prompt, temperature, max_tokens
        )
        return result

    def _call_impl(
        self,
        model_info: ModelInfo,
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

            if value is not None:
                params[name] = value

            elif (default := param_info[name]["default"]) != PROVIDER_DEFAULT:
                params[name] = default

        add_param("temperature", temperature)
        add_param("max_tokens", max_tokens)

        call_provider = getattr(self, f"_call_{model_info['provider']}")
        result = call_provider(model_info["model_id"], prompt, system_prompt, params)
        assert isinstance(result, str)
        return result

    def _call_openai(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        params: Dict[str, Any],
    ) -> str:
        oai = OpenAI(api_key=self.API_KEYS["openai"])
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
        anthr = Anthropic(api_key=self.API_KEYS["anthropic"])
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
        cohere = CohereClient(api_key=self.API_KEYS["cohere"])
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
        groq = Groq(api_key=self.API_KEYS["groq"])
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
        google.configure(api_key=self.API_KEYS["google"])

        model_params = {"model_name": model_id}
        if system_prompt is not None:
            # Earlier versions don't support system prompts, so prepend it to the prompt
            if model_id not in ["gemini-1.5-pro-latest"]:
                prompt = f"{system_prompt}\n{prompt}"
            else:
                model_params["system_instruction"] = system_prompt
        model = google.GenerativeModel(**model_params)

        config_map = {
            "temperature": "temperature",
            "max_tokens": "max_output_tokens",
        }
        config = {config_map[k]: v for k, v in params.items() if k in config_map}

        result = model.generate_content(prompt, generation_config=config)
        result = result.candidates[0]

        # Will sometimes fail due to safety filters
        if result.content:
            return str(result.content.parts[0].text)
        else:
            return str(result)
