from typing import Set, Dict, Optional

import google.generativeai as google
from cohere import Client as CohereClient
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq


_MODEL_INFO: Dict[str, Dict[str, str]] = {
    "gpt-4-turbo": {
        "provider": "openai",
        "model_id": "gpt-4-turbo-2024-04-09",
        "provider_homepage": "https://openai.com/product",
    },
    "gpt-4-turbo-0125": {
        "provider": "openai",
        "model_id": "gpt-4-0125-preview",
        "provider_homepage": "https://openai.com/product",
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "model_id": "gemini-1.5-pro-latest",
        "provider_homepage": "https://ai.google.dev/",
    },
    "gemini-1.0-pro": {
        "provider": "google",
        "model_id": "gemini-1.0-pro-latest",
        "provider_homepage": "https://ai.google.dev/",
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "model_id": "claude-3-opus-20240229",
        "provider_homepage": "https://www.anthropic.com/api",
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-sonnet-20240229",
        "provider_homepage": "https://www.anthropic.com/api",
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "provider_homepage": "https://www.anthropic.com/api",
    },
    "command-r": {
        "provider": "cohere",
        "model_id": "command-r",
        "provider_homepage": "https://docs.cohere.com/",
    },
    "command-r-plus": {
        "provider": "cohere",
        "model_id": "command-r-plus",
        "provider_homepage": "https://docs.cohere.com/",
    },
    "llama2-70b": {
        "provider": "groq",
        "model_id": "llama2-70b-4096",
        "provider_homepage": "https://wow.groq.com/",
    },
    "mixtral-8x7b": {
        "provider": "groq",
        "model_id": "mixtral-8x7b-32768",
        "provider_homepage": "https://wow.groq.com/",
    },
    "gemma-7b": {
        "provider": "groq",
        "model_id": "gemma-7b-it",
        "provider_homepage": "https://wow.groq.com/",
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
        return set([info["provider"] for info in _MODEL_INFO.values()])

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
        prompt: str,
        model: str,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
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

        model_info = _MODEL_INFO[model]
        call_impl = getattr(self, f"_call_{model_info['provider']}", None)
        if call_impl is None:
            raise ValueError(f"Malformed model info entry: {model_info}")
        result = call_impl(model_info, prompt, temperature, system_prompt)
        assert isinstance(result, str)
        return result

    def _call_openai(
        self,
        model_info: Dict[str, str],
        prompt: str,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        oai = OpenAI(api_key=self.API_KEYS["openai"])
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        result = oai.chat.completions.create(
            model=model_info["model_id"],
            messages=messages,  # type: ignore
            temperature=temperature,
        )
        return str(result.choices[0].message.content)

    def _call_anthropic(
        self,
        model_info: Dict[str, str],
        prompt: str,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        anthr = Anthropic(api_key=self.API_KEYS["anthropic"])
        result = anthr.messages.create(
            model=model_info["model_id"],
            max_tokens=1000,
            temperature=temperature,
            system=system_prompt,  # type: ignore
            messages=[{"role": "user", "content": prompt}],
        )
        return str(result.content[0].text)

    def _call_cohere(
        self,
        model_info: Dict[str, str],
        prompt: str,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        cohere = CohereClient(api_key=self.API_KEYS["cohere"])
        result = cohere.chat(
            model=model_info["model_id"],
            message=prompt,
            preamble=system_prompt,
            temperature=temperature,
        )
        return str(result.text)

    def _call_groq(
        self,
        model_info: Dict[str, str],
        prompt: str,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        groq = Groq(api_key=self.API_KEYS["groq"])
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        result = groq.chat.completions.create(
            model=model_info["model_id"],
            messages=messages,  # type: ignore
            temperature=temperature,
        )
        return str(result.choices[0].message.content)

    def _call_google(
        self,
        model_info: Dict[str, str],
        prompt: str,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        google.configure(api_key=self.API_KEYS["google"])

        # Earlier versions don't support system prompts
        if model_info["model_id"] not in ["gemini-1.5-pro-latest"]:
            prompt = f"{system_prompt}\n{prompt}"
            model = google.GenerativeModel(model_name=model_info["model_id"])
        else:
            model = google.GenerativeModel(
                model_name=model_info["model_id"], system_instruction=system_prompt
            )

        config = {"max_output_tokens": 2048, "temperature": temperature, "top_p": 1}
        result = model.generate_content(prompt, generation_config=config)
        return str(result.candidates[0].content.parts[0].text)
