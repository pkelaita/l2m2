import google.generativeai as google
from cohere import Client as CohereClient
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq


_MODEL_INFO = {
    "gpt-4-turbo": {
        "provider": "openai",
        "model_id": "gpt-4-0125-preview",
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "model_id": "gemini-1.5-pro-latest",
    },
    "gemini-1.0-pro": {
        "provider": "google",
        "model_id": "gemini-1.0-pro-latest",
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "model_id": "claude-3-opus-20240229",
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-sonnet-20240229",
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
    },
    "command-r": {
        "provider": "cohere",
        "model_id": "command-r",
    },
    "command-r-plus": {
        "provider": "cohere",
        "model_id": "command-r-plus",
    },
    "llama2-70b": {
        "provider": "groq",
        "model_id": "llama2-70b-4096",
    },
    "mixtral-8x7b": {
        "provider": "groq",
        "model_id": "mixtral-8x7b-32768",
    },
    "gemma-7b": {
        "provider": "groq",
        "model_id": "gemma-7b-it",
    },
}

_PROVIDERS = set([info["provider"] for info in _MODEL_INFO.values()])


class LLMClient:

    def __init__(self):
        self.API_KEYS = {}
        self.active_providers = set()
        self.active_models = set()

    def add_provider(self, provider, api_key):
        if provider not in _PROVIDERS:
            msg = f"Invalid provider: {provider}. Must be one of {_PROVIDERS}"
            raise ValueError(msg)

        self.API_KEYS[provider] = api_key
        self.active_providers.add(provider)
        self.active_models.update(
            model for model, info in _MODEL_INFO.items() if info["provider"] == provider
        )

    def get_active_providers(self):
        return set(self.active_providers)

    def get_active_models(self):
        return set(self.active_models)

    def get_available_providers(self):
        return set(_PROVIDERS)

    def get_available_models(self):
        return set(_MODEL_INFO.keys())

    def remove_provider(self, provider):
        if provider not in self.active_providers:
            raise ValueError(f"Provider not active: {provider}")

        del self.API_KEYS[provider]
        self.active_providers.remove(provider)
        self.active_models.difference_update(
            model for model, info in _MODEL_INFO.items() if info["provider"] == provider
        )

    def call(self, *, prompt, model, temperature=0.0, system_prompt=None):
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

        if model_info["provider"] == "openai":
            oai = OpenAI(api_key=self.API_KEYS["openai"])
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            result = oai.chat.completions.create(
                model=model_info["model_id"],
                messages=messages,
                temperature=temperature,
            )
            return result.choices[0].message.content

        elif model_info["provider"] == "anthropic":
            anthr = Anthropic(api_key=self.API_KEYS["anthropic"])
            result = anthr.messages.create(
                model=model_info["model_id"],
                max_tokens=1000,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return result.content[0].text

        elif model_info["provider"] == "cohere":
            cohere = CohereClient(api_key=self.API_KEYS["cohere"])
            result = cohere.chat(
                model=model_info["model_id"],
                message=prompt,
                preamble=system_prompt,
                temperature=temperature,
            )
            return result.text

        elif model_info["provider"] == "groq":
            groq = Groq(api_key=self.API_KEYS["groq"])
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            result = groq.chat.completions.create(
                model=model_info["model_id"],
                messages=messages,
                temperature=temperature,
            )
            return result.choices[0].message.content

        elif model_info["provider"] == "google":
            print("here")
            google.configure(api_key=self.API_KEYS["google"])
            if model_info["model_id"] not in ["gemini-1.5-pro-latest"]:
                prompt = f"{system_prompt}\n{prompt}"
                model = google.GenerativeModel(model_name=model_info["model_id"])
            else:
                model = google.GenerativeModel(
                    model_name=model_info["model_id"],
                    system_instruction=system_prompt,
                )
            config = {"max_output_tokens": 2048, "temperature": temperature, "top_p": 1}
            response = model.generate_content(prompt, generation_config=config)
            return response.candidates[0].content.parts[0].text
