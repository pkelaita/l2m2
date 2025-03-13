"""Information about models and providers supported by L2M2."""

from typing import Any, Dict, Union, Literal, TypedDict
import sys

PROVIDER_DEFAULT: Literal["<<PROVIDER_DEFAULT>>"] = "<<PROVIDER_DEFAULT>>"

API_KEY = "<<API_KEY>>"
MODEL_ID = "<<MODEL_ID>>"
SERVICE_BASE_URL = "<<SERVICE_BASE_URL>>"

INF: int = sys.maxsize

ParamName = Literal["temperature", "max_tokens"]


class ParamOptionalFields(TypedDict, total=False):
    custom_key: str


class FloatParam(ParamOptionalFields):
    default: Union[float, Literal["<<PROVIDER_DEFAULT>>"]]
    max: float


class IntParam(ParamOptionalFields):
    default: Union[int, Literal["<<PROVIDER_DEFAULT>>"]]
    max: int


class ModelParams(TypedDict):
    temperature: FloatParam
    max_tokens: IntParam


class ModelEntry(TypedDict):
    model_id: str
    params: ModelParams
    extras: Dict[str, Any]


class GenericModelEntry(TypedDict):
    params: ModelParams
    extras: Dict[str, Any]


class ProviderEntry(TypedDict):
    name: str
    homepage: str
    endpoint: str
    headers: Dict[str, str]


class LocalProviderEntry(TypedDict):
    name: str
    homepage: str
    endpoint: str
    headers: Dict[str, str]
    default_base_url: str
    model_entry: GenericModelEntry


HOSTED_PROVIDERS: Dict[str, ProviderEntry] = {
    "openai": {
        "name": "OpenAI",
        "homepage": "https://openai.com/api/",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "headers": {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    },
    "google": {
        "name": "Google",
        "homepage": "https://ai.google.dev/",
        "endpoint": f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={API_KEY}",
        "headers": {"Content-Type": "application/json"},
    },
    "anthropic": {
        "name": "Anthropic",
        "homepage": "https://www.anthropic.com/api",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "headers": {
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    },
    "cohere": {
        "name": "Cohere",
        "homepage": "https://docs.cohere.com/",
        "endpoint": "https://api.cohere.com/v2/chat",
        "headers": {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    },
    "mistral": {
        "name": "Mistral",
        "homepage": "https://docs.mistral.ai/deployment/laplateforme/overview/",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "headers": {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    },
    "groq": {
        "name": "Groq",
        "homepage": "https://wow.groq.com/",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "headers": {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    },
    "replicate": {
        "name": "Replicate",
        "homepage": "https://replicate.com/",
        "endpoint": f"https://api.replicate.com/v1/models/{MODEL_ID}/predictions",
        "headers": {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    },
    "cerebras": {
        "name": "Cerebras",
        "homepage": "https://inference-docs.cerebras.ai",
        "endpoint": "https://api.cerebras.ai/v1/chat/completions",
        "headers": {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    },
}

LOCAL_PROVIDERS: Dict[str, LocalProviderEntry] = {
    "ollama": {
        "name": "Ollama",
        "homepage": "https://ollama.ai/",
        "endpoint": f"{SERVICE_BASE_URL}/api/chat",
        "headers": {"Content-Type": "application/json"},
        "default_base_url": "http://localhost:11434",
        "model_entry": {
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": INF,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": INF,
                },
            },
            "extras": {"json_mode_arg": {"format": "json"}},
        },
    },
}

MODEL_INFO: Dict[str, Dict[str, ModelEntry]] = {
    "gpt-4.5": {
        "openai": {
            "model_id": "gpt-4.5-preview-2025-02-27",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "custom_key": "max_completion_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 2**14,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "o3-mini": {
        "openai": {
            "model_id": "o3-mini-2025-01-31",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "custom_key": "max_completion_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "o1": {
        "openai": {
            "model_id": "o1-2024-12-17",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "custom_key": "max_completion_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "o1-preview": {
        "openai": {
            "model_id": "o1-preview-2024-09-12",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "custom_key": "max_completion_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {},
        },
    },
    "o1-mini": {
        "openai": {
            "model_id": "o1-mini-2024-09-12",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "custom_key": "max_completion_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {},
        },
    },
    "gpt-4o": {
        "openai": {
            "model_id": "gpt-4o-2024-11-20",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_completion_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "gpt-4o-mini": {
        "openai": {
            "model_id": "gpt-4o-mini-2024-07-18",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_completion_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "gpt-4-turbo": {
        "openai": {
            "model_id": "gpt-4-turbo-2024-04-09",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_completion_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "gpt-3.5-turbo": {
        "openai": {
            "model_id": "gpt-3.5-turbo-0125",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_completion_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "gemini-2.0-pro": {
        "google": {
            "model_id": "gemini-2.0-pro-exp-02-05",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models
                    "max": 8192,
                },
            },
            "extras": {"json_mode_arg": {"response_mime_type": "application/json"}},
        },
    },
    "gemini-2.0-flash": {
        "google": {
            "model_id": "gemini-2.0-flash-001",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models
                    "max": 8192,
                },
            },
            "extras": {"json_mode_arg": {"response_mime_type": "application/json"}},
        },
    },
    "gemini-2.0-flash-lite": {
        "google": {
            "model_id": "gemini-2.0-flash-lite-preview-02-05",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models
                    "max": 8192,
                },
            },
            "extras": {"json_mode_arg": {"response_mime_type": "application/json"}},
        },
    },
    "gemini-1.5-flash": {
        "google": {
            "model_id": "gemini-1.5-flash-001",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models
                    "max": 8192,
                },
            },
            "extras": {"json_mode_arg": {"response_mime_type": "application/json"}},
        },
    },
    "gemini-1.5-flash-8b": {
        "google": {
            "model_id": "gemini-1.5-flash-8b",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models
                    "max": 8192,
                },
            },
            "extras": {"json_mode_arg": {"response_mime_type": "application/json"}},
        },
    },
    "gemini-1.5-pro": {
        "google": {
            "model_id": "gemini-1.5-pro",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models
                    "max": 8192,
                },
            },
            "extras": {"json_mode_arg": {"response_mime_type": "application/json"}},
        },
    },
    "claude-3.7-sonnet": {
        "anthropic": {
            "model_id": "claude-3-7-sonnet-20250219",
            "params": {
                "temperature": {
                    "default": 0.0,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": 4096,  # L2M2 default, field is required
                    "max": 128000,
                },
            },
            "extras": {},
        },
    },
    "claude-3.5-sonnet": {
        "anthropic": {
            "model_id": "claude-3-5-sonnet-20241022",
            "params": {
                "temperature": {
                    "default": 0.0,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": 4096,  # L2M2 default, field is required
                    "max": 8192,
                },
            },
            "extras": {},
        },
    },
    "claude-3.5-haiku": {
        "anthropic": {
            "model_id": "claude-3-5-haiku-20241022",
            "params": {
                "temperature": {
                    "default": 0.0,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": 4096,  # L2M2 default, field is required
                    "max": 8192,
                },
            },
            "extras": {},
        },
    },
    "claude-3-opus": {
        "anthropic": {
            "model_id": "claude-3-opus-20240229",
            "params": {
                "temperature": {
                    "default": 0.0,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": 2048,  # L2M2 default, field is required
                    "max": 4096,
                },
            },
            "extras": {},
        },
    },
    "claude-3-sonnet": {
        "anthropic": {
            "model_id": "claude-3-sonnet-20240229",
            "params": {
                "temperature": {
                    "default": 0.0,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": 2048,  # L2M2 default, field is required
                    "max": 4096,
                },
            },
            "extras": {},
        },
    },
    "claude-3-haiku": {
        "anthropic": {
            "model_id": "claude-3-haiku-20240307",
            "params": {
                "temperature": {
                    "default": 0.0,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": 2048,  # L2M2 default, field is required
                    "max": 4096,
                },
            },
            "extras": {},
        },
    },
    "command-a": {
        "cohere": {
            "model_id": "command-a-03-2025",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**13,
                },
            },
            "extras": {},
        },
    },
    "command-r-plus": {
        "cohere": {
            "model_id": "command-r-plus-08-2024",
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
            "extras": {},
        },
    },
    "command-r": {
        "cohere": {
            "model_id": "command-r-08-2024",
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
            "extras": {},
        },
    },
    "command-r7b": {
        "cohere": {
            "model_id": "command-r7b-12-2024",
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
            "extras": {},
        },
    },
    "mistral-large": {
        "mistral": {
            "model_id": "mistral-large-2411",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": INF,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "mistral-small": {
        "mistral": {
            "model_id": "mistral-small-2501",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": INF,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "ministral-3b": {
        "mistral": {
            "model_id": "ministral-3b-2410",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": INF,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "ministral-8b": {
        "mistral": {
            "model_id": "ministral-8b-2410",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": INF,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "mixtral-8x7b": {
        "groq": {
            "model_id": "mixtral-8x7b-32768",
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
            "extras": {},
        },
    },
    "gemma-2-9b": {
        "groq": {
            "model_id": "gemma2-9b-it",
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
            "extras": {},
        },
    },
    "llama-3.3-70b": {
        "groq": {
            "model_id": "llama-3.3-70b-versatile",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**15,
                },
            },
            "extras": {"preview": True},
        },
        "cerebras": {
            "model_id": "llama3.3-70b",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**31 - 1,
                },
            },
            "extras": {},
        },
    },
    "llama-3.2-3b": {
        "groq": {
            "model_id": "llama-3.2-3b-preview",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**13,
                },
            },
            "extras": {"preview": True},
        },
    },
    "llama-3.2-1b": {
        "groq": {
            "model_id": "llama-3.2-1b-preview",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**13,
                },
            },
            "extras": {"preview": True},
        },
    },
    "llama-3.1-405b": {
        "replicate": {
            "model_id": "meta/meta-llama-3.1-405b-instruct",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": INF,
                },
                "max_tokens": {
                    "custom_key": "max_new_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": INF,
                },
            },
            "extras": {},
        },
    },
    "llama-3.1-8b": {
        "groq": {
            "model_id": "llama-3.1-8b-instant",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 8000,
                },
            },
            "extras": {},
        },
        "cerebras": {
            "model_id": "llama3.1-8b",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**31 - 1,
                },
            },
            "extras": {},
        },
    },
    "llama-3-70b": {
        "groq": {
            "model_id": "llama3-70b-8192",
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
            "extras": {},
        },
        "replicate": {
            "model_id": "meta/meta-llama-3-70b-instruct",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 5.0,
                },
                "max_tokens": {
                    "custom_key": "max_new_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": INF,
                },
            },
            "extras": {},
        },
    },
    "llama-3-8b": {
        "groq": {
            "model_id": "llama3-8b-8192",
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
            "extras": {},
        },
        "replicate": {
            "model_id": "meta/meta-llama-3-8b-instruct",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 5.0,
                },
                "max_tokens": {
                    "custom_key": "max_new_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": INF,
                },
            },
            "extras": {},
        },
    },
    "qwen-2.5-32b": {
        "groq": {
            "model_id": "qwen-2.5-32b",
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
            "extras": {},
        },
    },
    "deepseek-r1-distill-qwen-32b": {
        "groq": {
            "model_id": "deepseek-r1-distill-qwen-32b",
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
            "extras": {},
        },
    },
    "deepseek-r1-distill-llama-70b": {
        "groq": {
            "model_id": "deepseek-r1-distill-llama-70b",
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
            "extras": {},
        },
    },
}


def get_id(provider: str, model_id: str) -> str:
    return MODEL_INFO[model_id][provider]["model_id"]
