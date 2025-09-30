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
        "endpoint": "https://api.openai.com/v1/responses",
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
    "gpt-5": {
        "openai": {
            "model_id": "gpt-5-2025-08-07",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 2**16,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
        },
    },
    "gpt-5-mini": {
        "openai": {
            "model_id": "gpt-5-mini-2025-08-07",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 2**16,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
        },
    },
    "gpt-5-nano": {
        "openai": {
            "model_id": "gpt-5-nano-2025-08-07",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 2**16,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
        },
    },
    "o4-mini": {
        "openai": {
            "model_id": "o4-mini-2025-04-16",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
        },
    },
    "o3-pro": {
        "openai": {
            "model_id": "o3-pro-2025-06-10",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
        },
    },
    "o3": {
        "openai": {
            "model_id": "o3-2025-04-16",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
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
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
        },
    },
    "o1-pro": {
        "openai": {
            "model_id": "o1-pro-2025-03-19",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
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
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
        },
    },
    "gpt-4.5": {
        "openai": {
            "model_id": "gpt-4.5-preview-2025-02-27",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 2**14,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
        },
    },
    "gpt-4.1": {
        "openai": {
            "model_id": "gpt-4.1-2025-04-14",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 2**15,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
        },
    },
    "gpt-4.1-mini": {
        "openai": {
            "model_id": "gpt-4.1-mini-2025-04-14",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 2**15,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
        },
    },
    "gpt-4.1-nano": {
        "openai": {
            "model_id": "gpt-4.1-nano-2025-04-14",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 2**15,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
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
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
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
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
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
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
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
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
            "extras": {"json_mode_arg": {"text": {"format": {"type": "json_object"}}}},
        },
    },
    "gemini-2.5-pro": {
        "google": {
            "model_id": "gemini-2.5-pro",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models
                    "max": 2**31 - 1,
                },
            },
            "extras": {"json_mode_arg": {"response_mime_type": "application/json"}},
        },
    },
    "gemini-2.5-flash": {
        "google": {
            "model_id": "gemini-2.5-flash",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 2**31 - 1,
                },
            },
            "extras": {"json_mode_arg": {"response_mime_type": "application/json"}},
        },
    },
    "gemini-2.5-flash-lite": {
        "google": {
            "model_id": "gemini-2.5-flash-lite",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "custom_key": "max_output_tokens",
                    "default": PROVIDER_DEFAULT,
                    "max": 2**31 - 1,
                },
            },
            "extras": {"json_mode_arg": {"response_mime_type": "application/json"}},
        },
    },
    "gemini-2.0-flash": {
        "google": {
            "model_id": "gemini-2.0-flash",
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
            "model_id": "gemini-2.0-flash-lite",
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
    "claude-opus-4.1": {
        "anthropic": {
            "model_id": "claude-opus-4-1-20250805",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": 4096,  # L2M2 default, field is required,
                    "max": 32000,
                },
            },
            "extras": {},
        },
    },
    "claude-opus-4": {
        "anthropic": {
            "model_id": "claude-opus-4-20250514",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": 4096,  # L2M2 default, field is required,
                    "max": 32000,
                },
            },
            "extras": {},
        },
    },
    "claude-sonnet-4.5": {
        "anthropic": {
            "model_id": "claude-sonnet-4-20250514",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": 4096,  # L2M2 default, field is required,
                    "max": 64000,
                },
            },
            "extras": {},
        },
    },
    "claude-sonnet-4": {
        "anthropic": {
            "model_id": "claude-sonnet-4-20250514",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": 4096,  # L2M2 default, field is required,
                    "max": 64000,
                },
            },
            "extras": {},
        },
    },
    "claude-3.7-sonnet": {
        "anthropic": {
            "model_id": "claude-3-7-sonnet-20250219",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.0,
                },
                "max_tokens": {
                    "default": 4096,  # L2M2 default, field is required
                    "max": 64000,
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
                    "default": PROVIDER_DEFAULT,
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
                    "default": PROVIDER_DEFAULT,
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
                    "default": PROVIDER_DEFAULT,
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
                    "default": PROVIDER_DEFAULT,
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
                    "default": PROVIDER_DEFAULT,
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
                    "max": 8000,
                },
            },
            "extras": {},
        },
    },
    "command-a-reasoning": {
        "cohere": {
            "model_id": "command-a-reasoning-08-2025",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 32000,
                },
            },
            "extras": {},
        },
    },
    "command-a-translate": {
        "cohere": {
            "model_id": "command-a-translate-08-2025",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 8000,
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
    "magistral-medium": {
        "mistral": {
            "model_id": "magistral-medium-2509",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**63 - 1,
                },
            },
            # JSON mode is technically supported on magistral models, but is incredibly
            # buggy so I'm choosing to not support it here...
            "extras": {},
        },
    },
    "magistral-small": {
        "mistral": {
            "model_id": "magistral-small-2509",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**63 - 1,
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
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**63 - 1,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "mistral-medium": {
        "mistral": {
            "model_id": "mistral-medium-2508",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**63 - 1,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "mistral-small": {
        "mistral": {
            "model_id": "mistral-small-2506",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**63 - 1,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "codestral": {
        "mistral": {
            "model_id": "codestral-2508",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**63 - 1,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "devstral-medium": {
        "mistral": {
            "model_id": "devstral-medium-2507",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**63 - 1,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "devstral-small": {
        "mistral": {
            "model_id": "devstral-small-2507",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**63 - 1,
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
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**63 - 1,
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
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**63 - 1,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
        },
    },
    "qwen-qwq-32b": {
        "groq": {
            "model_id": "qwen-qwq-32b",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2.0,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**17,
                },
            },
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
    },
    "mistral-saba": {
        "groq": {
            "model_id": "mistral-saba-24b",
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
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
        "mistral": {
            "model_id": "mistral-saba-2502",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 1.5,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": INF,
                },
            },
            "extras": {"json_mode_arg": {"response_format": {"type": "json_object"}}},
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
                    "max": 2**13,
                },
            },
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
    },
    "llama-4-maverick": {
        "groq": {
            "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct",
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
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
        "cerebras": {
            "model_id": "llama-4-maverick-17b-128e-instruct",
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
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
    },
    "llama-4-scout": {
        "groq": {
            "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
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
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
        "cerebras": {
            "model_id": "llama-4-scout-17b-16e-instruct",
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
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
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
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
        "cerebras": {
            "model_id": "llama-3.3-70b",
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
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
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
                    "max": 2**17,
                },
            },
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
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
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
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
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
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
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
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
    "gpt-oss-120b": {
        "groq": {
            "model_id": "openai/gpt-oss-120b",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**17,
                },
            },
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
        "cerebras": {
            "model_id": "gpt-oss-120b",
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
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
    },
    "gpt-oss-20b": {
        "groq": {
            "model_id": "openai/gpt-oss-20b",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**17,
                },
            },
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
    },
    "kimi-k2": {
        "groq": {
            "model_id": "moonshotai/kimi-k2-instruct-0905",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**14,
                },
            },
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
    },
    "qwen-3-480b": {
        "cerebras": {
            "model_id": "qwen-3-coder-480b",
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
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
    },
    "qwen-3-235b": {
        "cerebras": {
            "model_id": "qwen-3-235b-a22b-instruct-2507",
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
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
    },
    "qwen-3-235b-thinking": {
        "cerebras": {
            "model_id": "qwen-3-235b-a22b-thinking-2507",
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
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
    },
    "qwen-3-32b": {
        "cerebras": {
            "model_id": "qwen-3-32b",
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
            "extras": {
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
        "groq": {
            "model_id": "qwen/qwen3-32b",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 40960,
                },
            },
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
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
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
    },
    "allam-2-7b": {
        "groq": {
            "model_id": "allam-2-7b",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**12,
                },
            },
            "extras": {
                "preview": True,
                "json_mode_arg": {"response_format": {"type": "json_object"}},
            },
        },
    },
}
