"""Information about models and providers supported by L2M2."""

from typing import Any, Dict

ModelInfo = Dict[str, Any]

PROVIDER_DEFAULT = "<<PROVIDER_DEFAULT>>"

MODEL_INFO: Dict[str, ModelInfo] = {
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
                "custom_key": "max_output_tokens",
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
                "custom_key": "max_output_tokens",
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
    "llama3-8b": {
        "provider": "replicate",
        "model_id": "meta/meta-llama-3-8b",
        "provider_homepage": "https://replicate.com/",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": 5.0,
            },
            "max_tokens": {
                "custom_key": "max_new_tokens",
                "default": PROVIDER_DEFAULT,
                "max": float("inf"),
            },
        },
    },
    "llama3-8b-instruct": {
        "provider": "replicate",
        "model_id": "meta/meta-llama-3-8b-instruct",
        "provider_homepage": "https://replicate.com/",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": 5.0,
            },
            "max_tokens": {
                "custom_key": "max_new_tokens",
                "default": PROVIDER_DEFAULT,
                "max": float("inf"),
            },
        },
    },
    "llama3-70b": {
        "provider": "replicate",
        "model_id": "meta/meta-llama-3-70b",
        "provider_homepage": "https://replicate.com/",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": float("inf"),
            },
            "max_tokens": {
                "default": PROVIDER_DEFAULT,
                "max": float("inf"),
            },
        },
    },
    "llama3-70b-instruct": {
        "provider": "replicate",
        "model_id": "meta/meta-llama-3-70b-instruct",
        "provider_homepage": "https://replicate.com/",
        "params": {
            "temperature": {
                "default": PROVIDER_DEFAULT,
                "max": float("inf"),
            },
            "max_tokens": {
                "default": PROVIDER_DEFAULT,
                "max": float("inf"),
            },
        },
    },
}
