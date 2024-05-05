"""Information about models and providers supported by L2M2."""

from typing import Any, Dict

PROVIDER_DEFAULT = "<<PROVIDER_DEFAULT>>"


PROVIDER_INFO = {
    "openai": {
        "name": "OpenAI",
        "homepage": "https://openai.com/product",
    },
    "google": {
        "name": "Google",
        "homepage": "https://ai.google.dev/",
    },
    "anthropic": {
        "name": "Anthropic",
        "homepage": "https://www.anthropic.com/api",
    },
    "cohere": {
        "name": "Cohere",
        "homepage": "https://docs.cohere.com/",
    },
    "groq": {
        "name": "Groq",
        "homepage": "https://wow.groq.com/",
    },
    "replicate": {
        "name": "Replicate",
        "homepage": "https://replicate.com/",
    },
}

MODEL_INFO: Dict[str, Any] = {
    "gpt-4-turbo": {
        "openai": {
            "model_id": "gpt-4-turbo-2024-04-09",
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
                    "default": PROVIDER_DEFAULT,
                    "max": 4096,
                },
            },
        },
    },
    "gemini-1.5-pro": {
        "google": {
            "model_id": "gemini-1.5-pro-latest",
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
    },
    "gemini-1.0-pro": {
        "google": {
            "model_id": "gemini-1.0-pro-latest",
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
                    "default": 1000,  # L2M2 default, field is required
                    "max": 4096,
                },
            },
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
                    "default": 1000,  # L2M2 default, field is required
                    "max": 4096,
                },
            },
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
                    "default": 1000,  # L2M2 default, field is required
                    "max": 4096,
                },
            },
        },
    },
    "command-r": {
        "cohere": {
            "model_id": "command-r",
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
    },
    "command-r-plus": {
        "cohere": {
            "model_id": "command-r-plus",
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
        },
    },
    "gemma-7b": {
        "groq": {
            "model_id": "gemma-7b-it",
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
    },
    "llama3-8b": {
        "groq": {
            "model_id": "llama3-8b-8192",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**16 - 1,
                },
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
                    "max": float("inf"),
                },
            },
        },
    },
    "llama3-70b": {
        "groq": {
            "model_id": "llama3-70b-8192",
            "params": {
                "temperature": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2,
                },
                "max_tokens": {
                    "default": PROVIDER_DEFAULT,
                    "max": 2**16 - 1,
                },
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
                    "max": float("inf"),
                },
            },
        },
    },
}
