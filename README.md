# L2M2: A Simple Python LLM Manager 💬👍

**[L2M2](https://pypi.org/project/l2m2/)** ("LLM Manager" &rarr; "LLMM" &rarr; "L2M2") is a very simple LLM manager for Python that allows you to expose lots of models through a single API. This is useful for evaluation, demos, and production LLM apps that use multiple models.

## Supported Models

L2M2 currently supports the following models:

| Provider                                     | Model Name         | Model Version              |
| -------------------------------------------- | ------------------ | -------------------------- |
| [`openai`](https://openai.com/product)       | `gpt-4-turbo`      | `gpt-4-turbo-2024-04-09`   |
| [`openai`](https://openai.com/product)       | `gpt-4-turbo-0125` | `gpt-4-0125-preview`       |
| [`google`](https://ai.google.dev/)           | `gemini-1.5-pro`   | `gemini-1.5-pro-latest`    |
| [`google`](https://ai.google.dev/)           | `gemini-1.0-pro`   | `gemini-1.0-pro-latest`    |
| [`anthropic`](https://www.anthropic.com/api) | `claude-3-opus`    | `claude-3-opus-20240229`   |
| [`anthropic`](https://www.anthropic.com/api) | `claude-3-sonnet`  | `claude-3-sonnet-20240229` |
| [`anthropic`](https://www.anthropic.com/api) | `claude-3-haiku`   | `claude-3-haiku-20240307`  |
| [`cohere`](https://docs.cohere.com/)         | `command-r`        | `command-r`                |
| [`cohere`](https://docs.cohere.com/)         | `command-r-plus`   | `command-r-plus`           |
| [`groq`](https://wow.groq.com/)              | `llama2-70b`       | `llama2-70b-4096`          |
| [`groq`](https://wow.groq.com/)              | `mixtral-8x7b`     | `mixtral-8x7b-32768`       |
| [`groq`](https://wow.groq.com/)              | `gemma-7b`         | `gemma-7b-it`              |

You can also call any language model from the above providers that L2M2 doesn't officially support, without guarantees of well-defined behavior.

## Requirements

- Python >= 3.12

## Installation

```sh
pip install l2m2
```

## Usage

**Import the LLM Client**

```python
from l2m2.client import LLMClient
```

**Add Providers**

In order to activate any of the available models, you must add the provider of that model and pass in your API key for that provider's API. Make sure to pass in a valid provider as shown in the table above.

```python
client = LLMClient()
client.add_provider("<provider-name>", "<api-key>")

# Alternatively, you can pass in providers via the constructor
client = LLMClient({
    "<provider-a>": "<api-key-a>",
    "<provider-b>": "<api-key-b>",
    ...
})
```

**Call your LLM 💬👍**

The `call` API is the same regardless of model or provider.

```python
response = client.call(
    model="<model name>",
    prompt="<prompt>",
    system_prompt="<system prompt>",
    temperature=<temperature>,
    max_tokens=<max_tokens>
)
```

`model` and `prompt` are required, while the remaining fields are optional. When possible, L2M2 uses the provider's default model parameter values when they are not given.

If you'd like to call a language model from one of the supported providers that isn't officially supported by L2M2 (for example, older models such as `gpt-3.5-turbo`), you can similarly `call_custom` with the additional required parameter `provider`, and pass in the model name expected by the provider's API. Unlike `call`, `call_custom` doesn't guarantee correctness or well-defined behavior.

### Example

```python
from l2m2.client import LLMClient

client = LLMClient()
client.add_provider("openai", os.getenv("OPENAI_API_KEY"))

response = client.call(
    model="gpt-4-turbo",
    prompt="How's the weather today?",
    system_prompt="Respond as if you were a pirate.",
    temperature=0.5,
    max_tokens=250,
)

print(response)
```

```
Arrr, matey! The skies be clear as the Caribbean waters today, with the sun blazin' high 'bove us. A fine day fer settin' sail and huntin' fer treasure, it be. But keep yer eye on the horizon, for the weather can turn quicker than a sloop in a squall. Yarrr!
```
