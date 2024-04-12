# L2M2: Simple LLM Manager for Python

[L2M2](https://pypi.org/project/l2m2/) ("LLM Manager" &rarr; "LLMM" &rarr; "L2M2") is a very simple LLM manager for Python.

## Supported Models

L2M2 currently supports the following models:

| Provider                                     | Model Name        | Model Version              |
| -------------------------------------------- | ----------------- | -------------------------- |
| [`openai`](https://openai.com/product)       | `gpt-4-turbo`     | `gpt-4-0125-preview`       |
| [`google`](https://ai.google.dev/)           | `gemini-1.5-pro`  |                            |
| [`google`](https://ai.google.dev/)           | `gemini-1.0-pro`  |                            |
| [`anthropic`](https://www.anthropic.com/api) | `claude-3-opus`   | `claude-3-opus-20240229`   |
| [`anthropic`](https://www.anthropic.com/api) | `claude-3-sonnet` | `claude-3-sonnet-20240229` |
| [`anthropic`](https://www.anthropic.com/api) | `claude-3-haiku`  | `claude-3-haiku-20240307`  |
| [`cohere`](https://docs.cohere.com/)         | `command-r`       | `command-r`                |
| [`cohere`](https://docs.cohere.com/)         | `command-r-plus`  | `command-r-plus`           |
| [`groq`](https://wow.groq.com/)              | `llama2-70b`      | `llama2-70b-4096`          |
| [`groq`](https://wow.groq.com/)              | `mixtral-8x7b`    | `mixtral-8x7b-32768`       |
| [`groq`](https://wow.groq.com/)              | `gemma-7b`        | `gemma-7b-it`              |

## Installation

```sh
pip install l2m2
```

## Usage

**Import the LLM Client**

```python
from l2m2 import LLMClient

llms = LLMClient()
```

**Add a Provider**

In order to activate any of the available models, you must add the provider of that model and pass in your API key for that provider's API. Make sure to use the provider name as shown in the table above.

```python
llms.add_provider("<provider name>", "<API key>")
```

**Call your LLM**

The `call` API is the same regardless of model or provider.

```python
response = llms.call(
    system_prompt="<system prompt>",
    prompt="<prompt>",
    model="<model name>",
    temperature=<temperature>,
)
```

`system_prompt` and `temperature` are optional, and default to `None` and `0.0` respectively.

**List Available Models and Providers**

These will return all valid models that can be passed into `call` and providers that can be passed into `add_provider`.

```python
print(llms.get_available_models())
print(llms.get_available_providers())
```

**List Active Models and Providers**

These will only return models and providers added with `add_provider`.

```python
print(llms.get_active_models())
print(llms.get_active_providers())
```

### Example

```python
import os
from dotenv import load_dotenv
from l2m2 import LLMClient

load_dotenv()


llms = LLMClient()
llms.add_provider("openai", os.getenv("OAI_APIKEY"))

response = llms.call(
    system_prompt="Respond as if you were a pirate.",
    prompt="How's the weather today?",
    model="gpt-4-turbo",
    temperature=0.5,
)

print(response)
```

```
Arrr, matey! The skies be clear as the Caribbean waters today, with the sun blazin' high 'bove us. A fine day fer settin' sail and huntin' fer treasure, it be. But keep yer eye on the horizon, for the weather can turn quicker than a sloop in a squall. Yarrr!
```
