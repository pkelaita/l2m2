# L2M2: A Simple Python LLM Manager 💬👍

**[L2M2](https://pypi.org/project/l2m2/)** ("LLM Manager" &rarr; "LLMM" &rarr; "L2M2") is a very simple LLM manager for Python that allows you to expose lots of models through a single API. This is useful for evaluation, demos, and production LLM apps that use multiple models.

## Supported Models

L2M2 currently supports the following models:

| Provider                                     | Model Name        | Model Version              |
| -------------------------------------------- | ----------------- | -------------------------- |
| [`openai`](https://openai.com/product)       | `gpt-4-turbo`     | `gpt-4-0125-preview`       |
| [`google`](https://ai.google.dev/)           | `gemini-1.5-pro`  | `gemini-1.5-pro-latest`    |
| [`google`](https://ai.google.dev/)           | `gemini-1.0-pro`  | `gemini-1.0-pro-latest`    |
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

client = LLMClient()
```

**Add a Provider**

In order to activate any of the available models, you must add the provider of that model and pass in your API key for that provider's API. Make sure to pass in a valid provider as shown in the table above.

```python
client.add_provider("<provider name>", "<API key>")
```

**Call your LLM 💬👍**

The `call` API is the same regardless of model or provider. Make sure to pass in a valid model name as shown in the table above.

```python
response = client.call(
    system_prompt="<system prompt>",
    prompt="<prompt>",
    model="<model name>",
    temperature=<temperature>,
)
```

`system_prompt` and `temperature` are optional, and default to `None` and `0.0` respectively.

### Example

```python
import os
from dotenv import load_dotenv
from l2m2 import LLMClient

load_dotenv()


client = LLMClient()
client.add_provider("openai", os.getenv("OPENAI_API_KEY"))

response = client.call(
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
