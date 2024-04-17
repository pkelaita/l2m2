# L2M2: A Simple Python LLM Manager 💬👍

**[L2M2](https://pypi.org/project/l2m2/)** ("LLM Manager" &rarr; "LLMM" &rarr; "L2M2") is a very simple LLM manager for Python that exposes lots of models through a unified API. This is useful for evaluation, demos, and other apps that need to easily be model-agnostic.

## Features

- 12 supported models (see below), with more on the way
- Asynchronous and concurrent calls
- User-provided models from supported providers

#### Supported Models

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

#### Planned Featires

- Support for Huggingface & open-source LLMs
- Chat-specific features (e.g. context, history, etc)
- Typescript clone
- ...etc

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

#### Example

```python
# example.py

import os
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
>> python3 example.py

Arrr, matey! The skies be clear as the Caribbean waters today, with the sun blazin' high 'bove us. A fine day fer settin' sail and huntin' fer treasure, it be. But keep yer eye on the horizon, for the weather can turn quicker than a sloop in a squall. Yarrr!
```

### Async Calls

L2M2 utilizes `asyncio` to allow for multiple concurrent calls. This is useful for calling multiple models at with the same prompt, calling the same model with multiple prompts, mixing and matching parameters, etc.

`AsyncLLMClient`, which extends `LLMClient`, is provided for this purpose. Its usage is similar to above:

```python
# example_async.py

import asyncio
import os
from l2m2.client import AsyncLLMClient

client = AsyncLLMClient({
    "openai": os.getenv("OPENAI_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
})


async def make_two_calls():
    responses = await asyncio.gather(
        client.call_async(
            model="gpt-4-turbo",
            prompt="How's the weather today?",
            system_prompt="Respond as if you were a pirate.",
            temperature=0.3,
            max_tokens=100,
        ),
        client.call_async(
            model="gemini-1.0-pro",
            prompt="How's the weather today?",
            system_prompt="Respond as if you were a pirate.",
            temperature=0.3,
            max_tokens=100,
        ),
    )
    for response in responses:
        print(response)


if __name__ == "__main__":
    asyncio.run(make_two_calls())
```

```
>> python3 example_async.py

Arrr, the skies be clear and the winds be in our favor, matey! A fine day for sailin' the high seas, it be.
Avast there, matey! The weather be fair and sunny, with a gentle breeze from the east. The sea be calm, and the sky be clear. A perfect day for sailin' and plunderin'!
```

For convenience `AsyncLLMClient` also provides `call_concurrent`, which allows you to easily make concurrent calls mixing and matching models, prompts, and parameters. In the example shown below, parameter arrays of size `n` are applied linearly to the `n` concurrent calls, and arrays of size `1` are applied across all `n` calls.

```python
# example_concurrent.py

import asyncio
import os
from l2m2.client import AsyncLLMClient

client = AsyncLLMClient({
    "openai": os.getenv("OPENAI_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "cohere": os.getenv("COHERE_API_KEY"),
})


async def multiple_models_same_prompt():
    responses = await client.call_concurrent(
        n=3,
        models=["gpt-4-turbo", "gemini-1.0-pro", "command-r"],
        prompts=["What is your name, and which company made your model?"],
        system_prompts=["Your name is Bob, and you respond to questions briefly."],
        temperatures=[0.4, 0.5, 0.7],
        max_tokens=[75],
    )

    for response in responses:
        print(response)

if __name__ == "__main__":
    asyncio.run(multiple_models_same_prompt())
```

```
>> python3 example_concurrent.py

My name is Bob, and OpenAI created my model.
Bob; Google
My name is Bob, and I am a product of Cohere, a company that focuses on developing outstanding AI technology.
```

Similarly to `call_custom`, `call_custom_async` and `call_custom_concurrent` are provided as the custom counterparts to `call_async` and `call_concurrent`, with similar usage.

## Contact

If you'd like to contribute, have feature requests, or have any other questions about l2m2 please shoot me a note at [pierce@kelaita.com](mailto:pierce@kelaita.com), open an issue on the [Github repo](https://github.com/pkelaita/l2m2/issues), or DM me on the GenAI Collective [Slack Channel](https://join.slack.com/t/genai-collective/shared_invite/zt-285qq7joi-~bqHwFZcNtqntoRmGirAfQ).
