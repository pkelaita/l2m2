# L2M2: A Simple Python LLM Manager 💬👍

[![Tests](https://github.com/pkelaita/l2m2/actions/workflows/tests.yml/badge.svg?timestamp=1714942588)](https://github.com/pkelaita/l2m2/actions/workflows/tests.yml) [![PyPI version](https://badge.fury.io/py/l2m2.svg?timestamp=1714942588)](https://badge.fury.io/py/l2m2)

**L2M2** ("LLM Manager" &rarr; "LLMM" &rarr; "L2M2") is a very simple LLM manager for Python that exposes lots of models through a unified API. This is useful for evaluation, demos, and other apps that need to easily be model-agnostic.

## Features

- 13 supported models (see below), with more on the way
- Asynchronous and concurrent calls
- User-provided models from supported providers

### Supported Models

L2M2 currently supports the following models:

| Model Name        | Provider(s)                                                        | Model Version(s)                                   |
| ----------------- | ------------------------------------------------------------------ | -------------------------------------------------- |
| `gpt-4-turbo`     | [OpenAI](https://openai.com/product)                               | `gpt-4-turbo-2024-04-09`                           |
| `gpt-3.5-turbo`   | [OpenAI](https://openai.com/product)                               | `gpt-3.5-turbo-0125`                               |
| `gemini-1.5-pro`  | [Google](https://ai.google.dev/)                                   | `gemini-1.5-pro-latest`                            |
| `gemini-1.0-pro`  | [Google](https://ai.google.dev/)                                   | `gemini-1.0-pro-latest`                            |
| `claude-3-opus`   | [Anthropic](https://www.anthropic.com/api)                         | `claude-3-opus-20240229`                           |
| `claude-3-sonnet` | [Anthropic](https://www.anthropic.com/api)                         | `claude-3-sonnet-20240229`                         |
| `claude-3-haiku`  | [Anthropic](https://www.anthropic.com/api)                         | `claude-3-haiku-20240307`                          |
| `command-r`       | [Cohere](https://docs.cohere.com/)                                 | `command-r`                                        |
| `command-r-plus`  | [Cohere](https://docs.cohere.com/)                                 | `command-r-plus`                                   |
| `mixtral-8x7b`    | [Groq](https://wow.groq.com/)                                      | `mixtral-8x7b-32768`                               |
| `gemma-7b`        | [Groq](https://wow.groq.com/)                                      | `gemma-7b-it`                                      |
| `llama3-8b`       | [Groq](https://wow.groq.com/), [Replicate](https://replicate.com/) | `llama3-8b-8192`, `meta/meta-llama-3-8b-instruct`  |
| `llama3-70b`      | [Groq](https://wow.groq.com/), [Replicate](https://replicate.com/) | `llama3-70b-8192`, `meta/meta-llama-3-8b-instruct` |

You can also call any language model from the above providers that L2M2 doesn't officially support, without guarantees of well-defined behavior.

### Planned Featires

- Support for Huggingface & open-source LLMs
- Chat-specific features (e.g. context, history, etc)
- Typescript clone
- ...etc

## Requirements

- Python >= 3.9

## Installation

```
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

### Multi-Provider Models

Some models are available from multiple providers, such as `llama3-70b` from both Groq and Replicate. When multiple of such providers are active, you can use the parameter `prefer_provider` to specify which provider to use for a given inference.

```python
client.add_provider("groq", os.getenv("GROQ_API_KEY"))
client.add_provider("replicate", os.getenv("REPLICATE_API_TOKEN"))

response1 = client.call(
    model="llama3-70b",
    prompt="Hello there",
    prefer_provider="groq",
) # Uses Groq

response2 = client.call(
    model="llama3-70b",
    prompt="General Kenobi!",
    prefer_provider="replicate",
) # Uses Replicate
```

You can also set default preferred providers for the client using `set_preferred_providers`, to avoid having to specify `prefer_provider` for each call.

```python
client.set_preferred_providers({
    "llama3-70b": "groq",
    "llama3-8b": "replicate",
})

response1 = client.call(model="llama3-70b", prompt="Hello there") # Uses Groq
response2 = client.call(model="llama3-8b", prompt="General Kenobi!") # Uses Replicate
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
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "cohere": os.getenv("COHERE_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
    "replicate": os.getenv("REPLICATE_API_TOKEN"),
})

# Since llama3-8b is available from both Groq and Replicate
client.set_preferred_providers({"llama3-8b": "replicate"})

async def get_secret_word():
    system_prompt = "The secret word is {0}. When asked for the secret word, you must respond with {0}."
    responses = await client.call_concurrent(
        n=6,
        models=[
            "gpt-4-turbo",
            "claude-3-sonnet",
            "gemini-1.0-pro",
            "command-r",
            "mixtral-8x7b",
            "llama3-8b",
        ],
        prompts=["What is the secret word?"],
        system_prompts=[
            system_prompt.format("foo"),
            system_prompt.format("bar"),
            system_prompt.format("baz"),
            system_prompt.format("qux"),
            system_prompt.format("quux"),
            system_prompt.format("corge"),
        ],
        temperatures=[0.3],
        max_tokens=[100],
    )

    for response in responses:
        print(response)

if __name__ == "__main__":
    asyncio.run(get_secret_word())
```

```
>> python3 example_concurrent.py

foo
The secret word is bar.
baz
qux
The secret word is quux. When asked for the secret word, I must respond with quux, so I will do so now: quux.
The secret word is... corge!
```

Similarly to `call_custom`, `call_custom_async` and `call_custom_concurrent` are provided as the custom counterparts to `call_async` and `call_concurrent`, with similar usage.

## Contact

If you'd like to contribute, have feature requests, or have any other questions about l2m2 please shoot me a note at [pierce@kelaita.com](mailto:pierce@kelaita.com), open an issue on the [Github repo](https://github.com/pkelaita/l2m2/issues), or DM me on the GenAI Collective [Slack Channel](https://join.slack.com/t/genai-collective/shared_invite/zt-285qq7joi-~bqHwFZcNtqntoRmGirAfQ).
