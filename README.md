# L2M2: A Simple Python LLM Manager 💬👍

[![Tests](https://github.com/pkelaita/l2m2/actions/workflows/tests.yml/badge.svg?timestamp=1718864754)](https://github.com/pkelaita/l2m2/actions/workflows/tests.yml) [![codecov](https://codecov.io/github/pkelaita/l2m2/graph/badge.svg?token=UWIB0L9PR8)](https://codecov.io/github/pkelaita/l2m2) [![PyPI version](https://badge.fury.io/py/l2m2.svg?timestamp=1718864754)](https://badge.fury.io/py/l2m2)

**L2M2** ("LLM Manager" &rarr; "LLMM" &rarr; "L2M2") is a very simple LLM manager for Python that exposes lots of models through a unified API. This is useful for evaluation, demos, and other apps that need to easily be model-agnostic.

## Features

- <!--start-count-->14<!--end-count--> supported models (see below) through a unified interface – regularly updated and with more on the way
- Asynchronous and concurrent calls
- Session chat memory – even across multiple models
- JSON mode
- Optional prompt loader

### Supported Models

L2M2 currently supports the following models:

<!--start-model-table-->

| Model Name        | Provider(s)                                                        | Model Version(s)                                    |
| ----------------- | ------------------------------------------------------------------ | --------------------------------------------------- |
| `gpt-4o`          | [OpenAI](https://openai.com/product)                               | `gpt-4o-2024-05-13`                                 |
| `gpt-4-turbo`     | [OpenAI](https://openai.com/product)                               | `gpt-4-turbo-2024-04-09`                            |
| `gpt-3.5-turbo`   | [OpenAI](https://openai.com/product)                               | `gpt-3.5-turbo-0125`                                |
| `gemini-1.5-pro`  | [Google](https://ai.google.dev/)                                   | `gemini-1.5-pro-latest`                             |
| `gemini-1.0-pro`  | [Google](https://ai.google.dev/)                                   | `gemini-1.0-pro-latest`                             |
| `claude-3-opus`   | [Anthropic](https://www.anthropic.com/api)                         | `claude-3-opus-20240229`                            |
| `claude-3-sonnet` | [Anthropic](https://www.anthropic.com/api)                         | `claude-3-sonnet-20240229`                          |
| `claude-3-haiku`  | [Anthropic](https://www.anthropic.com/api)                         | `claude-3-haiku-20240307`                           |
| `command-r`       | [Cohere](https://docs.cohere.com/)                                 | `command-r`                                         |
| `command-r-plus`  | [Cohere](https://docs.cohere.com/)                                 | `command-r-plus`                                    |
| `mixtral-8x7b`    | [Groq](https://wow.groq.com/)                                      | `mixtral-8x7b-32768`                                |
| `gemma-7b`        | [Groq](https://wow.groq.com/)                                      | `gemma-7b-it`                                       |
| `llama3-8b`       | [Groq](https://wow.groq.com/), [Replicate](https://replicate.com/) | `llama3-8b-8192`, `meta/meta-llama-3-8b-instruct`   |
| `llama3-70b`      | [Groq](https://wow.groq.com/), [Replicate](https://replicate.com/) | `llama3-70b-8192`, `meta/meta-llama-3-70b-instruct` |

<!--end-model-table-->

### Planned Features

- Support for OSS and self-hosted (Hugging Face, Gpt4all, etc.)
- Basic (i.e., customizable & non-opinionated) agent & multi-agent system features
- HTTP-based calls instead of SDKs (this bring's L2M2's dependencies from ~50 to <10)
- Typescript clone (probably not soon)
- ...etc

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- **Usage**
  - [Basic Usage](#usage)
  - [Multi-Provider Models](#multi-provider-models)
  - **Memory**
    - [Chat Memory](#memory)
    - [External Memory](#external-memory)
  - [Async Calls](#async-calls)
  - **Tools**
    - [JSON Mode](#tools-json-mode)
    - [Prompt Loader](#tools-prompt-loader)
- [Contact](#contact)

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
client = LLMClient({
    "provider-a": "api-key-a",
    "provider-b": "api-key-b",
    ...
})

# Alternatively, you can add providers after initialization
client.add_provider("provider-c", "api-key-c")
```

**Call your LLM 💬👍**

The `call` API is the same regardless of model or provider.

```python
response = client.call(
    model="<model name>",
    prompt="<prompt>",
)
```

`model` and `prompt` are required, while `system_prompt`, `temperature`, and `max_tokens` are optional. When possible, L2M2 uses the provider's default model parameter values when they are not given.

```python
response = client.call(
    model="<model name>",
    prompt="<prompt>",
    system_prompt="<system prompt>",
    temperature=<temperature>,
    max_tokens=<max tokens>,
)
```

If you'd like to call a language model from one of the supported providers that isn't officially supported by L2M2 (for example, older models such as `gpt-4-0125-preview`), you can similarly `call_custom` with the additional required parameter `provider`, and pass in the model name expected by the provider's API. Unlike `call`, `call_custom` doesn't guarantee correctness or well-defined behavior.

### Example

```python
# example.py

import os
from l2m2.client import LLMClient

client = LLMClient()
client.add_provider("openai", os.getenv("OPENAI_API_KEY"))

response = client.call(
    model="gpt-4o",
    prompt="How's the weather today?",
    system_prompt="Respond as if you were a pirate.",
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

### Memory

L2M2 provides a simple memory system that allows you to maintain context and history across multiple calls and multiple models. There are two types of memory: **`ChatMemory`**, which natively hooks into models' conversation history, and **`ExternalMemory`**, which allows for custom memory implementations. Let's first take a look at `ChatMemory`.

```python
from l2m2.client import LLMClient
from l2m2.memory import MemoryType

# Use the MemoryType enum to specify the type of memory you want to use
client = LLMClient({
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
}, memory_type=MemoryType.CHAT)

print(client.call(model="gpt-4o", prompt="My name is Pierce"))
print(client.call(model="claude-3-haiku", prompt="I am a software engineer."))
print(client.call(model="llama3-8b", prompt="What's my name?"))
print(client.call(model="mixtral-8x7b", prompt="What's my job?"))
```

```
Hello, Pierce! How can I help you today?
A software engineer, you say? That's a noble profession.
Your name is Pierce.
You are a software engineer.
```

Chat memory is stored per session, with a sliding window of messages which defaults to the last 40 – this can be configured by passing `memory_window_size` to the client constructor.

You can access the client's memory using `client.get_memory()`. Once accessed, `ChatMemory` lets you add user and agent messages, clear the memory, and access the memory as a list of messages.

```python
client = LLMClient({"openai": os.getenv("OPENAI_API_KEY")}, memory_type=MemoryType.CHAT)

memory = client.get_memory() # ChatMemory object
memory.add_user_message("My favorite color is red.")
memory.add_user_message("My least favorite color is green.")
memory.add_agent_message("Ok, duly noted.")

print(client.call(model="gpt-4o", prompt="What are my favorite and least favorite colors?"))
memory.clear()
print(client.call(model="gpt-4o", prompt="What are my favorite and least favorite colors?"))
```

```
Your favorite color is red, and your least favorite color is green.
I'm sorry, I don't have that information.
```

You can also load in a memory object on the fly using `load_memory`, which will enable memory if none is already loaded, and overwrite the existing memory if it is.

```python

client = LLMClient({"openai": os.getenv("OPENAI_API_KEY")}, memory_type=MemoryType.CHAT)
client.call(model="gpt-4o", prompt="My favorite color is red.")
print(client.call(model="gpt-4o", prompt="What is my favorite color?"))

new_memory = ChatMemory()
new_memory.add_user_message("My favorite color is blue.")
new_memory.add_agent_message("Ok, noted.")

client.load_memory(memory)
print(client.call(model="gpt-4o", prompt="What is my favorite color?"))
```

```
Your favorite color is red.
Your favorite color is blue.
```

#### External Memory

**`ExternalMemory`** is a simple but powerful memory mode that allows you to define your own memory implementation. This can be useful for more complex memory constructions (e.g., planning, reflecting) or for implementing custom persistence (e.g., saving memory to a database or a file). Its usage is much like `ChatMemory`, but unlike `ChatMemory` you must manage initializing and updating the memory yourself with `get_contents` and `set_contents`.

Here's a simple example of a custom memory implementation that has a description and a list of previous user/model message pairs:

```python
# example_external_memory.py

from l2m2.client import LLMClient
from l2m2.memory import MemoryType

client = LLMClient({"openai": os.getenv("OPENAI_API_KEY")}, memory_type=MemoryType.EXTERNAL)

messages = [
    "My name is Pierce",
    "I am a software engineer",
    "What is my name?",
    "What is my profession?",
]

def update_memory(user_input, model_output):
    memory = client.get_memory() # ExternalMemory object
    contents = memory.get_contents()
    if contents == "":
        contents = "You are mid-conversation with me. Your memory of it is below:\n\n"
    contents += f"Me: {user_input}\nYou: {model_output}\n"
    memory.set_contents(contents)

for message in messages:
    response = client.call(model="gpt-4o", prompt=message)
    print(response)
    update_memory(message, response)
```

```
>> python3 example_external_memory.py

Nice to meet you, Pierce!
Nice! What kind of projects do you work on?
Your name is Pierce.
You are a software engineer.
```

By default, `ExternalMemory` contents are appended to the system prompt, or passed in as the system prompt if one is not given. Generally, models perform best when external memory is stored in the system prompt; however, you can configure the client to append the memory contents to the user prompt instead as follows:

```python
from l2m2.memory import ExternalMemoryLoadingType

client = LLMClient(
    {"openai": os.getenv("OPENAI_API_KEY")},
    memory_type=MemoryType.EXTERNAL,
    memory_loading_type=ExternalMemoryLoadingType.USER_PROMPT_APPEND,
)
```

Similarly to `ChatMemory`, `ExternalMemory` can be passed into `client.load_memory` to load in new custom memory on the fly, and can be shared across multiple models and providers.

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
            model="gpt-4o",
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
            "gpt-4o",
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

Similarly to `call_custom`, `call_custom_async` and `call_custom_concurrent` are provided as the custom counterparts to `call_async` and `call_concurrent`, with similar usage. `AsyncLLMClient` also supports memory in the same way as `LLMClient`.

### Tools: JSON Mode

L2M2 provides an optional `json_mode` flag that enforces JSON formatting on LLM responses. Importantly, this flag is applicable to all models and providers, whether or not they natively support JSON output enforcement. When JSON mode is not natively supported, `json_mode` will apply strategies to maximize the likelihood of valid JSON output.

```python
# example_json_mode.py

response = client.call(
    model="gpt-4o",
    prompt="What are the capitals of each state of Australia?",
    system_prompt="Respond with the JSON format {'region': 'capital'}",
    json_mode=True,
)

print(response)
```

```
>> python3 example_json_mode.py

{
  "New South Wales": "Sydney",
  "Victoria": "Melbourne",
  "Queensland": "Brisbane",
  "South Australia": "Adelaide",
  "Western Australia": "Perth",
  "Tasmania": "Hobart",
  "Northern Territory": "Darwin",
  "Australian Capital Territory": "Canberra"
}
```

> [!IMPORTANT]
> Regardless of the model and even when `json_mode` is enabled, it's crucial to ensure that either the prompt or the system prompt mentions to return the output in JSON - and ideally, to specify the JSON format, as shown above.

The following models natively support JSON mode:

- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`
- `gemini-1.5-pro`

#### JSON Mode Non-Native Strategies

For models that do not natively support JSON mode, L2M2 will attempt to enforce JSON formatting by applying one of the following two strategies under the hood:

1. **Strip**: This is the default strategy. It will attempt to extract the JSON from the response by searching for the first instance of `{` and the last instance of `}` in the response, and returning the between substring (inclusive). If no JSON is found, the response will be returned as-is.
2. **Prepend**: This strategy will attempt to enforce a valid JSON output by inserting a message ending with an opening `{` from the model into the conversation just after the user prompt and just before the model response, and re-prepending the opening `{` to the model response. By default this message is `"Here is the JSON output:"`, but can be customized. More information is available on this strategy [here](https://github.com/anthropics/anthropic-cookbook/blob/main/misc/how_to_enable_json_mode.ipynb). Importantly, the **Prepend** strategy is available whether or not memory is enabled, and will not interfere with memory.

**Strip** is the default strategy, but you can specify a strategy by passing either `JsonModeStrategy.strip()` or `JsonModeStrategy.prepend()` to the `json_mode_strategy` parameter in `call`.

```python
# example_json_mode.py

from l2m2.client import LLMClient
from l2m2.tools import JsonModeStrategy

client = LLMClient({"anthropic": os.getenv("ANTHROPIC_API_KEY")})

response = client.call(
    model="claude-3-sonnet",
    prompt="What are the capitals of each Canadian province?",
    system_prompt="Respond with the JSON format {'region': 'capital'}",
    json_mode=True,
    json_mode_strategy=JsonModeStrategy.prepend(),
)

print(response)
```

```
>> python3 example_json_mode.py

{
  "Alberta": "Edmonton",
  "British Columbia": "Victoria",
  "Manitoba": "Winnipeg",
  "New Brunswick": "Fredericton",
  "Newfoundland and Labrador": "St. John's",
  "Nova Scotia": "Halifax",
  "Ontario": "Toronto",
  "Prince Edward Island": "Charlottetown",
  "Quebec": "Quebec City",
  "Saskatchewan": "Regina"
}
```

Finally, you can customize the message that gets passed into the prepend strategy by passing `custom_prefix` as follows:

```python

response = client.call(
    model="claude-3-sonnet",
    prompt="What are the capitals of each Canadian province?",
    system_prompt="Respond with the JSON format {'region': 'capital'}",
    json_mode=True,
    json_mode_strategy=JsonModeStrategy.prepend(custom_prefix="Here is the JSON with provinces and capitals:"),
)
```

Ideally, this wouldn't change anything on the output – just under the hood – but this is useful for working with foreign languages, etc.

> [!TIP]
> I _highly_ recommend using `prepend()` when calling Anthropic's models, and sticking with the default `strip()` for all other models that don't natively support JSON mode. From my personal testing, valid JSON is almost always produced when using `prepend()` with Anthropic's models and almost never produced with `strip()`, and vice versa for other models. I'll gather rigorous data on this eventually, but if anyone has any insights, please let me know!

### Tools: Prompt Loader

L2M2 provides an optional prompt-loading utility that's useful for loading prompts with variables from a file. Usage is simple:

_prompt.txt_

```
Your name is {{name}} and you are a {{profession}}.
```

```python
# example_prompt_loader.py

from l2m2.tools import PromptLoader

loader = PromptLoader()
prompt = loader.load_prompt(
    prompt_file="prompt.txt",
    variables={"name": "Pierce", "profession": "software engineer"},
)
print(prompt)
```

```
>> python3 example_prompt_loader.py

Your name is Pierce and you are a software engineer.
```

You can also optionally specify a prompt directory or customize the variable delimiters if needed.

_path/to/prompts/prompt.txt_

```
Your name is <<name>> and you are a <<profession>>.
```

```python
# example_prompt_loader.py

from l2m2.tools import PromptLoader

loader = PromptLoader(
    prompts_base_dir="path/to/prompts",
    variable_delimiters=("<<", ">>"),
)
prompt = loader.load_prompt(
    prompt_file="prompt.txt",
    variables={"name": "Pierce", "profession": "software engineer"},
)
print(prompt)
```

```
>> python3 example_prompt_loader.py

Your name is Pierce and you are a software engineer.
```

## Contact

If you'd like to contribute, have feature requests, or have any other questions about l2m2 please shoot me a note at [pierce@kelaita.com](mailto:pierce@kelaita.com), open an issue on the [Github repo](https://github.com/pkelaita/l2m2/issues), or DM me on the GenAI Collective [Slack Channel](https://join.slack.com/t/genai-collective/shared_invite/zt-285qq7joi-~bqHwFZcNtqntoRmGirAfQ).
