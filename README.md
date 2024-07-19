# L2M2: A Simple Python LLM Manager 💬👍

[![Tests](https://github.com/pkelaita/l2m2/actions/workflows/tests.yml/badge.svg?timestamp=1721419447)](https://github.com/pkelaita/l2m2/actions/workflows/tests.yml) [![codecov](https://codecov.io/github/pkelaita/l2m2/graph/badge.svg?token=UWIB0L9PR8)](https://codecov.io/github/pkelaita/l2m2) [![PyPI version](https://badge.fury.io/py/l2m2.svg?timestamp=1721419447)](https://badge.fury.io/py/l2m2)

**L2M2** ("LLM Manager" &rarr; "LLMM" &rarr; "L2M2") is a tiny and very simple LLM manager for Python that exposes lots of models through a unified API. This is useful for evaluation, demos, production applications etc. that need to easily be model-agnostic.

### Features

- <!--start-count-->16<!--end-count--> supported models (see below) – regularly updated and with more on the way
- Session chat memory – even across multiple models
- JSON mode
- Prompt loading tools

### Advantages

- **Simple:** Completely unified interface – just swap out the model name
- **Tiny:** only two external dependencies (httpx and typing_extensions)
- **Fast**: Fully asynchronous if concurrent calls are needed

### Supported Models

L2M2 currently supports the following models:

<!--start-model-table-->

| Model Name          | Provider(s)                                                        | Model Version(s)                                    |
| ------------------- | ------------------------------------------------------------------ | --------------------------------------------------- |
| `gpt-4o`            | [OpenAI](https://openai.com/product)                               | `gpt-4o-2024-05-13`                                 |
| `gpt-4o-mini`       | [OpenAI](https://openai.com/product)                               | `gpt-4o-mini-2024-07-18`                            |
| `gpt-4-turbo`       | [OpenAI](https://openai.com/product)                               | `gpt-4-turbo-2024-04-09`                            |
| `gpt-3.5-turbo`     | [OpenAI](https://openai.com/product)                               | `gpt-3.5-turbo-0125`                                |
| `gemini-1.5-pro`    | [Google](https://ai.google.dev/)                                   | `gemini-1.5-pro`                                    |
| `gemini-1.0-pro`    | [Google](https://ai.google.dev/)                                   | `gemini-1.0-pro`                                    |
| `claude-3.5-sonnet` | [Anthropic](https://www.anthropic.com/api)                         | `claude-3-5-sonnet-20240620`                        |
| `claude-3-opus`     | [Anthropic](https://www.anthropic.com/api)                         | `claude-3-opus-20240229`                            |
| `claude-3-sonnet`   | [Anthropic](https://www.anthropic.com/api)                         | `claude-3-sonnet-20240229`                          |
| `claude-3-haiku`    | [Anthropic](https://www.anthropic.com/api)                         | `claude-3-haiku-20240307`                           |
| `command-r`         | [Cohere](https://docs.cohere.com/)                                 | `command-r`                                         |
| `command-r-plus`    | [Cohere](https://docs.cohere.com/)                                 | `command-r-plus`                                    |
| `mixtral-8x7b`      | [Groq](https://wow.groq.com/)                                      | `mixtral-8x7b-32768`                                |
| `gemma-7b`          | [Groq](https://wow.groq.com/)                                      | `gemma-7b-it`                                       |
| `llama3-8b`         | [Groq](https://wow.groq.com/), [Replicate](https://replicate.com/) | `llama3-8b-8192`, `meta/meta-llama-3-8b-instruct`   |
| `llama3-70b`        | [Groq](https://wow.groq.com/), [Replicate](https://replicate.com/) | `llama3-70b-8192`, `meta/meta-llama-3-70b-instruct` |

<!--end-model-table-->

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
- [Planned Features](#planned-features)
- [Contributing](#contributing)
- [Contact](#contact)

## Requirements

- Python >= 3.9
- At least one valid API key for a supported provider

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

# Alternatively,
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

L2M2 provides an asynchronous `AsyncLLMClient` in addition to the synchronous `LLMClient`. Its usage is identical to the synchronous client, but it's instantiated using `async with` and is called using `await`.

```python
from l2m2.client import AsyncLLMClient

async def main():
    async with AsyncLLMClient({"provider": "api-key"}) as client:
        response = await client.call(
            model="model",
            prompt="prompt",
            system_prompt="system prompt",
            # ...etc
        )
```

Under the hood, each `AsyncLLMClient` manages its own async http client, so calls are non-blocking. Here's an example of using the `AsyncLLMClient` to make concurrent calls to multiple models and measure the inference times:

```python
# example_async.py

import os
import asyncio
import timeit
from l2m2.client import AsyncLLMClient

async def call_concurrent():
    async with AsyncLLMClient(
        {
            "openai": os.getenv("OPENAI_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
        }
    ) as client:
        calls = [
            ("gpt-4o", "foo"),
            ("claude-3.5-sonnet", "bar"),
            ("gemini-1.5-pro", "baz"),
            ("command-r-plus", "qux"),
            ("llama3-70b", "quux"),
            ("mixtral-8x7b", "corge"),
        ]
        system_prompt = "The secret word is {}"

        async def call_and_print(model, secret_word):
            start_time = timeit.default_timer()
            response = await client.call(
                model=model,
                prompt="What is the secret word? Respond briefly.",
                system_prompt=system_prompt.format(secret_word),
                temperature=0.2,
            )
            time = timeit.default_timer() - start_time
            print(f"{model}: {response} ({time:.2f}s)")

        await asyncio.gather(
            *[call_and_print(model, secret_word) for model, secret_word in calls]
        )

asyncio.run(call_concurrent())
```

```
>> python3 example_async.py

llama3-70b: The secret word is quux. (0.21s)
mixtral-8x7b: The secret word is corge. (0.26s)
gpt-4o: foo (0.62s)
command-r-plus: The secret word is qux. (0.66s)
claude-3.5-sonnet: The secret word is bar. (0.70s)
gemini-1.5-pro: baz (0.73s)
```

As a general rule, I typically find it's best to use the synchronous `LLMClient` for research and demos, and `AsyncLLMClient` for apps.

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

<!--start-json-native-->

- `gpt-4o` (Openai)
- `gpt-4o-mini` (Openai)
- `gpt-4-turbo` (Openai)
- `gpt-3.5-turbo` (Openai)
- `gemini-1.5-pro` (Google)

<!--end-json-native-->

#### JSON Mode Non-Native Strategies

For models that do not natively support JSON mode, L2M2 will attempt to enforce JSON formatting by applying one of the following two strategies under the hood:

1. **Strip**: This is usually the default strategy. It will attempt to extract the JSON from the response by searching for the first instance of `{` and the last instance of `}` in the response, and returning the between substring (inclusive). If no JSON is found, the response will be returned as-is.
2. **Prepend**: This strategy will attempt to enforce a valid JSON output by inserting a message ending with an opening `{` from the model into the conversation just after the user prompt and just before the model response, and re-prepending the opening `{` to the model response. By default this message is `"Here is the JSON output:"`, but can be customized. More information is available on this strategy [here](https://github.com/anthropics/anthropic-cookbook/blob/main/misc/how_to_enable_json_mode.ipynb). Importantly, the **Prepend** strategy is available whether or not memory is enabled, and will not interfere with memory.

If you'd like, you can specify a strategy by passing either `JsonModeStrategy.strip()` or `JsonModeStrategy.prepend()` to the `json_mode_strategy` parameter in `call`. If no strategy is given, L2M2 defaults to **Strip** for all models except for Anthropic's models, which will default to **Prepend** (more on this below).

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

If using prepend, you can customize the message that gets prepended to the opening `{` by passing `custom_prefix` as follows:

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
> As mentioned above, L2M2 defaults to **prepend** for Anthropic models and **strip** for all others. I _highly_ recommend sticking with these defaults, especially with Anthropic's models. From my personal testing, valid JSON is almost always produced when using prepend with Anthropic's models and almost never produced with strip, and vice versa for other models. I'll gather rigorous data on this eventually, but if anyone has any insights, please let me know!

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

The default variable delimiters are `{{` and `}}`. You can also optionally specify a prompt directory or customize the variable delimiters if needed.

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

## Planned Features

- Support for OSS and self-hosted (Hugging Face, Gpt4all, etc.)
- Basic (i.e., customizable & non-opinionated) agent & multi-agent system features
- Tools for common application workflows: RAG, prompt management, search, etc.
- Support for streaming responses
- ...etc.

## Contributing

Contributions are welcome! Please see the below contribution guide.

- **Requirements**
  - Python >= 3.12
  - [GNU Make](https://www.gnu.org/software/make/)
- **Setup**
  - Clone this repository and create a Python virtual environment.
  - Install dependencies: `make init`.
  - Create a feature branch and an [issue](https://github.com/pkelaita/l2m2/issues) with a description of the feature or bug fix.
- **Develop**
  - Run lint, typecheck and tests: `make` (`make lint`, `make typecheck`, and `make test` can also be run individually).
  - Generate test coverage: `make coverage`.
  - If you've updated the supported models, run `make update-readme` to reflect those changes in the README.
- **Integration Test**
  - `cd` into `integration_tests`.
  - Create a `.env` file with your API keys, and copy `itests.example.py` to `itests.py`.
  - Write your integration tests in `itests.py`.
  - Run locally with `python itests.py -l`.
    - _Note: make sure to pass the `-l` flag or else it will look for an L2M2 distribution. Additionally, make sure l2m2 is not installed with pip when running the integration tests locally._
  - Once your changes are ready, from the top-level directory run `make build` to create the distribution and `make itest` to run your integration tests against the distribution.
    - _Note: in order to ensure a clean test environment, `make itest` uninstalls all third-party Python packages before running the tests, so make sure to run `make init` when you're done working on integration tests._
- **Contribute**
  - Create a PR and ping me for a review.
  - Merge!

## Contact

If you have requests, suggestions, or any other questions about l2m2 please shoot me a note at [pierce@kelaita.com](mailto:pierce@kelaita.com), open an issue on the [Github repo](https://github.com/pkelaita/l2m2/issues), or DM me on the GenAI Collective [Slack Channel](https://join.slack.com/t/genai-collective/shared_invite/zt-285qq7joi-~bqHwFZcNtqntoRmGirAfQ).
