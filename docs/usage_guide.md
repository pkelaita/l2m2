# L2M2 Usage Guide

## Table of Contents

- [Getting Started](#getting-started)
- [Multi-Provider Models](#multi-provider-models)
- [Memory](#memory)
- [Asynchronous Usage](#asynchronous-usage)
- [Local Models](#local-models)
- Tools
  - [JSON Mode](#tools-json-mode)
  - [Prompt Loader](#tools-prompt-loader)
- [Other Capabilities](#other-capabilities-extra-parameters)

## Getting Started

**Import the LLM Client**

```python
from l2m2.client import LLMClient

client = LLMClient()
```

**Activate Providers**

To activate any of the providers, set the provider's API key in the corresponding environment variable shown below, and L2M2 will read it in to activate the provider.

| Provider                | Environment Variable  |
| ----------------------- | --------------------- |
| OpenAI                  | `OPENAI_API_KEY`      |
| Anthropic               | `ANTHROPIC_API_KEY`   |
| Cohere                  | `CO_API_KEY`          |
| Google                  | `GOOGLE_API_KEY`      |
| Groq                    | `GROQ_API_KEY`        |
| Replicate               | `REPLICATE_API_TOKEN` |
| Mistral (La Plateforme) | `MISTRAL_API_KEY`     |
| Cerebras                | `CEREBRAS_API_KEY`    |

Additionally, you can activate providers programmatically as follows:

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

#### Example

```python
# example.py

from l2m2.client import LLMClient

client = LLMClient()

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

## Multi-Provider Models

Some models are available from multiple providers, such as `llama-3-70b` from both Groq and Replicate. When multiple of such providers are active, you can use the parameter `prefer_provider` to specify which provider to use for a given inference.

```python
response1 = client.call(
    model="llama-3-70b",
    prompt="Hello there",
    prefer_provider="groq",
) # Uses Groq

response2 = client.call(
    model="llama-3-70b",
    prompt="General Kenobi!",
    prefer_provider="replicate",
) # Uses Replicate
```

You can also set default preferred providers for the client using `set_preferred_providers`, to avoid having to specify `prefer_provider` for each call.

```python
client.set_preferred_providers({
    "llama-3-70b": "groq",
    "llama-3-8b": "replicate",
})

response1 = client.call(model="llama-3-70b", prompt="Hello there") # Uses Groq
response2 = client.call(model="llama-3-8b", prompt="General Kenobi!") # Uses Replicate
```

## Memory

L2M2 provides a simple memory system that allows you to maintain context and history across multiple calls and multiple models. There are two types of memory: **`ChatMemory`**, which natively hooks into models' conversation history, and **`ExternalMemory`**, which allows for custom memory implementations. Let's first take a look at `ChatMemory`.

```python
from l2m2.client import LLMClient
from l2m2.memory import ChatMemory

client = LLMClient(memory=ChatMemory())

print(client.call(model="gpt-4o", prompt="My name is Pierce"))
print(client.call(model="claude-3-haiku", prompt="I am a software engineer."))
print(client.call(model="llama-3-8b", prompt="What's my name?"))
print(client.call(model="llama-3.3-70b", prompt="What's my job?"))
```

```
Hello, Pierce! How can I help you today?
A software engineer, you say? That's a noble profession.
Your name is Pierce.
You are a software engineer.
```

Chat memory is stored per session, with a sliding window of messages which defaults to the last 40 – this can be configured by passing `memory_window_size` to the client constructor.

For more control, you can instantiate a `ChatMemory` object on its own and manipulate it directly.

```python
memory = ChatMemory()

memory.add_user_message("My favorite color is red.")
memory.add_user_message("My least favorite color is green.")
memory.add_agent_message("Ok, noted.")

client = LLMClient(memory=memory)
print(client.call(model="gpt-4o", prompt="What are my favorite and least favorite colors?"))
memory.clear()
print(client.call(model="gpt-4o", prompt="What are my favorite and least favorite colors?"))
```

```
Your favorite color is red, and your least favorite color is green.
I'm sorry, I don't have that information.
```

> [!CAUTION]
> Some providers such as Anthropic enforce that chat messages in memory strictly alternate between one user and one agent message and will throw an error if this is not the case.

You can also load in alternate memory streams on the fly using the `alt_memory` parameter in `call` (This is especially useful for parallel memory streams – an example of this is shown in the [Async Calls](#async-calls) section).

```python
m1 = ChatMemory()
m1.add_user_message("My favorite color is red.")
m1.add_user_message("My least favorite color is green.")
m1.add_agent_message("Ok, noted.")

m2 = ChatMemory()
m2.add_user_message("My favorite color is blue.")
m2.add_user_message("My least favorite color is yellow.")
m2.add_agent_message("Got it.")

client = LLMClient(memory=m1)
prompt = "What are my favorite and least favorite colors?"
print(client.call(model="gpt-4o", prompt=prompt)
print(client.call(model="gpt-4o", prompt=prompt, alt_memory=m2))
```

```
Your favorite color is red, and your least favorite color is green.
Your favorite color is blue, and your least favorite color is yellow.
```

Finally, memory can be bypassed for a single call by passing `bypass_memory=True` to `call`. This will cause the client to ignore previously stored memory and not write to it for the current call.

```python
client = LLMClient(memory=ChatMemory())
client.call(model="gpt-4o", prompt="My name is Pierce")
client.call(model="gpt-4o", prompt="I am 25 years old")

print(client.call(model="gpt-4o", prompt="What is my name?"))
print(client.call(model="gpt-4o", prompt="What is my name?", bypass_memory=True))

client.call(model="gpt-4o", prompt="I am a software engineer", bypass_memory=True)
print(client.call(model="gpt-4o", prompt="What is my profession?"))
```

```
Your name is Pierce.
I'm sorry, but I don't have access to personal information, so I can't know your name.
You haven't mentioned your profession yet, Pierce.
```

### External Memory

**`ExternalMemory`** is a simple but powerful memory mode that allows you to define your own memory implementation. This can be useful for more complex memory constructions (e.g., planning, reflecting) or for implementing custom persistence (e.g., saving memory to a database or a file). Its usage is much like `ChatMemory`, but unlike `ChatMemory` you must manage initializing and updating the memory yourself with `get_contents` and `set_contents`.

Here's a simple example of a custom memory implementation that has a description and a list of previous user/model message pairs:

```python
# example_external_memory.py

from l2m2.client import LLMClient
from l2m2.memory import ExternalMemory

client = LLMClient(memory=ExternalMemory())

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

memory = ExternalMemory(loading_type=ExternalMemoryLoadingType.USER_PROMPT_APPEND)
client = LLMClient(memory=memory)
```

Similarly to `ChatMemory`, `ExternalMemory` can be passed into `alt_memory` and bypassed with `bypass_memory`.

## Asynchronous Usage

L2M2 provides an asynchronous `AsyncLLMClient` in addition to the synchronous `LLMClient`. Its usage is identical to the synchronous client, but it's instantiated using `async with` and is called using `await`.

```python
from l2m2.client import AsyncLLMClient

async def main():
    async with AsyncLLMClient() as client:
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
    async with AsyncLLMClient() as client:
        # Assumes no conflicts between active providers
        calls = [
            ("gpt-4o", "foo"),
            ("claude-3.5-sonnet", "bar"),
            ("gemini-1.5-pro", "baz"),
            ("command-r-plus", "qux"),
            ("llama-3-70b", "quux"),
            ("llama-3.3-70b", "corge"),
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

llama-3-70b: The secret word is quux. (0.21s)
llama-3.3-70b: The secret word is corge. (0.26s)
gpt-4o: foo (0.62s)
command-r-plus: The secret word is qux. (0.66s)
claude-3.5-sonnet: The secret word is bar. (0.70s)
gemini-1.5-pro: baz (0.73s)
```

As a general rule, I typically find it's best to use the synchronous `LLMClient` for research and demos, and `AsyncLLMClient` for apps.

### Use Case: Parallel Memory Streams ⚡

One of the most powerful features of `AsyncLLMClient` is the ability to run maintain memory streams in parallel, such as in multi-agent systems with multiple interactions happening concurrently. Here's a simple example of how to easily achieve this using `AsyncLLMClient` and `alt_memory`.

```python
# example_parallel_memory.py

import asyncio
from l2m2.client import AsyncLLMClient
from l2m2.memory import ChatMemory

async def call_concurrent_with_memory():
    m1 = ChatMemory()
    m2 = ChatMemory()

    calls1 = ["My name is Pierce", "My favorite color is red", "I am 25 years old"]
    calls2 = ["My name is Paul", "My favorite color is blue", "I am 60 years old"]
    question = "What is my name, favorite color, and age?"

    async with AsyncLLMClient() as client:
        client.set_preferred_providers({"llama-3.3-70b": "groq"})

        async def make_calls_1():
            for prompt in calls1:
                await client.call(model="llama-3.3-70b", prompt=prompt, alt_memory=m1)

        async def make_calls_2():
            for prompt in calls2:
                await client.call(model="llama-3.3-70b", prompt=prompt, alt_memory=m2)

        await asyncio.gather(make_calls_1(), make_calls_2())

        [res1, res2] = await asyncio.gather(
            client.call(model="llama-3.3-70b", prompt=question, alt_memory=m1),
            client.call(model="llama-3.3-70b", prompt=question, alt_memory=m2),
        )

        print("Memory 1:", res1)
        print("Memory 2:", res2)

asyncio.run(call_concurrent_with_memory())
```

```
>> python3 example_parallel_memory.py

Memory 1: Your name is Pierce, your favorite color is red, and you are 25 years old. I hope this information is helpful!
Memory 2: Your name is Paul, your favorite color is blue, and you are 60 years old. 😊
```

## Local Models

L2M2 supports local models via [Ollama](https://ollama.ai/), with the exact same usage and features as API-based models (memory, async, JSON mode, the works) and almost the exact same setup, with a few minor additions which we'll go through here.

This guide assumed you have a working Ollama installation with at least one working model – see Ollama's [docs](https://github.com/ollama/ollama#readme) if this is not the case.

To use local models, add them to your client as follows:

```python
client = LLMClient() # or AsyncLLMClient
client.add_local_model("phi4", "ollama") # Currently ollama is the only available local provider
```

And run them as you would any other model:

```python
response = client.call(
    model="phi4",
    prompt="What's the capital of France?",
    # ... system_prompt, temperature, etc.
)
```

> [!IMPORTANT]
> When you add and call a local model, make sure it's already running in your Ollama server. L2M2 does not run any Ollama commands like `pull`, `run`, etc. and assumes any model you're calling is available.

If a model you want to run locally is already available from another provider, you can just use [preferred providers](#multi-provider-models) to specify Ollama as you would with any other provider.

```python
client.add_local_model("mistral-small", "ollama") # Also available from Mistral's API

client.call(
    model="mistral-small",
    prompt="Hello world",
    prefer_provider="ollama",
)
# Or equivalently,
client.set_preferred_providers({"mistral-small": "ollama"})
client.call(
    model="mistral-small",
    prompt="Hello world",
)
```

### Specifying a Custom Local LLM Server

By default, L2M2 will use the Ollama server running on `http://localhost:11434`. You can override this with a custom URL if you're running Ollama on a remote server, want to specify a different port, or otherwise.

```python
client = LLMClient()
client.override_local_base_url("ollama", "https://my-ollama-webservice:1234")
# add models and call as usual
```

For remote services, you might have some sort of authentication in the request headers. You can handle this by passing in the `extra_headers` parameter to `call`.

```python
response = client.call(
    model="phi4",
    prompt="What's the capital of France?",
    extra_headers={"X-my-auth-header": "my-auth-value"},
)
```

If for some reason you have authentication in the request body, you can pass it in the `extra_params` parameter instead.

```python
response = client.call(
    model="phi4",
    prompt="What's the capital of France?",
    extra_params={"username": "AzureDiamond", "password": "hunter2"},
)
```

Since authentication is not the intended purpose of `extra_params`, the values can only be strings, ints, or floats. However, if you really need to do auth in the body let me know and I'll add some official support for this in an update.

Beyond this, everything else with local models works exactly as it does with API-based models.

## Tools: JSON Mode

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

The following models natively support JSON mode via the given provider:

<!--start-json-native-->

- `o3-mini` (via Openai)
- `o1-pro` (via Openai)
- `o1` (via Openai)
- `gpt-4.5` (via Openai)
- `gpt-4o` (via Openai)
- `gpt-4o-mini` (via Openai)
- `gpt-4-turbo` (via Openai)
- `gpt-3.5-turbo` (via Openai)
- `gemini-2.5-pro` (via Google)
- `gemini-2.0-pro` (via Google)
- `gemini-2.0-flash` (via Google)
- `gemini-2.0-flash-lite` (via Google)
- `gemini-1.5-flash` (via Google)
- `gemini-1.5-flash-8b` (via Google)
- `gemini-1.5-pro` (via Google)
- `mistral-large` (via Mistral)
- `mistral-small` (via Mistral)
- `ministral-3b` (via Mistral)
- `ministral-8b` (via Mistral)
- `qwen-qwq-32b` (via Groq)
- `mistral-saba` (via Groq)
- `mistral-saba` (via Mistral)
- `gemma-2-9b` (via Groq)
- `llama-4-maverick` (via Groq)
- `llama-4-scout` (via Groq)
- `llama-3.3-70b` (via Groq)
- `llama-3.2-3b` (via Groq)
- `llama-3.2-1b` (via Groq)
- `llama-3.1-8b` (via Groq)
- `llama-3-70b` (via Groq)
- `llama-3-8b` (via Groq)
- `qwen-2.5-32b` (via Groq)
- `deepseek-r1-distill-qwen-32b` (via Groq)
- `deepseek-r1-distill-llama-70b` (via Groq)

<!--end-json-native-->

- Any local model via Ollama

### JSON Mode Non-Native Strategies

For models that do not natively support JSON mode, L2M2 will attempt to enforce JSON formatting by applying one of the following two strategies under the hood:

1. **Strip**: This is usually the default strategy. It will attempt to extract the JSON from the response by searching for the first instance of `{` and the last instance of `}` in the response, and returning the between substring (inclusive). If no JSON is found, the response will be returned as-is.
2. **Prepend**: This strategy will attempt to enforce a valid JSON output by inserting a message ending with an opening `{` from the model into the conversation just after the user prompt and just before the model response, and re-prepending the opening `{` to the model response. By default this message is `"Here is the JSON output:"`, but can be customized. More information is available on this strategy [here](https://github.com/anthropics/anthropic-cookbook/blob/main/misc/how_to_enable_json_mode.ipynb). Importantly, the **Prepend** strategy is available whether or not memory is enabled, and will not interfere with memory.

If you'd like, you can specify a strategy by passing either `JsonModeStrategy.strip()` or `JsonModeStrategy.prepend()` to the `json_mode_strategy` parameter in `call`. If no strategy is given, L2M2 defaults to **Strip** for all models except for Anthropic's models, which will default to **Prepend** (more on this below).

```python
# example_json_mode.py

from l2m2.client import LLMClient
from l2m2.tools import JsonModeStrategy

client = LLMClient()

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

## Tools: Prompt Loader

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

## Other Capabilities: Extra Parameters

You can pass in extra parameters to the provider's API (For example, [reasoning_effort](https://platform.openai.com/docs/api-reference/chat/create#chat-create-reasoning_effort) on OpenAI's o1 series, or [thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) on Anthropic's claude 3.7+) by passing in the `extra_params` parameter to `call`. These parameters are passed in as a dictionary of key-value pairs, where the values are of type `str`, `int`, or `float`. Using `extra_params` does not guarantee correctness or well-defined behavior, and you should refer to the provider's documentation for correct usage.

```python
response = client.call(
    model="<model name>",
    prompt="<prompt>",
    extra_params={"foo": "bar", "baz": 123},
    ...
)
```

Example usage with Claude 3.7 Sonnet:

```python
response = client.call(
    model="claude-3.7-sonnet",
    prompt=f"Reverse engineer a business plan for this company: {company_description}",
    max_tokens=20000,
    extra_params={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 16000,
        },
    },
)
```

Additionally, you can pass in extra headers to access the beta [128k extended output](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) for this:

```python
response = client.call(
    model="claude-3.7-sonnet",
    prompt=f"Reverse engineer a business plan for this company: {company_description}",
    extra_headers={"anthropic-beta": "output-128k-2025-02-19"},
    max_tokens=128000,
    extra_params={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 32000,
        },
    },
)
```
