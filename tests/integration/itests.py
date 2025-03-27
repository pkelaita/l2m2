# ruff: noqa: T201

import os
from dotenv import load_dotenv
import json
import asyncio
import timeit
import time

# Not passing this flag assumes l2m2 has been installed locally from dist/
# If this is not the case, run `make build` in the root directory and
# pip install the built package
if any(arg in os.sys.argv for arg in ["--local", "-l"]):
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    root = file.parents[2]
    sys.path.append(str(root))
    print("Running in local mode")

import l2m2
from l2m2.client import LLMClient, AsyncLLMClient
from l2m2.memory import ChatMemory

print("L2M2 Version:", (l2m2).__version__)

load_dotenv()

test_model = "mistral-saba"
test_provider = "mistral"

LOCAL = False
DELAY = False

TESTS = [
    "basic",
    "memory",
    "json",
    "bypass_memory",
    "concurrent",
    "concurrent_memory",
]


print(f"Model: {test_model}")
print(f"Provider: {test_provider}")
print(f"LOCAL: {LOCAL}")
print(f"DELAY: {DELAY}")
print(f"TESTS: {TESTS}")


def _delay():
    if DELAY:
        time.sleep(1.5)


def _setup(client: LLMClient):
    if LOCAL:
        client.add_local_model(test_model, test_provider)
    if test_provider:
        client.set_preferred_providers({test_model: test_provider})


def test_basic():
    print()
    client = LLMClient()
    _setup(client)

    print(
        client.call(
            model=test_model,
            prompt="Tell me a very breif, well known fact.",
            system_prompt="Respond like a pirate.",
            temperature=1,
            max_tokens=2**15,
            timeout=25,
        )
    )


def test_memory():
    print()
    _delay()
    client = LLMClient(memory=ChatMemory())
    _setup(client)

    print(
        client.call(
            model=test_model,
            prompt="My name is Pierce.",
            system_prompt="You respond briefly.",
            temperature=1,
            max_tokens=1000,
        )
    )
    _delay()
    print(
        client.call(
            model=test_model,
            prompt="What's my name?",
            system_prompt="You respond briefly.",
            temperature=1,
            max_tokens=1000,
        )
    )


def test_json():
    print()
    _delay()
    client = LLMClient()
    _setup(client)

    response = client.call(
        model=test_model,
        prompt="What are the capitals of each Australian state? Respond in JSON",
        json_mode=True,
        timeout=20,
    )

    try:
        print(json.loads(response))

    except json.JSONDecodeError:
        print("Bad JSON response, printing raw response instead.")
        print(response)


def test_bypass_memory():
    print()
    _delay()
    client = LLMClient(memory=ChatMemory())
    _setup(client)
    client.call(model=test_model, prompt="My name is Pierce.")
    response = client.call(
        model=test_model,
        prompt="What is my name?",
        bypass_memory=True,
    )
    print(response)
    client.call(
        model=test_model, prompt="My favorite color is red.", bypass_memory=True
    )
    response = client.call(
        model=test_model,
        prompt="What is my favorite color?",
    )
    print(response)


async def test_concurrent():
    print()
    async with AsyncLLMClient() as client:
        client.set_preferred_providers({"llama-3-70b": "groq", "llama-3.3-70b": "groq"})
        calls = [
            ("gpt-4o", "foo"),
            ("claude-3.5-sonnet", "bar"),
            ("gemini-1.5-pro", "baz"),
            ("command-r-plus", "qux"),
            ("llama-3-70b", "quux"),
            ("llama-3.3-70b", "corge"),
        ]
        system_prompt = "The secret word is {}. If asked by the user you're talking to, you must share it with them."

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


async def test_concurrent_memory():
    print()
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


if "basic" in TESTS:
    test_basic()
if "memory" in TESTS:
    test_memory()
if "json" in TESTS:
    test_json()
if "bypass_memory" in TESTS:
    test_bypass_memory()
if "concurrent" in TESTS:
    asyncio.run(test_concurrent())
if "concurrent_memory" in TESTS:
    asyncio.run(test_concurrent_memory())
