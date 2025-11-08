# Changelog

_Current version: 0.0.61_

[PyPi link](https://pypi.org/project/l2m2/)

### 0.0.61 - November 8, 2025

#### Added

- Support for provider [Moonshot AI](https://www.moonshot.ai/).
- Support for the following models via Moonshot AI:
  - `kimi-k2`
  - `kimi-k2-turbo`
  - `kimi-k2-thinking`
  - `kimi-k2-thinking-turbo`

### 0.0.60 - October 20, 2025

> [!CAUTION]
> This release drops support for Python 3.9! If you are using Python 3.9, please upgrade to Python 3.10 or higher to use l2m2 v0.0.60 or higher.

#### Removed

- Dropped support for Python 3.9 as it is [no longer officially supported](https://devguide.python.org/versions/).

### 0.0.59 - October 17, 2025

#### Added

- Support for [Claude Haiku 4.5](https://www.anthropic.com/news/claude-haiku-4-5) via Anthropic.

#### Changed

- Updated the async client to use aiohttp instead of httpx, which is slightly faster and has a smaller dependecy footprint.

#### Fixed

- In v0.0.57 and v0.0.58, `claude-sonnet-4.5` from l2m2 mistakenly pointed to Anthropic's model ID for sonnet 4. This has been fixed to point to the correct sonnet 4.5 model ID.

### 0.0.58 - October 7, 2025

#### Added

- Compatibility with [Python 3.14](https://www.python.org/downloads/release/python-3140/).

### 0.0.57 - September 29, 2025

> [!CAUTION]
> This release has breaking changes! Please read the changelog carefully.

Sorry it took me so long to get to this! Work has been pretty busy and I've mostly been using typescript
so haven't had an excuse to work on this in a while. It's been 3 months... so LOTS of updates here.

#### Added

- Support for [GPT-5, GPT-5-mini, and GPT-5-nano](https://openai.com/index/introducing-gpt-5/) via OpenAI.
- Support for [Claude Opus 4.1](https://www.anthropic.com/news/claude-opus-4-1) and [Claude Sonnet 4.5](https://www.anthropic.com/news/claude-sonnet-4-5) via Anthropic.
- Support for [Commmand-A Reasoning](https://cohere.com/blog/command-a-reasoning) and [Commmand-A Translate](https://cohere.com/blog/command-a-translate) via Cohere.
- Support for [Devstral Medium](https://mistral.ai/news/devstral-2507) via Mistral.
- Support for [GPT-OSS](https://openai.com/index/introducing-gpt-oss/), with the 120b version available via Groq and Cerebras and the 20b version available via Groq.
- Support for [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) via Groq.
- Support for the following additional models via Cerebras:
  - [Llama 4 Maverick](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) (also supported via Groq)
  - [Qwen 3 Coder](https://github.com/QwenLM/Qwen3-Coder) in versions 480b, 235b, and 235b-thinking.
- Updated the following model tags on Mistral to their latest versions:
  - Magistral Small: `magistral-small-2506` → `magistral-small-2509`
  - Mistral Medium: `mistral-medium-2505` → `mistral-medium-2508`
  - Mistral Small: `mistral-small-2503` → `mistral-small-2506`
  - Codestral: `codestral-2501` → `codestral-2508`

#### Removed

- Support for the following Gemini models which have been deprecated:
  - Gemini 2.0 Pro
  - Gemini 1.5 Pro
  - Gemini 1.5 Flash
  - Gemini 1.5 Flash 8B

#### Fixed

- Switched from preview tags to production tags for all Gemini models, as recommended by Google AI Studio's [documentation](https://ai.google.dev/gemini-api/docs/models#preview).
- Fixed a bug where gemini models would throw an error when parsing some non-standard responses (such as those when exceeding the max token limit).

### 0.0.56 - June 19, 2025

#### Added

- Support for OpenAI's [o3-pro](https://platform.openai.com/docs/models/o3-pro).
- Support for [Magistral-small and medium](https://mistral.ai/news/magistral), [Devstral-small](https://mistral.ai/news/devstral), and [Codestral-mamba](https://mistral.ai/news/codestral-mamba) via Mistral.
- Support for [Qwen 3 32B](https://console.groq.com/docs/model/qwen/qwen3-32b) via Groq (was already available via Cerebras).

#### Fixed

- Added correct max token limits for `ministral-3b` and `ministral-8b` (both 2<sup>16</sup> - 1).

### 0.0.55 - June 5, 2025

#### Added

- Support for [Qwen 3.3 32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) via Cerebras.
- Support for [native JSON mode](https://inference-docs.cerebras.ai/capabilities/structured-outputs) on all models via Cerebras.

#### Changed

- Updated Gemini 2.5 Pro from `gemini-2.5-pro-preview-05-06` to `gemini-2.5-pro-preview-06-05`.

#### Fixed

- Changed mistral-saba's model ID for groq from `mistral-saba-2502` (which no longer works) to `mistral-saba-24b`.

### 0.0.54 - May 22, 2025

#### Added

- Support for Claude Opus 4 and Claude Sonnet 4 [released today](https://www.anthropic.com/news/claude-4).

#### Fixed

- Updated the maximum `max_tokens` value for `claude-3.7-sonnet` from 128000 to 64000 to match its actual limit.
- Switched the default temperature for previous Claude models (`claude-3.7-sonnet`, `claude-3.5-sonnet`, `claude-3.5-haiku`, `claude-3-opus`, `claude-3-sonnet`, and `claude-3-haiku`) from L2M2-imposed default of 0.0 to the provider fallback value from Anthropic's platform.

### 0.0.53 - May 14, 2025

#### Added

- Updated Gemini 2.5 Pro from `gemini-2.5-pro-exp-03-25` (which is now deprecated) to `gemini-2.5-pro-preview-05-06`.

### 0.0.52 - May 11, 2025

#### Added

- Support for [Gemini 2.5 Flash](https://developers.googleblog.com/en/start-building-with-gemini-25-flash/).

### 0.0.51 - April 28, 2025

#### Added

- Support for [o4-mini](https://platform.openai.com/docs/models/o4-mini), [o3](https://platform.openai.com/docs/models/o3), [gpt-4.1](https://platform.openai.com/docs/models/gpt-4.1), [gpt-4.1-mini](https://platform.openai.com/docs/models/gpt-4.1-mini), and [gpt-4.1-nano](https://platform.openai.com/docs/models/gpt-4.1-nano) via OpenAI.
- Support for [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) with Maverick available via Groq and Scout available via both Groq and Cerebras.
- Support for [Allam 2 70b](https://arxiv.org/html/2407.15390v1) via Groq.
- Support for Llama 3.3 70b via Cerebras (was already previously supported via Groq).
- Support for [Codestral](https://mistral.ai/news/codestral) via Mistral.
- Support for native JSON mode on any model via Groq.

#### Changed

- Updated the versions of the following models from Mistral:
  - `mistral-small`: `mistral-small-2501` → `mistral-small-2503`
  - `mistral-saba`: `mistral-saba-24b` → `mistral-saba-2502`

#### Removed

- Support `o1-mini` and `o1-preview` from OpenAI following their [deprecation](https://platform.openai.com/docs/deprecations).
- Support for the following models via Groq following their [deprecations](https://console.groq.com/docs/deprecations):
  - `llama-3.2-3b`
  - `llama-3.2-1b`
  - `qwen-2.5-32b`
  - `deepseek-r1-distill-qwen-32b`

### 0.0.50 - March 29, 2025

#### Added

- Support for OpenAI's [updated `gpt-4o` model](https://help.openai.com/en/articles/6825453-chatgpt-release-notes).

### 0.0.49 - March 27, 2025

#### Added

- Support for Google's [Gemini 2.5 Pro](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) model released yesterday.
- Support for [`mistral-saba`](https://mistral.ai/news/mistral-saba) via both Mistral Cloud and Groq.

#### Removed

- `mixtral-8x22b` has been removed as it was deprecated by Groq in March of 2025.

#### Changed

- Updated various models' max `temperature` values via Groq to be consistent with Groq's actual limits.
  - `mistral-large`: 1.0 → 1.5
  - `mistral-small`: 1.0 → 1.5
  - `ministral-3b`: 1.0 → 1.5
  - `ministral-8b`: 1.0 → 1.5
  - `gemma-2-9b`: 1.0 → 1.5
- Updated various models' max `max_tokens` values to be consistent with actual provider limits.
  - `gemini-2.0-pro` (via Google) : 8192 → 2<sup>31</sup>-1
  - `gemma-2-9b` (via Groq) : 2<sup>16</sup>-1 → 2<sup>13</sup>-1

### 0.0.48 - March 21, 2025

#### Added

- Support for OpenAI's [o1-pro](https://platform.openai.com/docs/models/o1-pro) model released this week.

#### Changed

- Migrated OpenAI calls from the legacy Chat Completion API to the new [Responses API](https://community.openai.com/t/introducing-the-responses-api/1140929).

### 0.0.47 - March 13, 2025

#### Added

- Support for [Command-A](https://cohere.com/blog/command-a), Cohere's latest model released today.

### 0.0.46 - March 11, 2025

#### Added

- Support for [GPT-4.5](https://openai.com/index/introducing-gpt-4-5/) released on February 27, 2025.

#### Fixed

- `o1` and `o3-mini` now correctly support native JSON mode.

### 0.0.45 - February 26, 2025

#### Fixed

- Patched an error where calls to Anthropic's `claude-3.7-sonnet` with [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) would fail.
- Updated the max tokens for `claude-3.7-sonnet` to 128000 and `claude-3.5-sonnet` and `claude-3.5-haiku` to 8192 as per the [Anthropic docs](https://docs.anthropic.com/en/docs/about-claude/models/all-models).

#### Changed

- Updated the default max tokens for the claude 3.5 and 3.7 models to 4096 and the claude 3 models to 2048 to be more reasonable given the max allowed values.

### 0.0.44 - February 24, 2025

#### Added

- Support for Anthropic's [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) released today.

### 0.0.43 - February 21, 2025

#### Fixed

- `o3-mini` now correctly uses the `developer` system message key rather than `system` as per the OpenAI API spec.

### 0.0.42 - February 15, 2025

#### Added

- Suppert for 7 new models:
  - `o3-mini` via OpenAI
  - `gemini-2.0-pro` and `gemini-2.0-flash-lite` via Google
  - `qwen-2.5-32b`, `deepseek-r1-distill-qwen-32b`, and `deepseek-r1-distill-llama-70b` via Groq
  - `command-r7b` via Cohere

#### Changed

- Updated the Cohere API from V1 to V2 ([their docs](https://cohere.com/blog/new-api-v2)).
- Replaced all instances of `ValueError` being raised with a new `L2M2UsageError` exception.
- Moved the `warnings` module to the top level (`l2m2.warnings` instead of `l2m2.client.warnings`).
- Increased the default timeout for LLM calls from 10 seconds to 25 seconds.
- Where possible, pinned l2m2 models to specific versions rather than an alias pointing to the latest version. This is for stability in production; however, I do plan to keep the versions up to date on a regular basis. The following model versions have been updated:
  - `o1` → `o1-2024-12-17`
  - `o1-preview` → `o1-preview-2024-09-12`
  - `o1-mini` → `o1-mini-2024-09-12`
  - `claude-3-5-sonnet-latest` → `claude-3-5-sonnet-20241022`
  - `claude-3-5-haiku-latest` → `claude-3-5-haiku-20241022`
  - `command-r` → `command-r-08-2024`
  - `command-r-plus` → `command-r-plus-08-2024`
  - `mistral-large-latest` → `mistral-large-2411`
  - `ministral-3b-latest` → `ministral-3b-2410`
  - `gemini-2.0-flash-exp` → `gemini-2.0-flash-001`
  - `gemini-1.5-flash-exp` → `gemini-1.5-flash-001`
    Note that this is _not_ a breaking change – the model IDs are purely internal. This doesn't change any behavior, just adds stability.

#### Fixed

- Previously, the synchronous `call` method in `LLMClient` would throw a gibberish unhandled exception when used in an async context, such as within FastAPI. This has been fixed - it is now handled by an `L2M2UsageError` with a helpful message recommending the use of `AsyncLLMClient` instead. This error is also thrown when instantiating `LLMClient` in an async context.
- For some reason, OpenAI doesn't support either the `system` or `developer` keyword in `o1-mini` and `o1-preview`, effectively making system prompts unusable with them. While I'm not sure why this is, I've properly handled these with `L2M2UsageError` instead of throwing unhandled exceptions.

### 0.0.41 - February 3, 2025

#### Added

- **Big update!** Added support for running local llms via [Ollama](https://ollama.ai/). 🎉
  See the docs for running local models [here](docs/usage_guide.md#local-models).

#### Changed

- The `providers` parameter to `LLMClient` and `AsyncLLMClient` has been renamed to `api_keys`.
- L2M2 no longer depends on `typing_extensions` and now officially only has a single external dependency: `httpx`.

#### Removed

- The static method `get_available_models` in `LLMClient` and `AsyncLLMClient` has been deprecated as it is no longer meaningful with the addition of local models (which can be arbitrary). It will be removed in a future release.

### 0.0.40 - January 16, 2025

> [!CAUTION]
> This release has breaking changes! Please read the changelog carefully.

#### Removed

- The `call_custom` method has been removed from `LLMClient` and `AsyncLLMClient` due to lack of use and unnecessary complexity. **This is a breaking change!!!** If you need to call a model that is not officially supported by L2M2, please open an issue on the [Github repo](https://github.com/pkelaita/l2m2/issues).

### 0.0.39 - December 17, 2024

> [!CAUTION]
> This release has breaking changes! Please read the changelog carefully.

#### Added

- Support for [Llama 3.3 70b](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3/) via [Groq](https://console.groq.com/docs/models) and [Cerebras](https://inference-docs.cerebras.ai/introduction).
- Support for OpenAI's [o1 series](https://openai.com/o1/): `o1`, `o1-preview`, and `o1-mini`.
- The `extra_params` parameter to `call` and `call_custom`.

> [!NOTE]
> At the time of this release, you must be on OpenAI's [usage tier](https://platform.openai.com/docs/guides/rate-limits) 5 to use `o1` and tier 1+ to use `o1-preview` and `o1-mini`.

#### Removed

- `gemma-7b` has been removed as it has been [deprecated](https://console.groq.com/docs/models) by Groq.
- `llama-3.1-70b` has been removed as it has been deprecated by both [Groq](https://console.groq.com/docs/models) and [Cerebras](https://inference-docs.cerebras.ai/introduction).

### v0.0.38 - December 12, 2024

> [!CAUTION]
> This release has breaking changes! Please read the changelog carefully.

#### Added

- Support for [Python 3.13](https://www.python.org/downloads/release/python-3130/).
- Support for Google's [Gemini 2.0 Flash](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-2.0-flash), [Gemini 1.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash), and [Gemini 1.5 Flash 8B](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash-8b) models.

#### Removed

- Gemini 1.0 Pro is no longer supported, as it is [deprecated](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.0-pro) by Google. **This is a breaking change!!!** Calls to Gemini 1.0 Pro will fail.

### 0.0.37 - December 9, 2024

> [!CAUTION]
> This release has _significant_ breaking changes! Please read the changelog carefully.

#### Added

- Support for Anthropic's [claude-3.5-haiku](https://www.anthropic.com/news/3-5-models-and-computer-use).
- Support for provider [Cerebras](https://cerebras.ai/), offering `llama-3.1-8b` and `llama-3.1-70b`.
- Support for Mistral's `mistral-small`, `ministral-8b`, and `ministral-3b` models via La Plateforme.

#### Changed

- `mistral-large-2` has been renamed to `mistral-large`, to keep up with Mistral's naming scheme. **This is a breaking change!!!** Calls to `mistral-large-2` will fail.

#### Removed

- `mixtral-8x22b`, `mixtral-8x7b`, and `mistral-7b` are no longer available from provider Mistral as they have been [deprecated](https://docs.mistral.ai/getting-started/models/models_overview/). **This is a breaking change!!!** Calls to `mixtral-8x7b` and `mistral-7b` will fail, and calls to `mixtral-8x22b` via provider Mistral will fail.

> [!NOTE]
> The model `mixtral-8x22b` is still available via Groq.

### 0.0.36 - November 21, 2024

#### Changed

- Updated `gpt-4o` version from `gpt-4o-2024-08-06` to `gpt-4o-2024-11-20` ([Announcement](https://twitter.com/OpenAI/status/1859296125947347164))

### 0.0.35 - October 22, 2024

#### Added

- Support for Anthropic's updated [Claude 3.5 Sonnet](https://www.anthropic.com/news/3-5-models-and-computer-use) released today

#### Changed

- `claude-3.5-sonnet` now points to version `claude-3-5-sonnet-latest`

### 0.0.34 - September 30, 2024

> [!CAUTION]
> This release has breaking changes! Please read the changelog carefully.

#### Added

- New supported models `gemma-2-9b`, `llama-3.2-1b`, and `llama-3.2-3b` via Groq.

#### Changed

- In order to be more consistent with l2m2's naming scheme, the following model ids have been updated:
  - `llama3-8b` → `llama-3-8b`
  - `llama3-70b` → `llama-3-70b`
  - `llama3.1-8b` → `llama-3.1-8b`
  - `llama3.1-70b` → `llama-3.1-70b`
  - `llama3.1-405b` → `llama-3.1-405b`
- **This is a breaking change!!!** Calls using the old `model_id`s (`llama3-8b`, etc.) will fail.

#### Removed

- Provider `octoai` has been removed as they have [been acquired](https://www.geekwire.com/2024/chip-giant-nvidia-acquires-octoai-a-seattle-startup-that-helps-companies-run-ai-models/) and are shutting down their cloud platform. **This is a breaking change!!!** Calls using the `octoai` provider will fail.
  - All previous OctoAI supported models (`mixtral-8x22b`, `mixtral-8x7b`, `mistral-7b`, `llama-3-70b`, `llama-3.1-8b`, `llama-3.1-70b`, and `llama-3.1-405b`) are still available via Mistral, Groq, and/or Replicate.

### 0.0.33 - September 11, 2024

#### Changed

- Updated gpt-4o version from `gpt-4o-2024-05-13` to `gpt-4o-2024-08-06`.

### 0.0.32 - August 5, 2024

#### Added

- [Mistral](https://mistral.ai/) provider support via La Plateforme.
- [Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) model availibility from Mistral.
- Mistral 7B, Mixtral 8x7B, and Mixtral 8x22B model availibility from Mistral in addition to existing providers.

- _0.0.30 and 0.0.31 are skipped due to a packaging error and a model key typo._

### 0.0.29 - August 4, 2024

> [!CAUTION]
> This release has breaking changes! Please read the changelog carefully.

#### Added

- `alt_memory` and `bypass_memory` have been added as parameters to `call` and `call_custom` in `LLMClient` and `AsyncLLMClient`. These parameters allow you to specify alternative memory streams to use for the call, or to bypass memory entirely.

#### Changed

- Previously, the `LLMClient` and `AsyncLLMClient` constructors took `memory_type`, `memory_window_size`, and `memory_loading_type` as arguments. Now, it just takes `memory` as an argument, while `window_size` and `loading_type` can be set on the memory object itself. These changes make the memory API far more consistent and easy to use, especially with the additions of `alt_memory` and `bypass_memory`.

#### Removed

- The `MemoryType` enum has been removed. **This is a breaking change!!!** Instances of `client = LLMClient(memory_type=MemoryType.CHAT)` should be replaced with `client = LLMClient(memory=ChatMemory())`, and so on.

### 0.0.28 - August 3, 2024

#### Added

- Providers can now be activated by default via the following environment variables:
  - `OPENAI_API_KEY` for OpenAI
  - `ANTHROPIC_API_KEY` for Anthropic
  - `CO_API_KEY` for Cohere
  - `GOOGLE_API_KEY` for Google
  - `GROQ_API_KEY` for Groq
  - `REPLICATE_API_TOKEN` for Replicate
  - `OCTOAI_TOKEN` for OctoAI

### 0.0.27 - July 24, 2024

#### Added

- [OctoAI](https://octoai.cloud/) provider support.
- [Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/) availibility, in sizes 8B (via OctoAI), 70B (via OctoAI), and 405B (via both OctoAI and Replicate).
- [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/) and [Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b/) via OctoAI.
- `LLMOperationError` exception, raised when a feature or mode is not supported by a particular model.

#### Fixed

- Rate limit errors would sometimes give the model id as `None` in the error message. This has been fixed.

### 0.0.26 - July 19, 2024

#### Added

- [GPT-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) availibility.

### 0.0.25 - July 12, 2024

#### Added

- Custom exception `LLMRateLimitError`, raised when an LLM call returns a 429 status code.

### 0.0.24 - July 11, 2024

#### Added

- The ability to specify a custom timeout for LLM calls by passing a `timeout` argument to `call` or `call_custom` (defaults to 10 seconds).
- A custom exception `LLMTimeoutError` which is raised when an LLM call times out, along with a more helpful message than httpx's default timeout error.

#### Fixed

- Calls to Anthropic with large context windows were sometimes timing out, prompting this change.

### 0.0.23 - June 30, 2024

#### Fixed

- Major bug where l2m2 would cause environments without `typing_extensions` installed to crash due to it not being listed as an external dependency. This has been fixed by adding `typing_extensions` as an external dependency.

#### Changed

- This bug wasn't caught becuase integration tests were not running in a clean environment – (i.e., `typing_extensions` was already installed from one of the dev dependencies). To prevent this from happening again, I made `make itest` uninstall all Python dependencies before running.

### 0.0.22 - June 22, 2024

#### Fixed

- In 0.0.21, async calls were blocking due to the use of `requests`. 0.0.22 replaces `requests` with `httpx` to allow for fully asynchoronous behavior.

#### Changed

- `AsyncLLMClient` should now be instantiated with a context manager (`async with AsyncLLMClient() as client:`) to ensure proper cleanup of the `httpx` client.
- In `AsyncLLMClient`, `call_async` and `call_custom_async` have been renamed to `call` and `call_custom` respectively, with asynchronous behavior.

#### Removed

- `call_concurrent` and `call_custom_concurrent` have been removed due to unnecessary complexity and lack of use.

### 0.0.21 - June 20, 2024

#### Added

- This changelog (finally – oops)
- Support for Anthropic's [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) released today

#### Changed

- L2M2 is now fully HTTP based with no external dependencies, taking the total recursive dependency count from ~60 to 0 and massively simplifying the unit test suite.
- Non-native JSON mode strategy now defaults to prepend for Anthropic models and strip for all others.
