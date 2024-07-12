# Changelog

_Current version: 0.0.24_

[PyPi link](https://pypi.org/project/l2m2/)

### 0.0.24 - July 12, 2024

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
