# Changelog

_Current version: 0.0.22_

[PyPi link](https://pypi.org/project/l2m2/)

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
