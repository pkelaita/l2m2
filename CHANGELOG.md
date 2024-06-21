# Changelog

_Current version: 0.0.21_

[PyPi link](https://pypi.org/project/l2m2/)

### [0.0.21] - June 20, 2024

#### Added

- This changelog (finally – oops)
- Support for Anthropic's [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) released today

#### Changed

- L2M2 is now fully HTTP based with no external dependencies, taking the total recursive dependency count from ~60 to 0 and massively simplifying the unit test suite.
- Non-native JSON mode strategy now defaults to prepend for Anthropic models and strip for all others.
