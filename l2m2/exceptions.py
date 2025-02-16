class LLMTimeoutError(Exception):
    """Raised when a request to an LLM provider API times out."""

    pass


class LLMRateLimitError(Exception):
    """Raised when a request to an LLM provider API is rate limited."""

    pass


class LLMOperationError(Exception):
    """Raised when a model does not support a particular feature or mode."""

    pass


class L2M2UsageError(Exception):
    """Raised when l2m2 is used incorrectly or inappropriately."""

    pass
