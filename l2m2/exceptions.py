class LLMTimeoutError(Exception):
    """Raised when a request to an LLM provider API times out."""

    pass


class LLMRateLimitError(Exception):
    """Raised when a request to an LLM provider API is rate limited."""

    pass
