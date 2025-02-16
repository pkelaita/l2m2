from typing import Dict, Any, Optional, Union
import asyncio
import httpx

from l2m2.client.base_llm_client import BaseLLMClient, DEFAULT_TIMEOUT_SECONDS
from l2m2.memory.base_memory import BaseMemory
from l2m2.tools.json_mode_strategies import JsonModeStrategy
from l2m2.exceptions import L2M2UsageError


# Helper function to check if we are currently in an async context.
def _is_async_context() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    # It hurts my soul to use try-catch for control flow here, but as far as I know it's
    # the only way to do this. If anyone knows a cleaner method, please let me know...
    except RuntimeError as e:
        if "no running event loop" in str(e):
            return False
        raise  # pragma: no cover


class LLMClient(BaseLLMClient):
    def __init__(
        self,
        providers: Optional[Dict[str, str]] = None,
        memory: Optional[BaseMemory] = None,
    ) -> None:
        """Initializes the LLM Client.

        Args:
            providers ([Dict[str, str]], optional): Mapping from provider name to API key.
                For example::

                    {
                        "openai": "openai-api
                        "anthropic": "anthropic-api-key",
                        "google": "google-api-key",
                    }

                Defaults to `None`.
            memory (BaseMemory, optional): The memory object to use. Defaults to `None`, in which
                case memory is not enabled.

        Raises:
            L2M2UsageError: If an invalid provider is specified in `providers`.
            L2M2UsageError: If `LLMClient` is instantiated in an asynchronous context.
        """
        if _is_async_context():
            raise L2M2UsageError(
                "LLMClient cannot be instantiated in an async context. Use AsyncLLMClient instead."
            )

        super(LLMClient, self).__init__(api_keys=providers, memory=memory)

    async def _sync_fn_wrapper(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        async with httpx.AsyncClient() as temp_client:
            original_client = self.httpx_client
            self.httpx_client = temp_client
            try:
                return await fn(*args, **kwargs)
            finally:
                self.httpx_client = original_client

    def call(  # type: ignore
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        prefer_provider: Optional[str] = None,
        json_mode: bool = False,
        json_mode_strategy: Optional[JsonModeStrategy] = None,
        timeout: Optional[int] = DEFAULT_TIMEOUT_SECONDS,
        bypass_memory: bool = False,
        alt_memory: Optional[BaseMemory] = None,
        extra_params: Optional[Dict[str, Union[str, int, float]]] = None,
    ) -> str:
        """Performs inference on any active model.

        Args:
            model (str): The active model to call.
            prompt (str): The user prompt for which to generate a completion.
            system_prompt (str, optional): The system prompt to send to the model. If the specified
                model does not support system prompts, it is prepended to the user prompt. Defaults
                to None.
            temperature (float, optional): The sampling temperature for the model. If not specified,
                the provider's default value for the model is used. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. If not specified,
                the provider's default value for the model is used. Defaults to None.
            prefer_provider (str, optional): The preferred provider to use for the model, if the
                model is available from multiple active providers. Defaults to None.
            json_mode (bool, optional): Whether to return the response in JSON format. Defaults to False.
            json_mode_strategy (JsonModeStrategy, optional): The strategy to use to enforce JSON outputs
                when `json_mode` is True. If `None`, the default strategy will be used:
                `JsonModeStrategy.prepend()` for Anthropic, and `JsonModeStrategy.strip()` for all other
                providers. Defaults to `None`.
            timeout (int, optional): The timeout in seconds for the LLM request. Can be set to `None`,
                in which case the request will be allowed to run indefinitely. Defaults to `10`.
            bypass_memory (bool, optional): Whether to bypass memory when calling the model. If `True`, the
                model will not read from or write to memory during the call if memory is enabled. Defaults
                to `False`.
            alt_memory (BaseMemory, optional): An alternative memory object to use for this call only. This
                is very useful for asynchronous workflows where you want to keep track of multiple memory
                streams in parallel without risking race conditions. Defaults to `None`.
            extra_params (Dict[str, Union[str, int, float]], optional): Extra parameters to pass to the model.
                Defaults to `None`.
            extra_headers (Dict[str, str], optional): Extra HTTP headers to pass in the request to the service
                hosting the model. Defaults to `None`.

        Raises:
            L2M2UsageError: If the provided model is not active and/or not available.
            L2M2UsageError: If the model is available from multiple active providers neither `prefer_provider`
                nor a default provider is specified.
            L2M2UsageError: If `prefer_provider` is specified but not active.
            L2M2UsageError: If `LLMClient.call` is called in an asynchronous context.

        Returns:
            str: The model's completion for the prompt, or an error message if the model is
                unable to generate a completion.
        """
        if _is_async_context():
            raise L2M2UsageError(
                "LLMClient cannot be instantiated in an async context. Use AsyncLLMClient instead."
            )
        result = asyncio.run(
            self._sync_fn_wrapper(
                super(LLMClient, self).call,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                prefer_provider=prefer_provider,
                json_mode=json_mode,
                json_mode_strategy=json_mode_strategy,
                timeout=timeout,
                bypass_memory=bypass_memory,
                alt_memory=alt_memory,
                extra_params=extra_params,
            )
        )
        return str(result)
