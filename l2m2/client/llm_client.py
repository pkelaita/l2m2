from typing import Dict, Any, Optional
import asyncio
import httpx

from l2m2.client.base_llm_client import BaseLLMClient, DEFAULT_TIMEOUT_SECONDS
from l2m2.memory.base_memory import BaseMemory
from l2m2.tools.json_mode_strategies import JsonModeStrategy


class LLMClient(BaseLLMClient):
    def __init__(
        self,
        providers: Optional[Dict[str, str]] = None,
        memory: Optional[BaseMemory] = None,
    ) -> None:
        super(LLMClient, self).__init__(providers=providers, memory=memory)

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
    ) -> str:
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
            )
        )
        return str(result)

    def call_custom(  # type: ignore
        self,
        *,
        provider: str,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        json_mode_strategy: Optional[JsonModeStrategy] = None,
        timeout: Optional[int] = DEFAULT_TIMEOUT_SECONDS,
        bypass_memory: bool = False,
        alt_memory: Optional[BaseMemory] = None,
    ) -> str:
        result = asyncio.run(
            self._sync_fn_wrapper(
                super(LLMClient, self).call_custom,
                provider=provider,
                model_id=model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                json_mode_strategy=json_mode_strategy,
                timeout=timeout,
                bypass_memory=bypass_memory,
                alt_memory=alt_memory,
            )
        )
        return str(result)

    # Inherit docstrings
    __init__.__doc__ = BaseLLMClient.__init__.__doc__
    call.__doc__ = BaseLLMClient.call.__doc__
    call_custom.__doc__ = BaseLLMClient.call_custom.__doc__
