import asyncio
import inspect
from typing import Optional, List, Any

from l2m2.client import LLMClient


class AsyncLLMClient(LLMClient):
    """A high-level interface for asynchronously interacting with L2M2's language models."""

    def from_client(client: LLMClient) -> "AsyncLLMClient":
        """Create an AsyncLLMClient from an LLMClient.

        Args:
            client (LLMClient): The base client to wrap.

        Returns:
            AsyncLLMClient: The new asynchronous client with the same models and providers active.
        """
        return AsyncLLMClient(client.API_KEYS)

    async def call_async(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Asynchronously performs inference on any active model.

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

        Raises:
            ValueError: If the provided model is not active and/or not available.

        Returns:
            str: The model's completion for the prompt, or an error message if the model is
                unable to generate a completion.
        """
        return await self._call_wrapper(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def call_concurrent(
        self,
        *,
        n: int,
        models: List[str],
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        temperatures: Optional[List[float]] = None,
        max_tokens: Optional[List[int]] = None,
    ) -> List[str]:
        """Makes multiple concurrent calls to a given set of models, with a given set oof
        parameters (e.g. model, prompt, temperature, etc). Each parameter is passed in as a list
        of length 1 or n, where n is the number of calls to make. If a parameter is a list of
        length 1, the value is shared across all calls.

        For example, ::

            await client.call_concurrent(
                n=2,
                models=["gpt-4-turbo", "claude-3-opus"],
                prompts=["foo", "bar"],
            )

        will do send `foo` to `gpt-4-turbo` and `bar` to `claude-3-opus`, while ::

            await client.call_concurrent(
                n=2,
                models=["gpt-4-turbo", "claude-3-opus"],
                prompts=["foo"],
            )

        will send `foo` to both models. Similarly, ::

            await client.call_concurrent(
                n=2,
                models=["gpt-4-turbo"],
                prompts=["foo", "bar"],
            )

        will prompt `gpt-4-turbo` with both `foo` and `bar`.

        Args:
            n (int): Number of concurrent calls to make.
            models (List[str]): List of models to call, or a single model to use for all calls.
            prompts (List[str]): List of prompts to use, or a single prompt to use for all calls.
            system_prompts ([List[str]], optional): List of system prompts to use, or a single
                system prompt to use for all calls. Defaults to None.
            temperatures ([List[float]], optional): List of temperatures to use, or a single
                temperature to use for all calls. Defaults to None.
            max_tokens ([List[int]], optional): List of max_tokens to use, or a single max_tokens
                to use for all calls. Defaults to None.

        Raises:
            ValueError: If `n < 1`, or if any of the parameters are not of length `1` or `n`.

        Raises:
            ValueError: If the provided model is not active and/or not available.

        Returns:
            List[str]: The responses from each call, in the same order as the input parameters.
        """
        if n < 1:
            raise ValueError("n must be at least 1")

        params = [models, prompts, system_prompts, temperatures, max_tokens]
        for i, param in enumerate(params):
            if param is not None and len(param) not in [1, n]:
                argname = inspect.getfullargspec(self.call_concurrent).kwonlyargs[i]
                raise ValueError(
                    f"{argname} must have length 1 or {n}, got {len(param)}"
                )

        def get(param: Optional[List[Any]], i: int) -> Optional[Any]:
            if param is None:
                return None
            return param[0] if len(param) == 1 else param[i]

        calls = [
            self._call_wrapper(
                model=get(models, i),
                prompt=get(prompts, i),
                system_prompt=get(system_prompts, i),
                temperature=get(temperatures, i),
                max_tokens=get(max_tokens, i),
            )
            for i in range(n)
        ]
        return await asyncio.gather(*calls)

    async def _call_wrapper(self, **kwargs: Any) -> str:
        return self.call(**kwargs)
