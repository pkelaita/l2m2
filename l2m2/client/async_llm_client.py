import asyncio
import inspect
from typing import Optional, List, Any

from l2m2.client import LLMClient
from l2m2.tools.json_mode_strategies import JsonModeStrategy


class AsyncLLMClient(LLMClient):
    """A high-level interface for asynchronously interacting with L2M2's language models."""

    def from_client(client: LLMClient) -> "AsyncLLMClient":
        """Create an AsyncLLMClient from an LLMClient.

        Args:
            client (LLMClient): The base client to wrap.

        Returns:
            AsyncLLMClient: The new asynchronous client with the same models and providers active.
        """
        aclient = AsyncLLMClient(client.api_keys)
        aclient.set_preferred_providers(client.preferred_providers)
        return aclient

    async def call_async(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        prefer_provider: Optional[str] = None,
        json_mode: bool = False,
        json_mode_strategy: JsonModeStrategy = JsonModeStrategy.strip(),
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
            prefer_provider (str, optional): The preferred provider to use for the model, if the
                model is available from multiple active providers. Defaults to None.
            json_mode (bool, optional): Whether to return the response in JSON format. Defaults to False.
            json_mode_strategy (JsonModeStrategy, optional): The strategy to use to enforce JSON outputs
                when `json_mode` is True. Defaults to `JsonModeStrategy.strip()`.

        Raises:
            ValueError: If the provided model is not active and/or not available.
            ValueError: If the model is available from multiple active providers neither `prefer_provider`
                nor a default provider is specified.
            ValueError: If `prefer_provider` is specified but not active.

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
            prefer_provider=prefer_provider,
            json_mode=json_mode,
            json_mode_strategy=json_mode_strategy,
        )

    async def call_custom_async(
        self,
        *,
        provider: str,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        json_mode_strategy: JsonModeStrategy = JsonModeStrategy.strip(),
    ) -> str:
        """Asynchronously Performs inference on any model from an active provider that is not
        officially supported by L2M2. This method does not guarantee correctness.

        Args:
            provider (str): The provider to use. Must be one of the active providers.
            model_id (str): The ID of model to call. Must be the exact match to how you would call
                it with the provider's API. For example, `gpt-3.5-turbo-0125` can be used to call
                a legacy model from OpenAI as per the OpenAI API docs.
                (https://platform.openai.com/docs/api-reference/chat)
            prompt (str): The user prompt for which to generate a completion.
            system_prompt (str, optional): The system prompt to send to the model. Defaults to None.
            temperature (float, optional): The sampling temperature for the model. If not specified,
                the provider's default value for the model is used. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. If not specified,
                the provider's default value for the model is used. Defaults to None.
            json_mode (bool, optional): Whether to return the response in JSON format. Defaults to False.
            json_mode_strategy (JsonModeStrategy, optional): The strategy to use to enforce JSON outputs
                when `json_mode` is True. Defaults to `JsonModeStrategy.strip()`.

        Raises:
            ValueError: If the provided model is not active and/or not available.

        Returns:
            str: The model's completion for the prompt (correctness not guaranteed).
        """
        return await self._call_custom_wrapper(
            provider=provider,
            model_id=model_id,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            json_mode_strategy=json_mode_strategy,
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
        prefer_providers: Optional[List[str]] = None,
        json_modes: Optional[List[bool]] = None,
        json_mode_strategies: Optional[List[JsonModeStrategy]] = None,
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
            prefer_providers ([List[str]], optional): List of preferred providers to use for each
                model, if the model is available from multiple active providers. Defaults to None.
            json_modes ([List[bool]], optional): Whether to use JSON mode for each call. Defaults to None.
            json_mode_strategies ([List[JsonModeStrategy]], optional): The strategies to use to enforce
                JSON outputs. Defaults to None.

        Raises:
            ValueError: If `n < 1`, or if any of the parameters are not of length `1` or `n`.
            ValueError: If the provided model is not active and/or not available.
            ValueError: If the model is available from multiple active providers neither `prefer_provider`
                nor a default provider is specified.
            ValueError: If `prefer_provider` is specified but not active.

        Returns:
            List[str]: The responses from each call, in the same order as the input parameters.
        """
        _check_concurrent_params(
            n,
            [models, prompts, system_prompts, temperatures, max_tokens, json_modes],
            inspect.getfullargspec(self.call_concurrent).kwonlyargs,
        )

        calls = [
            self._call_wrapper(
                model=_get_helper(models, i),
                prompt=_get_helper(prompts, i),
                system_prompt=_get_helper(system_prompts, i),
                temperature=_get_helper(temperatures, i),
                max_tokens=_get_helper(max_tokens, i),
                prefer_provider=_get_helper(prefer_providers, i),
                json_mode=_get_helper(json_modes, i),
                json_mode_strategy=_get_helper(json_mode_strategies, i),
            )
            for i in range(n)
        ]
        return await asyncio.gather(*calls)

    async def call_custom_concurrent(
        self,
        *,
        n: int,
        providers: List[str],
        model_ids: List[str],
        prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        temperatures: Optional[List[float]] = None,
        max_tokens: Optional[List[int]] = None,
        json_modes: Optional[List[bool]] = None,
        json_mode_strategies: Optional[List[JsonModeStrategy]] = None,
    ) -> List[str]:
        """Makes multiple concurrent calls to a given set of user-given models, with a given set oof
        parameters (e.g. model_d, prompt, temperature, etc). Each parameter is passed in as a list
        of length 1 or n, where n is the number of calls to make. If a parameter is a list of
        length 1, the value is shared across all calls.

        For example, ::

            await client.call_custom_concurrent(
                n=2,
                providers=["openai", "anthropic"],
                model_ids=["gpt-4-turbo", "claude-3-opus"],
                prompts=["foo", "bar"],
            )

        will do send `foo` to `gpt-4-turbo` and `bar` to `claude-3-opus`, while ::

            await client.call_custom_concurrent(
                n=2,
                providers=["openai", "anthropic"],
                model_ids=["gpt-4-turbo", "claude-3-opus"],
                prompts=["foo"],
            )

        will send `foo` to both models. Similarly, ::

            await client.call_custom_concurrent(
                n=2,
                providers=["openai"],
                model_ids=["gpt-4-turbo"],
                prompts=["foo", "bar"],
            )

        will prompt `gpt-4-turbo` with both `foo` and `bar`.

        Simmiarly to `call_custom_async`, `call_custom_concurrent` allows you to call models
        from active providers that are not officially supported by L2M2.
        `call_custom_concurrent` does not guarantee correctness.

        Args:
            n (int): Number of concurrent calls to make.
            providers (List[str]): List of providers to use, or a single provider to use for all calls.
            model_ids (List[str]): List of models to call, or a single model to use for all calls.
            prompts (List[str]): List of prompts to use, or a single prompt to use for all calls.
            system_prompts ([List[str]], optional): List of system prompts to use, or a single
                system prompt to use for all calls. Defaults to None.
            temperatures ([List[float]], optional): List of temperatures to use, or a single
                temperature to use for all calls. Defaults to None.
            max_tokens ([List[int]], optional): List of max_tokens to use, or a single max_tokens
                to use for all calls. Defaults to None.
            json_modes ([List[bool]], optional): Whether to use JSON mode for each call. Defaults to None.
            json_mode_strategies ([List[JsonModeStrategy]], optional): The strategies to use to enforce
                JSON outputs. Defaults to None.

        Raises:
            ValueError: If `n < 1`, or if any of the parameters are not of length `1` or `n`.

        Raises:
            ValueError: If the provided model is not active and/or not available.

        Returns:
            List[str]: The responses from each call, in the same order as the input parameters.
        """
        _check_concurrent_params(
            n,
            [
                providers,
                model_ids,
                prompts,
                system_prompts,
                temperatures,
                max_tokens,
                json_modes,
                json_mode_strategies,
            ],
            inspect.getfullargspec(self.call_custom_concurrent).kwonlyargs,
        )

        calls = [
            self._call_custom_wrapper(
                provider=_get_helper(providers, i),
                model_id=_get_helper(model_ids, i),
                prompt=_get_helper(prompts, i),
                system_prompt=_get_helper(system_prompts, i),
                temperature=_get_helper(temperatures, i),
                max_tokens=_get_helper(max_tokens, i),
                json_mode=_get_helper(json_modes, i),
                json_mode_strategy=_get_helper(json_mode_strategies, i),
            )
            for i in range(n)
        ]
        return await asyncio.gather(*calls)

    async def _call_wrapper(self, **kwargs: Any) -> str:
        return self.call(**kwargs)

    async def _call_custom_wrapper(self, **kwargs: Any) -> str:
        return self.call_custom(**kwargs)


def _get_helper(param: Optional[List[Any]], i: int) -> Optional[Any]:
    if param is None:
        return None
    if len(param) == 1:
        return param[0]
    return param[i]


def _check_concurrent_params(
    n: int,
    params: List[Optional[List[Any]]],
    argnames: List[str],
) -> None:
    if n < 1:
        raise ValueError("n must be at least 1")

    for i, param in enumerate(params):
        if param is not None and len(param) not in [1, n]:
            raise ValueError(
                f"{argnames[i]} must have length 1 or {n}, got {len(param)}"
            )
