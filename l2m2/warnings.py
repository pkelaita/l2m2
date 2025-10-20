import warnings  # pragma: no cover
from typing import Any
from collections.abc import Callable


def deprecated(message: str) -> Callable[..., Any]:  # pragma: no cover
    def deprecated_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def deprecated_func(*args: Any, **kwargs: Any) -> Any:
            warnings.simplefilter("default", DeprecationWarning)
            func_name = getattr(func, "__name__", repr(func))
            warnings.warn(
                f"{func_name} is deprecated and will be removed in a future version. {message}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator
