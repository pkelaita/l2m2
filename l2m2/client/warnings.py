import warnings  # pragma: no cover
from typing import Any, Callable


def deprecated(message: str) -> Callable[..., Any]:  # pragma: no cover
    def deprecated_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def deprecated_func(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                "{} is deprecated and will be removed in a future version. {}".format(
                    func.__name__, message
                ),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator
