from typing import Optional
from enum import Enum

DEFAULT_PREFIX = "Here is the JSON response: "


class StrategyName(Enum):
    STRIP = "strip"
    PREPEND = "prepend"


class JsonModeStrategy:
    def __init__(
        self,
        strategy_name: StrategyName,
        prefix: Optional[str] = None,
    ) -> None:
        self.strategy_name = strategy_name
        self.prefix = prefix

    @classmethod
    def strip(cls) -> "JsonModeStrategy":
        return cls(StrategyName.STRIP)

    @classmethod
    def prepend(cls, custom_prefix: str = DEFAULT_PREFIX) -> "JsonModeStrategy":
        return cls(StrategyName.PREPEND, custom_prefix)


def get_extra_message(strategy: JsonModeStrategy) -> Optional[str]:
    if strategy.strategy_name == StrategyName.PREPEND:
        assert strategy.prefix is not None
        return strategy.prefix + "{"

    return None


def run_json_strats_out(
    strategy: JsonModeStrategy,
    output: str,
) -> str:
    if strategy.strategy_name == StrategyName.PREPEND:
        return "{" + output

    if strategy.strategy_name == StrategyName.STRIP:
        start = output.find("{")
        end = output.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return output
        return output[start : end + 1]  # noqa: E203
