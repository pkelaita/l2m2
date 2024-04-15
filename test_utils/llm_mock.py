from typing import Optional
from unittest.mock import Mock
from functools import reduce


def construct_mock_from_path(
    response_path: str,
    final_response: Optional[str] = "response",
) -> Mock:
    """
    Creates a Mock object based on a dot-separated path, assuming any list indices specified are '[0]'.
    Example input: 'choices[0].message.content' results in nested Mock objects,
    where each list index assumed to be '[0]', and the final 'content' returns 'response'.
    """
    parts = response_path.split(".")
    mock = Mock()
    current = mock

    # Parts 1 to n-1 create nested Mock objects
    for part in parts[:-1]:
        new_mock = Mock()

        if "[0]" in part:
            name = part.split("[")[0]
            setattr(current, name, [new_mock])
        else:
            new_mock = Mock()
            setattr(current, part, new_mock)

        current = new_mock

    # The last part sets the final return value
    final_part = parts[-1]
    if "[0]" in final_part:
        # name = final_part.split("[")[0]
        # final_mock = Mock(return_value=final_response)
        # setattr(current, name, [final_mock])
        # TODO: Doesn't work yet, fix this
        raise NotImplementedError("Final part cannot be a list")
    else:
        setattr(current, final_part, final_response)

    return mock


def get_nested_attribute(
    obj: object,
    attr_path: str,
) -> object:
    """
    Get a nested attribute of an object by following a dot-separated path.
    """

    def getattr_with_check(obj, attr):  # type: ignore
        if not hasattr(obj, attr):
            raise AttributeError(f"Object {obj} has no attribute {attr}")
        return getattr(obj, attr)

    return reduce(getattr_with_check, attr_path.split("."), obj)
