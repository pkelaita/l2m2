import pytest

# We don't actually use this, we just need it for a type assertion
from unittest.mock import Mock

from test_utils.llm_mock import (
    construct_mock_from_path,
    get_nested_attribute,
)


# -- Tests for construct_mock_from_path -- #


def test_simple_path():
    """Test simple attribute access"""
    mock = construct_mock_from_path("attribute", "value")
    assert mock.attribute == "value"


def test_nested_path():
    """Test nested attribute access"""
    mock = construct_mock_from_path("level1.level2.attribute", "value")
    assert mock.level1.level2.attribute == "value"


def test_indexed_path():
    """Test indexed attribute access assumes [0]"""
    mock = construct_mock_from_path("level1[0].level2.attribute", "value")
    assert mock.level1[0].level2.attribute == "value"


def test_index_at_end_of_path():
    """Test indexed attribute access assumes [0]"""
    # Doesn't work yet, so throw an error
    with pytest.raises(NotImplementedError):
        construct_mock_from_path("level1.level2[0]", "value")


def test_deeply_nested_indexed_path():
    """Test deeply nested and indexed attribute access"""
    mock = construct_mock_from_path(
        "level1.level2[0].level3.level4[0].attribute", "value"
    )
    assert mock.level1.level2[0].level3.level4[0].attribute == "value"


# -- Tests for get_nested_attribute -- #


def test_get_attribute():
    """Test that a top-level attribute can be retrieved"""
    obj = Mock()
    obj.level1 = Mock()
    result = get_nested_attribute(obj, "level1")
    assert result == obj.level1


def test_get_nested_attribute():
    """Test that a nested attribute can be retrieved"""
    obj = Mock()
    obj.level1 = Mock()
    obj.level1.level2 = Mock()
    obj.level1.level2.level3 = Mock()
    obj.level1.level2.level3.attribute = "value"
    result = get_nested_attribute(obj, "level1.level2.level3.attribute")
    assert result == "value"


def test_invalid_path():
    """Test that an invalid path raises an appropriate exception"""

    class SomeClass:
        pass

    obj = SomeClass()
    with pytest.raises(AttributeError):
        get_nested_attribute(obj, "nonexistent.level2")
