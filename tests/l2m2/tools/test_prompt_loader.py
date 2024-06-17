import pytest
from unittest.mock import mock_open, patch

from l2m2.tools import PromptLoader


def test_load_prompt_str():
    prompt_loader = PromptLoader()
    prompt = "Hello, {{name}}!"
    variables = {"name": "world"}
    assert prompt_loader.load_prompt_str(prompt, variables) == "Hello, world!"

    prompt = "{{name}} is {{age}} years old, and {{name}} is a {{job}}."
    variables = {
        "name": "Pierce",
        "age": "24",
        "job": "developer",
    }
    assert (
        prompt_loader.load_prompt_str(prompt, variables)
        == "Pierce is 24 years old, and Pierce is a developer."
    )


def test_load_prompt_str_missing_variable():
    prompt_loader = PromptLoader()
    prompt = "{{name}} is {{age}} years old, and {{name}} is a {{job}}."
    variables = {
        "name": "Pierce",
        "age": "24",
    }
    with pytest.raises(ValueError):
        prompt_loader.load_prompt_str(prompt, variables)


def test_load_prompt_str_custom_var_markers():
    prompt_loader = PromptLoader(var_open="<%", var_close="%>")
    prompt = "Hello, <%name%>!"
    variables = {"name": "world"}
    assert prompt_loader.load_prompt_str(prompt, variables) == "Hello, world!"

    prompt = "<%name%> is <%age%> years old, and <%name%> is a <%job%>."
    variables = {
        "name": "Pierce",
        "age": "24",
        "job": "developer",
    }
    assert (
        prompt_loader.load_prompt_str(prompt, variables)
        == "Pierce is 24 years old, and Pierce is a developer."
    )


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="Hello, {{name}}! This is a test prompt.",
)
def test_load_prompt(mock_open):
    prompt_loader = PromptLoader(prompts_base_dir="a/b/c")
    prompt_file = "test_prompt_loader.txt"
    variables = {"name": "world"}
    assert (
        prompt_loader.load_prompt(prompt_file, variables)
        == "Hello, world! This is a test prompt."
    )

    prompt_file_path = f"{prompt_loader.prompts_base_dir}/{prompt_file}"
    mock_open.assert_called_once_with(prompt_file_path, "r")
