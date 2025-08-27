import pytest
from datasets import Dataset, DatasetDict
from jinja2.exceptions import UndefinedError

from blossomtune_mlx.dataset import (
    extract_field_names_from_template,
    reformat_dynamic,
    render_template,
    process_dataset_dynamic,
)


def test_render_template_valid():
    """Test rendering a valid template."""
    result = render_template("Hello {{ name }}!", {"name": "Alice"})
    assert result == "Hello Alice!"


def test_render_template_missing_context():
    """Test template with missing context variables."""
    with pytest.raises(UndefinedError):
        render_template("Hello {{ name }}!", {})


@pytest.mark.parametrize(
    "template, expected",
    [
        ("Hello {{ name }}!", ["name"]),
        ("{{a}} {{b}} {{c}}", ["a", "b", "c"]),
        ("{{ a }} {{a}} {{ b }}", ["a", "b"]),
        ("No variables", []),
        ("", []),
        ("{{ valid }} {{ valid_with_space }}", ["valid", "valid_with_space"]),
    ],
)
def test_extract_field_names_from_template(template, expected):
    """Test field name extraction from templates."""
    result = extract_field_names_from_template(template)
    assert sorted(result) == sorted(expected)  # Compare sorted lists


def test_reformat_dynamic_success():
    """Test successful reformatting of an example."""
    example = {"user": "Bob", "response": "Hi!"}
    prompt_tpl = "User: {{ user }}"
    completion_tpl = "Response: {{ response }}"

    updated = reformat_dynamic(example, prompt_tpl, completion_tpl)
    assert updated["prompt"] == "User: Bob"
    assert updated["completion"] == "Response: Hi!"


def test_reformat_dynamic_invalid_template(capsys):
    """Test invalid templates print warnings and return empty strings."""
    example = {"user": "Bob"}
    invalid_tpl = "{{ user | nonexistent_filter }}"

    updated = reformat_dynamic(example, invalid_tpl, invalid_tpl)
    captured = capsys.readouterr()

    assert "Warning: Prompt formatting error" in captured.out
    assert updated["prompt"] == ""
    assert updated["completion"] == ""


def test_process_dataset_single():
    """Test processing a single Dataset."""
    dataset = Dataset.from_dict({"user": ["Alice", "Bob"]})
    prompt_tpl = "Hello {{ user }}!"
    completion_tpl = "Goodbye {{ user }}!"

    processed = process_dataset_dynamic(dataset, prompt_tpl, completion_tpl)
    assert processed["prompt"] == ["Hello Alice!", "Hello Bob!"]
    assert processed["completion"] == ["Goodbye Alice!", "Goodbye Bob!"]


def test_process_dataset_dict():
    """Test processing a DatasetDict with multiple splits."""
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"user": ["Alice"]}),
            "test": Dataset.from_dict({"user": ["Bob"]}),
        }
    )
    prompt_tpl = "Hi {{ user }}"
    completion_tpl = "Bye {{ user }}"

    processed = process_dataset_dynamic(dataset, prompt_tpl, completion_tpl)
    assert processed["train"]["prompt"] == ["Hi Alice"]
    assert processed["test"]["completion"] == ["Bye Bob"]


def test_process_dataset_invalid_type():
    """Test passing invalid type raises error."""
    with pytest.raises(TypeError):
        process_dataset_dynamic("not_a_dataset", "", "")
