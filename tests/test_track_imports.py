import pytest
import re
from types import FunctionType
from importlib import import_module
from typing import Dict

# Assuming the function `track_imports` is defined in a separate file, let's import it here.
from sdbx.nodes.base.nodes import track_imports


def test_track_basic():
    input_string = "types.NoneType"
    expected_output = {"types": import_module("types", "NoneType")}
    assert track_imports(input_string) == expected_output


def test_track_imports_multiple():
    input_string = "os.path"
    expected_output = {"types": import_module("types", "NoneType"), "os": import_module("os")}
    assert track_imports(input_string) == expected_output


def test_track_imports_non_extant():
    input_string = "non_existent_module.something"
    with pytest.raises(ModuleNotFoundError):
        assert track_imports(input_string)


def test_track_nested_modules():
    # we dont actually need to import the whole path, since the type annotations show it
    input_string = "os.path.join"
    expected_output = {"types": import_module("types", "NoneType"), "os": import_module("os")}
    assert track_imports(input_string) == expected_output
