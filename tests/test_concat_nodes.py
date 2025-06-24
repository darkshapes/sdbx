import pytest
from types import FunctionType
from ast import parse
from sdbx.nodes.base.nodes import serialize_function


def test_serialize_function_default_return_type():
    func_name = "test_func"
    function = serialize_function(func_name, "arg1: int, arg2: str", body="    return arg1 + arg2")
    assert isinstance(function[func_name], FunctionType)
    assert function[func_name]("1", "2") == "12"


def test_serialize_function_specified_return_type():
    func_name = "test_func"
    function = serialize_function(func_name, "arg1: int, arg2: str", body="    return str(arg1) + arg2")
    assert isinstance(function[func_name], FunctionType)
    assert function[func_name](1, "2") == "12"


def test_serialize_function_no_body():
    func_name = "test_func"
    function = serialize_function(func_name, "arg1: int, arg2: str", body="\n    pass")
    assert isinstance(function[func_name], FunctionType)
    assert function[func_name](1, "2") is None


def test_serialize_function_import_dependencies():
    func_name = "test_func"
    function = serialize_function(func_name, "arg1: int, arg2: str", body="    return arg1 + int(arg2)")
    assert isinstance(function[func_name], FunctionType)
    assert function[func_name](1, "2") == 3


def test_serialize_function_invalid_return():
    with pytest.raises(IndentationError):
        serialize_function("test_func", "arg1: int, arg2: str", body="")


def test_serialize_function_invalid_inputs():
    with pytest.raises(SyntaxError):
        serialize_function("test_func", "arg1 int, arg2: str", body="")


def test_serialize_function_invalid_input():
    with pytest.raises(SyntaxError):
        serialize_function("test_func", "arg1 Numerical, arg2: str", body="    return data")

    # notice that the type MUST be annotated with the import path to be fully functional
    serialize_function("test_func", "arg1: sdbx.nodes.types.Numerical, arg2: str", body="    return data")
