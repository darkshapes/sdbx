# ### <!-- // /*  d a r k s h a p e s */ -->

from types import NoneType
from nnll.metadata.helpers import make_callable
from mir.constants import PkgType
from sdbx.nodes.generate import terminal_gen_cls, terminal_gen_meth
from typing import Callable, Optional, Type, Dict, Any, Annotated as A
from sdbx.nodes.types import Text, Slider, node
from types import NoneType
from pydantic import create_model
import re
from importlib import import_module

inputs = terminal_gen_cls("FluxPipeline", PkgType.DIFFUSERS)


def find_imports(input_string):
    # Define the regex pattern
    pattern = r"(\b\w+\.\w+\b)"
    package_names = set()
    matches = re.findall(pattern, input_string)

    for match in matches:
        package_name = match.split(".")[0]
        package_names.add(package_name)

    # Print the unique root packages
    return package_names


arg_str = str(inputs["generation"])[1:-1].replace("<class '", "").replace("'>", "").replace("'", "")


def generate_function(name: str, arg_str: dict[str, type], return_type: type = None, body: str = "    return None") -> Callable:
    """
    Dynamically generate a function with specified keyword arguments and types.

    Parameters:
        name (str): Function name.
        kwarg_types (dict): Mapping of keyword arg names to types.
        return_type (type): Return type annotation.
        body (str): Function body as string (indented with spaces).

    Returns:
        Callable: The generated function.
    """
    # ret_str = f"{return_type}" if return_type else ""
    func_code = f"def {name}({arg_str}):\n{body}"

    local_ns = {}
    import_pkgs = find_imports(arg_str)
    for pkg in import_pkgs:
        local_ns.setdefault(pkg, import_module(pkg))

    # pattern = r"'.*?(?<!\\)'"
    exec(func_code, globals(), local_ns)
    # found = re.findall(pattern, str(e))
    # import_module(found[0].strip('"').strip("'"))

    return node(name=name, path="utils")(local_ns[name])


test_function_name = generate_function(name=inputs["class_name"] + "Node", arg_str=arg_str, return_type=bool, body="    return 12")
