# ### <!-- // /*  d a r k s h a p e s */ -->

from types import FunctionType, NoneType
from typing import Optional, Type, Dict, Any, List
import re
from pydantic import create_model, Field
from nnll.metadata.helpers import make_callable
from nnll.monitor.file import dbuq
from mir.json_cache import JSONCache, MIR_PATH_NAMED, TEMPLATE_PATH_NAMED
from mir.constants import PkgType

from sdbx.nodes.generate import terminal_gen  # , autotype_methods
from sdbx.nodes.types import Text, Slider, node, A

MIR_DATA = JSONCache(MIR_PATH_NAMED)
TEMPLATE_DATA = JSONCache(TEMPLATE_PATH_NAMED)


def track_imports(input_string: Optional[str]) -> Dict[str, FunctionType]:
    """Track imports from a given string and return a dictionary mapping module names to their base import functions.\n
    :param input_string: A string containing import statements or paths to modules.
    :return: A dictionary where keys are module names and values are the imported functions."""

    from importlib import import_module

    pattern: str = r"(\b\w+\.\w+\b)"  # locate "[word."" or " word."
    local_name_server: Dict[str, str] = {"types": import_module("types", "NoneType")}
    if input_string:
        matches: List[List[str]] = re.findall(pattern, input_string)
        found: List[str]
        for found in matches:
            package_name = found.split(".")[0]
            local_name_server.setdefault(package_name, import_module(package_name))
    return local_name_server


def serialize_function(name: str, args_def: dict[str, type], body: str = "    ") -> FunctionType:
    """Dynamically generate a function string with specified keyword arguments and types.\n
    :param name: Function name.
    :param kwarg_types: Mapping of keyword arg names to types.
    :body: Function body as string (indented with spaces).
    :returns: A callable function object with the provided attributes"""

    import ast

    static_args = str(args_def)[1:-1].replace("<class '", "").replace("'>", "").replace("'", "")
    func_code = f"def {name}({static_args}):\n{body}\n"
    local_name_server = track_imports(static_args)
    dbuq(local_name_server)
    node_function = ast.parse(func_code, mode="exec")
    code = compile(source=node_function, filename="<dynamic>", mode="exec")
    exec(code, globals(), local_name_server)
    return local_name_server


@MIR_DATA.decorator
def _read_data(data: Optional[Dict[str, str]] = None):
    return data


class NodeArray:
    """"""


node_array = NodeArray()

field_book = []
TEMPLATE_DATA._load_cache()
template_data = TEMPLATE_DATA._cache
mir_data = _read_data()
for series, compatibility in mir_data.items():
    arch_name = series.split(".")[1:]
    for comp_name, comp_data in compatibility.items():
        if any(arch in arch_name for arch in template_data["arch"]["diff"]) and comp_data.get("pkg", 0) and comp_data["pkg"]["0"].get("diffusers"):
            class_name = comp_data["pkg"]["0"]["diffusers"]
            if "Pipeline" in class_name:
                dbuq(class_name)
                node_frame = terminal_gen(class_name, PkgType.DIFFUSERS)
                func_name = node_frame["func_name"]
                try:
                    local_name_server = serialize_function(name=func_name, args_def=node_frame["generation_args"], body="    return 12")
                    static_node = node(name=node_frame["node_name"], path="diffusers")(local_name_server[func_name])
                    locals()[func_name] = static_node
                    # field = {node_frame["node_name"]: (node, ...), "default": static_node}
                    # field_book.append(create_model(class_name + "ElementBase", func_name=(node, static_node)))

                    # setattr(node_array, func_name, static_node)

                except ModuleNotFoundError:
                    pass
        elif any(arch in arch_name for arch in template_data["arch"]["xfmr"]) and comp_data.get("pkg", 0) and comp_data["pkg"]["0"].get("transformers"):
            class_name = comp_data["pkg"]["0"]["transformers"]
            if "Model" in class_name:
                dbuq(class_name)
                node_frame = terminal_gen(class_name, PkgType.TRANSFORMERS)
                func_name = node_frame["func_name"]
                try:
                    print(node_frame)
                    local_name_server = serialize_function(name=func_name, args_def=node_frame["generation_args"], body="    return 12")
                    static_node = node(name=node_frame["node_name"], path="transformers")(local_name_server[func_name])
                    locals()[func_name] = static_node

                    # field_book.append(
                    #     create_model(
                    #         class_name + "ElementBase",
                    #         func_name=(node, static_node),
                    #     ),
                    # )
                    # setattr(node_array, func_name, static_node)

                except ModuleNotFoundError:
                    pass
