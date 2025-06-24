from typing import Dict, Callable, Union, List
from types import FunctionType
from mir.constants import PkgType
import typing


def terminal_gen(class_name: str, pkg_name: PkgType) -> Dict[str, Dict[str, Union[Callable, List[str]]]]:
    """Generate parameter groups based on class name and package name.\n
    :param class_name: The name of the class from which to generate parameters.
    :param class_obj: The class object referred to in.
    :returns: A dictionary containing three groups of parameters: 'generation', 'pipe', and 'aux'.
    :raises: `ModuleNotFoundError` or `AttributeError` if path or class is not found"""

    from nnll.metadata.helpers import snake_caseify, make_callable
    from nnll.monitor.file import dbuq

    class_obj = make_callable(module_name=class_name, pkg_name_or_abs_path=pkg_name.value[1].lower())
    pipe_args = {}
    pipe_aux = {}
    gen_args = typing.get_type_hints(class_obj.__call__)
    pipe_args = typing.get_type_hints(class_obj.__init__)
    dbuq(class_obj)

    if hasattr(class_obj, "_optional_components"):
        optional_components = class_obj._optional_components
        pipe_aux = {}
        if optional_components and pipe_args:
            for component in optional_components:
                if pipe_args.get(component):
                    pipe_aux.setdefault(component, pipe_args[component])

    parameter_groups = {
        "node_name": class_name + "Node",
        "func_name": snake_caseify(class_name),
        "class_obj": class_obj,
        "generation_args": gen_args,
        "pipeline_args": pipe_args,
        "aux_classes": pipe_aux,
    }
    return parameter_groups


# def autotype_methods(class_name: Union[str, Callable], pkg_type: PkgType) -> Dict[str, Dict[str, Callable]]:
#     from sdbx.nodes.types import Text, Numerical, Slider
#     from nnll.metadata.helpers import make_callable

#     type_map = {
#         str: Text,
#         float: Numerical,
#         int: Slider,
#     }
#     if isinstance(class_name, str):
#         class_obj = make_callable(class_name, pkg_type.value[1].lower())
#     sub_functions = [getattr(class_obj, meth_name) for meth_name in vars(class_obj) if meth_name[0] != "_"]
#     function_parameters = [meth_obj.__annotations__ for meth_obj in sub_functions if hasattr(meth_obj, "__annotations__") and meth_obj.__annotations__][0]
#     for k, v in function_parameters.items():
#         for data_type, note in type_map.items():
#             if isinstance(v, data_type):
#                 v.update(note(v))
#     return function_parameters
