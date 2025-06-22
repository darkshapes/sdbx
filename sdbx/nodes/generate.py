from typing import Dict, Optional, Callable, Union, List
from mir.constants import PkgType


def terminal_gen_cls(class_name: str, pkg_name: PkgType) -> Dict[str, Dict[str, Union[Callable, List[str]]]]:
    """Generate parameter groups based on class name and package name.\n
    :param class_name: The name of the class from which to generate parameters.
    :param pkg_name: The name of the package where the class resides.
    :returns: A dictionary containing three groups of parameters: 'generation', 'pipe', and 'aux'.
    :raises: ModuleNotFoundError or AttributeError if path or class is not found"""

    from nnll.metadata.helpers import make_callable, class_parent
    from nnll.tensor_pipe.deconstructors import get_code_names, root_class
    from nnll.monitor.file import dbuq

    pkg_name = pkg_name.value[1].lower()
    pipe_args = {}
    pipe_aux = {}
    code_name = get_code_names(class_name=class_name, pkg_name=pkg_name)
    parent_folder = class_parent(code_name=code_name, pkg_name=pkg_name)
    dbuq(code_name, parent_folder)
    if pkg_name == "diffusers":
        pipe_class_obj = make_callable(module_name=class_name, pkg_name_or_abs_path=".".join(parent_folder))
    elif pkg_name == "transformers":
        module_path = root_class(class_name, "transformers").get("config")
        pipe_class_obj = make_callable(module_name=class_name, pkg_name_or_abs_path=".".join(module_path[:3]))
    gen_args = pipe_class_obj.__call__.__annotations__
    pipe_args = pipe_class_obj.__init__.__annotations__
    dbuq(pipe_class_obj)
    if hasattr(pipe_class_obj, "_optional_components"):
        optional_components = pipe_class_obj._optional_components
        dbuq(optional_components)
        pipe_aux = {k: Optional[v] for k, v in pipe_args.items() if k in optional_components}
    parameter_groups = {
        "class_name": class_name,
        "pipe_class_obj": pipe_class_obj,
        "generation": gen_args,
        "pipe": pipe_args,
        "aux": pipe_aux,
    }
    return parameter_groups


def terminal_gen_meth(class_name: Union[str, Callable], pkg_type: PkgType) -> Dict[str, Dict[str, Callable]]:
    from sdbx.nodes.types import Text, Numerical, Slider
    from nnll.metadata.helpers import make_callable

    type_map = {
        str: Text,
        float: Numerical,
        int: Slider,
    }
    if isinstance(class_name, str):
        class_obj = make_callable(class_name, pkg_type.value[1].lower())
    sub_functions = [getattr(class_obj, meth_name) for meth_name in vars(class_obj) if meth_name[0] != "_"]
    function_parameters = [meth_obj.__annotations__ for meth_obj in sub_functions if hasattr(meth_obj, "__annotations__") and meth_obj.__annotations__][0]
    for k, v in function_parameters.items():
        for data_type, note in type_map.items():
            if isinstance(v, data_type):
                v.update(note(v))
    return function_parameters
