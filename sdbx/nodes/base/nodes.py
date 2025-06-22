# ### <!-- // /*  d a r k s h a p e s */ -->

from nnll.metadata.helpers import make_callable
from mir.constants import PkgType
from sdbx.nodes.types.decorator import node
from sdbx.nodes.generate import terminal_gen_cls, terminal_gen_meth
from typing import Dict, Any, Callable, Type


def generate_node(node_name: str, function: Callable, inputs: Dict[str, Type]) -> Callable:
    """
    Generate a dynamic node with the given name, input types, and implementation function.

    Args:
        node_name: Name of the node
        inputs: Dictionary mapping parameter names to their types
        function: Implementation function that accepts **kwargs

    Returns:
        A dynamically created node function decorated with @node
    """

    def dynamic_node(**kwargs) -> Any:
        """Dynamic node implementation that uses the provided function"""
        return function(**kwargs)

    # Set the __annotations__ for type hints
    dynamic_node.__annotations__ = {"return": Callable}
    dynamic_node.__annotations__ = {k: v for k, v in inputs.items() if k != "return"}

    # Set a docstring with parameter information
    params_doc = "\n    ".join([f"{name}: {type_.__name__}" for name, type_ in inputs.items() if name != "return"])
    dynamic_node.__doc__ = f"""{node_name} node

    Parameters
    ----------
    {params_doc}

    Returns
    -------
    {inputs.get("return", Any).__name__}
    """
    decorated_node = node(name=node_name)(dynamic_node)
    return decorated_node


DynamicLoadVaeNode = generate_node(
    node_name="AutoencoderKL" + " Node",
    function=make_callable("AutoencoderKL", PkgType.DIFFUSERS.value[1].lower()),
    inputs=terminal_gen_meth("AutoencoderKL", PkgType.DIFFUSERS),
)
