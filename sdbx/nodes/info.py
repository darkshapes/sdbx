import typing
import inspect

from pathlib import Path
from dataclasses import asdict
from collections import OrderedDict
from collections.abc import Iterator

from sdbx import logger
from sdbx.nodes.helpers import format_name, timing
from sdbx.nodes.types import Slider, Numerical, Text, Validator, Dependent, Name

class NodeInfo:
    @timing(logger.debug)
    def __init__(self, fn, path=None, name=None, display=False):
        self.fname = getattr(fn, '__name__')

        if path:
            path = path.rstrip("/")

        self.name = name or format_name(self.fname)
        self.path = path or Path(inspect.getfile(fn)).stem

        self.display = display

        self.inputs = {
            "required": OrderedDict(),
            "optional": OrderedDict(),
        }
        self.outputs = OrderedDict()

        annotations = inspect.get_annotations(fn)
        signature = inspect.signature(fn)

        self.generator = fn.generator
        self.steps = getattr(fn, "steps", None)

        try:
            for key, param in signature.parameters.items():
                annotation = annotations.get(key, param.annotation)
                if param.default is inspect.Parameter.empty:
                    # No default value
                    self.put(key, annotation)
                else:
                    # Default value exists, differentiate between None and other values
                    self.put(key, annotation, param.default)
            
            # Handling the return annotation separately
            if 'return' in annotations:
                return_annotation = annotations['return']

                if self.generator:
                    assert typing.get_origin(return_annotation) is Iterator, "Generator node must return I[yield type] type"
                    iterator_args = typing.get_args(return_annotation)
                    assert len(iterator_args) > 0, "Generator return type requires a type in the I[yield type] brackets"
                    return_annotation = iterator_args[0] if len(iterator_args) == 1 else Tuple[iterator_args]

                if typing.get_origin(return_annotation) is tuple:
                    for v in typing.get_args(return_annotation):
                        self.put('return', v)
                else:
                    self.put('return', return_annotation)

        except Exception as e:
            logger.exception(e)
            raise Exception(f"Error parsing node {self.name}: {e}")

    def put(self, key, value, default=inspect.Parameter.empty):
        if value is None:
            if key == 'return':
                self.terminal = True
                return
            raise Exception(f"Argument {key} cannot be typed as None.")

        info = {}
        necessity = "required"
        output = key == "return"

        # TODO: multiple unnamed outputs of same type will overwrite each other
        name = format_name(value.__name__)

        if not output:
            info["fname"] = key
            name = format_name(key)

        vtype = typing.get_origin(value)

        # Handle default values
        if default is not inspect.Parameter.empty:
            info["default"] = default
            necessity = "optional"

        # Parse annotations
        if vtype is typing.Annotated:
            base_type, *metadata = typing.get_args(value)

            info["type"] = base_type.__name__.capitalize()

            for item in metadata:
                # Check for Name
                if isinstance(item, Name):
                    name = item.name

                if base_type in (int, float):
                    if isinstance(item, (Numerical, Slider)): # Check for Numerical or Slider
                        info["constraints"] = asdict(item)
                        info["display"] = type(item).__name__.lower() # numerical | slider

                if base_type is str:
                    if isinstance(item, Text): # Check for Text
                        info["constraints"] = asdict(item)
            
                # Check for Dependent
                if isinstance(item, Dependent):
                    info["dependent"] = asdict(item)

                # TODO: Check for Validator
        
        if vtype is typing.Literal:
            info["type"] = "OneOf"

            choices = typing.get_args(value)
            if len(choices) == 0:
                # raise Exception(f"{key} of type Literal has no values.")
                logger.warning(f"Literal-typed argument '{key}' in node '{self.name}' has no values. This will show as an empty list of choices to the client.")

            info["choices"] = choices
        
        if not vtype:
            info["type"] = value.__name__.capitalize()

        if output:
            self.outputs[name] = info
        else:
            self.inputs[necessity][name] = info
    
    def dict(self):
        return {
            "path": self.path,
            "fname": self.fname, 
            "inputs": self.inputs,
            "outputs": self.outputs,
            "display": self.display,
            **({"steps": self.steps} if self.steps is not None else {})
        }