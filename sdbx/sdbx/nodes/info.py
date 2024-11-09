import typing
import inspect

from pathlib import Path
from dataclasses import asdict
from collections import OrderedDict
from collections.abc import Iterator

from sdbx import logger
from sdbx.nodes.helpers import format_name #, timing
from sdbx.nodes.types import Annotation, Slider, Numerical, Text, Validator, Dependent, Name

class NodeInfo:
    # @timing(logger.debug)
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

                if isinstance(return_annotation, tuple):
                    for v in return_annotation:
                        self.put('return', v)
                else:
                    self.put('return', return_annotation)
            else:
                self.terminal = True

        except Exception as e:
            logger.exception(e)
            raise Exception(f"Error parsing node {self.name}: {e}")

    def put(self, key, value, default=inspect.Parameter.empty):
        if value is None:
            if key == 'return':
                self.terminal = True
                return
            raise Exception(f"Argument {key}: Cannot be typed as None.")

        necessity = "required"
        output = key == "return"

        name = format_name(value.__name__ if output else format_name(key))

        try:
            info, prospective = self._get_info(value)
            name = prospective or name
        except Exception as e:
            # logger.exception(e)
            raise Exception(f"Argument {key}: {e}")

        # Handle default values
        if default is not inspect.Parameter.empty:
            info["default"] = default
            necessity = "optional"

        if output:
            self.outputs[name] = info
        else:
            info["fname"] = key
            self.inputs[necessity][name] = info
    
    def _get_info(self, v, passed_type=False, error_on_list=False):
        info = {}
        name = None  # TODO: multiple unnamed outputs of same type will overwrite each other

        vt = (v if passed_type else typing.get_origin(v)) or v

        # Parse annotations
        if vt is typing.Annotated:
            base_type, *metadata = typing.get_args(v)

            info, name = self._get_info(base_type)
            info["type"] = base_type.__name__.capitalize()

            for item in metadata:
                if not isinstance(item, Annotation):
                    raise Exception(f"Cannot annotate {base_type} with non-annotation type {item}.")

                if not item.check(base_type):
                    raise Exception(f"Argument {item} is incompatible with Annotated of type {base_type}.")
                
                # Check for Name
                if isinstance(item, Name):
                    name = item.name
                else:
                    info |= item.serialize()
        elif vt is typing.Literal:
            info["type"] = "OneOf"

            choices = typing.get_args(v)
            if len(choices) == 0:
                # raise Exception(f"Literal-typed argument has no values.")
                logger.warning(f"Literal-typed argument in node '{self.name}' has no values. This will show as an empty list of choices to the client.")

            info["choices"] = choices
        elif vt is list or vt is tuple:
            if error_on_list:
                raise Exception("Argument cannot contain nested iterator types.")

            info["type"] = "List"

            base_types = set(typing.get_args(v))

            if len(base_types) == 0:
                raise Exception(f"Argument contains no member types.")
            elif len(base_types) != 1:
                raise Exception(f"Argument contains more than one member type.")
            else:
                base_type = next(iter(base_types))

            # if base_type not in primitives:
                # raise Exception(f"Argument cannot contain non-primitive type {t}.")
            
            info["sub"], name = self._get_info(base_type, passed_type=True, error_on_list=True)

            if vt is tuple:
                info["constraints"] = { "length": len(typing.get_args(v)) }
        else:
            info["type"] = v.__name__.capitalize()
        
        return info, name
    
    def dict(self):
        return {
            "path": self.path,
            "fname": self.fname, 
            "inputs": self.inputs,
            "outputs": self.outputs,
            "display": self.display,
            **({"steps": self.steps} if self.steps is not None else {})
        }