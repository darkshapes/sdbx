from enum import Enum
from functools import partial
from operator import lt, le, eq, ne, ge, gt
from dataclasses import asdict, dataclass, field
from inspect import signature, isgeneratorfunction
from typing import Annotated, Any, Callable, Dict, Generic, Literal, List, Optional, Tuple, TypeVar, Union, get_args, get_type_hints

# from torch import Tensor
# from torch.nn import Module

# from PIL import Image as ImageSource

# from sdbx.sd import CLIP as CLIPSource, VAE as VAESource
from sdbx.nodes.helpers import rename_class


## Node decorator ##
def node(fn=None, **kwargs): # Arguments defined in NodeInfo init
    """
    Decorator for nodes. All functions using this decorator will be automatically read by the node manager.

    Parameters
    ----------
        path : str
            The path that the node will appear in.
        name : str
            The display name of the node.
        display : bool
            Whether to display the output in the node.
    """
    if fn is None:
        return partial(node, **kwargs)

    fn.generator = isgeneratorfunction(fn)

    from sdbx.nodes.info import NodeInfo
    fn.info = NodeInfo(fn, **kwargs)

    # from sdbx.nodes.tuner import NodeTuner
    # fn.tuner = NodeTuner(fn)

    return fn


## Types ##

# Annotations
from typing import Annotated as A                   # A

# Iterators
from collections.abc import Iterator as I

# Primitives
# bool                                              # bool
# int                                               # int
# str                                               # str
# from typing import Dict                             # dict

# # From primitives
# class Conditioning(Dict[str, Tensor]): pass         # Conditioning; dict of tensors?

# # From existing classes
# class Image(CLIPSource): pass                       # Image
# # from typing import Literal                        # Literal; any choice between a fixed number of options
# class Model(Module): pass                           # Model

# class CLIP(CLIPSource):                             # CLIP
#     use_type_as_name = True
# class CLIPVision(CLIPSource):                       # CLIPVision
#     use_type_as_name = True
# class VAE(VAESource):                               # VAE
#     use_type_as_name = True

# # Annotated tensors, NOTE: need to define dimensionality?
# class Audio(Tensor): pass                           # Audio
# class ControlNet(Tensor): pass                      # ControlNet
# class Latent(Tensor): pass                          # Latent
# class Mask(Tensor): pass                            # Mask

# class CLIPVisionOutput(Tensor):                     # CLIPVisionOutput (??)
#     use_type_as_name = True

# # Model types
# class StyleModel(Module): pass                      # StyleModel

# class GLIGen(Module):                               # GLIGen
#     use_type_as_name = True


## Annotation classes ##
class AnnotationMeta(type):
    def __getitem__(cls, item):
        if not isinstance(item, tuple):
            item = (item,)
        return type('AnnotationInstance', (Annotation,), {'__args__': item})

class Annotation(metaclass=AnnotationMeta):
    def check(self, t):
        args = getattr(self, '__args__', ())
        return Any in args or any(issubclass(t, arg) for arg in args)

    def serialize(self):
        return { "constraints": asdict(self) }

@dataclass
class Name(Annotation[Any]):
    name: str = ""

    def serialize(self):  # special
        pass

@dataclass
class Condition:
    operator: Union[lt, le, eq, ne, ge, gt]
    value: Any

    def __init__(self, *args, **kwargs):
        if len(args) == 1:  # If only one positional argument, assume it's the value
            self.operator = eq  # default operator to 'equal'
            self.value = args[0]
        elif len(args) == 2:  # If two positional arguments, they should be (operator, value)
            self.operator, self.value = args
        else:
            self.operator = kwargs.get('operator', eq)
            try:
                self.value = kwargs.get('value')
            except KeyError:
                raise KeyError("Value not specified for Condition.")

@dataclass
class Dependent(Annotation[Any]):
    on: str
    when: Union[Condition, List[Condition]] = field(default_factory=list)  # OR, not AND

    def __post_init__(self):
        if not isinstance(self.when, list):
            self.when = [self.when]  # If 'when' is a singleton, wrap it in a list
        else:
            if len(self.when) == 0:
                self.when = [Condition(operator=ne, value=None)]

        self.when = [
            w if isinstance(w, Condition) else
            Condition(*(w if isinstance(w, tuple) or isinstance(w, list) else (w,)))
            for w in self.when
        ]

    def serialize(self):
        return {
            "dependent": {
                "on": self.on,
                "when": [asdict(w) for w in self.when]
            }
        }

@dataclass
class Validator(Annotation[Any]):
    condition: Callable[[Any], bool]
    error_message: str

@dataclass
class Slider(Annotation[int, float]):
    min: Union[int, float]
    max: Union[int, float]
    step: Union[int, float] = 1.0
    round: bool = False

    def serialize(self):
        return { **super().serialize(), "display": type(self).__name__.lower() }

@dataclass
class Numerical(Slider):
    randomizable: bool = False

@dataclass
class Text(Annotation[str]):
    multiline: bool = False
    dynamic_prompts: bool = False

@dataclass
class EqualLength(Annotation[List]):
    to: str