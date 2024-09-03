from enum import Enum
from inspect import signature
from functools import partial
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Dict, Generic, Optional, Literal, List, Tuple, Union, get_type_hints

# from torch import Tensor
# from torch.nn import Module

# from PIL import Image as ImageSource

# from sdbx.sd import CLIP as CLIPSource, VAE as VAESource
from sdbx.nodes.helpers import rename_class


## Path decorator ##
def node(fn=None, path=None, name=None):
    from sdbx.nodes.info import NodeInfo  # Avoid circular import

    if fn is None:
        return partial(node, path=path, name=name)

    fn.info = NodeInfo(fn, path=path, name=name)
    return fn


## Types ##

# Annotations
from typing import Annotated as A                   # A

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
@dataclass
class Name:
    name: str = ""

@dataclass
class Slider:
    min: Union[int, float]
    max: Union[int, float]
    step: Union[int, float] = 1.0
    round: bool = False

@dataclass
class Numerical(Slider):
    randomizable: bool = False # seed gen

@dataclass
class Text:
    multiline: bool = False
    dynamic_prompts: bool = False

@dataclass
class Dependent:
    on: str
    when: Any

@dataclass
class Validator:
    condition: Callable[[Any], bool]
    error_message: str


## Constants ##
MAX_RESOLUTION = 16384