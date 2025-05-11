from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Literal, List, Tuple, Union, get_type_hints

from sdbx.nodes.types.annotations import *
from sdbx.nodes.types.decorator import node

# Annotations
from typing import Annotated as A  # A

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

from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Literal, List, Tuple, Union, get_type_hints

from sdbx.nodes.types.annotations import *
from sdbx.nodes.types.decorator import node

# Annotations
from typing import Annotated as A  # A

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
