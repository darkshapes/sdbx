# ### <!-- // /*  d a r k s h a p e s */ -->

import re
import contextlib
from pathlib import Path
from types import FunctionType, NoneType
from typing import Dict, List, Optional, Union, Tuple, Literal, Any, Coroutine
from PIL.Image import Image
import sounddevice as sd

# from nnll.metadata.helpers import make_callable
from nnll.mir.json_cache import MIR_PATH_NAMED, TEMPLATE_PATH_NAMED, JSONCache
from nnll.monitor.file import dbuq

# from zodiac.providers.constants import PkgType
from sdbx.nodes.signatures import qa_program

# from sdbx.nodes.generate import terminal_gen  # , autotype_methods
from sdbx.nodes.types import A, Slider, Text, node, I, Name, Numerical

SIGNATURE_TYPES = [
    qa_program,
]
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


# @MIR_DATA.decorator
# def _read_data(data: Optional[Dict[str, str]] = None):
#     return data


# class NodeArray:
#     """"""


# node_array = NodeArray()

# field_book = []
# TEMPLATE_DATA._load_cache()
# template_data = TEMPLATE_DATA._cache
# mir_data = _read_data()
# for series, compatibility in mir_data.items():
#     arch_name = series.split(".")[1:]
#     for comp_name, comp_data in compatibility.items():
#         if any(arch in arch_name for arch in template_data["arch"]["diffuser"]) and comp_data.get("pkg", 0) and comp_data["pkg"]["0"].get("diffusers"):
#             class_name = comp_data["pkg"]["0"]["diffusers"]
#             if "Pipeline" in class_name:
#                 make_callable(class_name, PkgType.DIFFUSERS.value[1].lower())
#                 dbuq(class_name)
#                 node_frame = terminal_gen(class_name, PkgType.DIFFUSERS)
#                 if node_frame is not None:
#                     func_name = node_frame["func_name"]
#                     try:
#                         tray_path = "frameworks/diffusers"
#                         tray_path += "/" if node_frame["aux_classes"] else "/pipelines"

#                         local_name_server = serialize_function(name=func_name, args_def=node_frame["generation_args"], body="    return 12")
#                         static_node = node(name=node_frame["node_name"], path=tray_path)(local_name_server[func_name])
#                         locals()[func_name] = static_node

#                     except ModuleNotFoundError:
#                         pass
#         elif any(arch in arch_name for arch in template_data["arch"]["transformer"]) and comp_data.get("pkg", 0) and comp_data["pkg"]["0"].get("transformers"):
#             class_name = comp_data["pkg"]["0"]["transformers"]
#             if "Model" in class_name:
#                 dbuq(class_name)
#                 node_frame = terminal_gen(class_name, PkgType.TRANSFORMERS)
#                 func_name = node_frame["func_name"]
#                 try:
#                     local_name_server = serialize_function(name=func_name, args_def=node_frame["generation_args"], body="    return 12")
#                     static_node = node(name=node_frame["node_name"], path="frameworks/transformers")(local_name_server[func_name])
#                     locals()[func_name] = static_node

#                 except ModuleNotFoundError:
#                     pass

#                     # field = {node_frame["node_name"]: (node, ...), "default": static_node}
#                     # field_book.append(create_model(class_name + "ElementBase", func_name=(node, static_node)))
#                     # setattr(node_array, func_name, static_node)

#                     # field_book.append(
#                     #     create_model(
#                     #         class_name + "ElementBase",
#                     #         func_name=(node, static_node),
#                     #     ),
#                     # )
#                     # setattr(node_array, func_name, static_node)


# load prompt save generate
@node(path="load", name="Load LLM")
def load_llm(
    model: Union[Path, str],
    api_url: A[str, Text(multiline=False)],
    api_base: A[str, Text(multiline=False)],
    api_key: A[str, Text(multiline=False)] = None,
    model_type: A[str, Text(multiline=False)] = None,
    max_tokens: A[int, Numerical(min=0, max=10e7)] = 4000,
    temperature: A[float, Slider(min=0, max=2.0)] = 0.0,
    cache: bool = False,
) -> A[contextlib.contextmanager, Name("MODEL_CONTEXT")]:
    import dspy

    api_kwargs = {"api_base": api_base}
    api_kwargs.setdefault("model_type", model_type) if model_type else ()
    api_kwargs.setdefault("model_type", api_key) if api_key else ()
    api_kwargs.setdefault("max_tokens", max_tokens) if max_tokens else ()

    return dspy.context(lm=dspy.LM(model=model, cache=cache, **api_kwargs))


@node(path="prompt", name="Text Prompt")
def text_prompt(
    text: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
) -> A[str, Name("TEXT_PROMPT")]:
    return text


@node(path="prompt", name="Image Prompt")
def image_prompt(
    image: Optional[Image] = None,
) -> A[Image, Name("IMAGE")]:
    return image


@node(path="generate", name="Generate Text")
async def generate_text(
    model_context: Tuple[contextlib.contextmanager],
    prompts: str,
    voice_device: Optional[sd.DeviceList] = None,
    signature: A[str, Literal[next(iter([SIGNATURE_TYPES]), "")]] = next(iter(SIGNATURE_TYPES)),
) -> A[Any, Name("GENERATOR")]:
    with model_context:
        async for prediction in qa_program(question=text_prompt):
            yield prediction
    # self.status.value = "Complete"


@node(path="save", display=True, name="Display Text")
async def display_text(
    generator: Coroutine,
) -> I[str]:
    from dspy import Prediction
    from dspy.streaming import StreamResponse, StatusMessage

    if isinstance(generator, StreamResponse):
        yield generator.chunk
    elif isinstance(generator, Prediction):
        yield "prediction :", generator.answer
    elif isinstance(generator, StatusMessage):
        yield generator.message


@node(path="")
async def record_audio(self, frequency: int = 16000) -> None:
    """Get audio from mic"""
    self.frequency = frequency
    self.audio_stream = [0]
    self.audio_stream = sd.rec(int(self.precision), samplerate=self.frequency, channels=1)
    sd.wait()
    self.sample_length = str(float(len(self.audio_stream) / frequency))
    return self.audio_stream


async def play_audio(self) -> None:
    """Playback audio recordings"""
    try:
        sd.play(self.audio_stream, samplerate=self.frequency)
        sd.wait()
    except TypeError as error_log:
        dbuq(error_log)


async def erase_audio(self) -> None:
    """Clear audio graph and recording"""
    self.audio_stream = [0]
    self.frequency = 0.0
    self.sample_length = 0.0
