# ### <!-- // /*  d a r k s h a p e s */ -->

import contextlib
from decimal import Decimal
from typing import Any, AsyncGenerator, Coroutine, Generator, Literal, Optional, Union

import dspy
import sounddevice as sd
from PIL.Image import Image
from zodiac.providers.registry_entry import RegistryEntry
from zodiac.toga import app

from sdbx.nodes.types import A, Dependent, I, Name, Numerical, Slider, Text, node

# from sdbx import logger
# from sdbx.server.types import Graph


@node(path="load", name="Load LLM")
async def load_llm(
    model: RegistryEntry,
    cache: bool = True,
    max_tokens: A[int, Numerical(min=0, max=0xFFFFFFFFFFFFFFFF, step=16)] = 4096,
    max_workers: A[int, Numerical(min=0, max=16)] = 4,
    temperature: A[float, Slider(min=0, max=2.0)] = 0.0,
    bypass_api_options: bool = False,
    api_base: A[
        str,
        Dependent(on="bypass_api_options", when=True),
        Text(
            multiline=False,
            dynamic_prompts=False,
        ),
    ] = None,
    api_key: A[
        str,
        Dependent(on="bypass_api_options", when=True),
        Text(
            multiline=False,
            dynamic_prompts=False,
        ),
    ] = None,
    model_type: A[Optional[str], Dependent(on="bypass_api_options", when=True), Literal["chat", "text", ""]] = "",
) -> A[dspy.LM, Name("MODEL")]:
    dspy.configure_cache(enable_disk_cache=cache)
    lm_kwargs = {"async_max_workers": max_workers, "cache": cache, "max_tokens": max_tokens}
    lm_kwargs.setdefault("temperature", temperature) if temperature else ()
    if not bypass_api_options:
        api_kwargs = model.api_kwargs
    else:
        api_kwargs.setdefault("model_type", model_type) if model_type else ()
        api_kwargs.setdefault("api_key", api_key) if api_key else ()
        api_kwargs.setdefault("api_base", api_base) if api_base else ()
    lm_model = dspy.LM(
        model.model,
        **api_kwargs,
        **lm_kwargs,
    )
    return lm_model


@node(path="generate", name="Add Streaming")
async def add_streaming(
    model: dspy.LM,
    dspy_stream: bool = True,
    async_stream: bool = True,
    stream_type: A[
        str,
        Literal[
            "auto",
            "dspy",
            "litellm",
        ],
    ] = "auto",
    include_final_output: bool = False,
    adapter: Optional[dspy.ChatAdapter] = None,
) -> A[contextlib.contextmanager, Name("CONTEXT")]:
    from zodiac.toga.signatures import StreamActivity

    context_kwargs = {"lm": model}
    if adapter:
        context_kwargs.setdefault("adapter", adapter)
    if stream_type == "dspy" or stream_type == "auto":
        stream_listeners = [dspy.streaming.StreamListener(signature_field_name="answer")]  # TODO: automate from dspy.Signature,
    predictor_kwargs = {
        "status_message_provider": StreamActivity(),  # feedback to zodiac interface
        "async_streaming": async_stream,
        "include_final_prediction_in_output_stream": include_final_output,
    }
    if dspy_stream:
        predictor_kwargs["stream_listeners"] = stream_listeners

    return context_kwargs, predictor_kwargs


@node(path="prompt", name="Text Prompt")
async def text_prompt(
    text: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
) -> A[str, Name("PROMPT")]:
    return text


@node(path="generate", name="stream_inference")
async def stream_inference(
    context: contextlib.contextmanager,
    audio_prompt: list = None,
    image_prompt: Image = None,
    text_prompt: str = None,
) -> A[Any, Name("GENERATOR")]:
    from dspy import context as dspy_context
    from dspy import streamify
    from zodiac.toga.signatures import Predictor

    prompts = dict()
    prompts.setdefault("text", text_prompt) if text_prompt else ()
    prompts.setdefault("audio", audio_prompt) if audio_prompt else ()
    prompts.setdefault("image", image_prompt) if image_prompt else ()
    context_kwargs, predictor_kwargs = context  # ,dspy_stream=False)
    with dspy_context(**context_kwargs):
        generator = streamify(Predictor(), **predictor_kwargs)
        return generator(question=prompts["text"])


@node(path="save", display=True, name="Display Text")
async def display_text(
    generator: Any = None,
) -> I[Any]:  # pyright:ignore[reportInvalidTypeForm]
    from dspy import Prediction
    from dspy.streaming import StatusMessage, StreamResponse
    from litellm.types.utils import ModelResponseStream  # StatusStreamingCallback

    if isinstance(generator, AsyncGenerator):
        async for prediction in generator:
            if isinstance(generator, ModelResponseStream) and prediction["choices"][0]["delta"]["content"]:
                yield prediction["choices"][0]["delta"]["content"]
            elif isinstance(generator, StreamResponse):
                yield str(generator.chunk)
            elif isinstance(generator, Prediction):
                yield str(generator.answer)
            # elif isinstance(generator, StatusMessage):
            # app.app.Interface.status_display.text = app.app.Interface.status_text_prefix + str(generator.message)

    if isinstance(generator, StreamResponse):
        yield generator.chunk
    elif isinstance(generator, Prediction):
        yield "prediction :", generator.answer
    elif isinstance(generator, StatusMessage):
        yield generator.message


@node(path="prompt", name="Audio Prompt")
async def audio_prompt(
    sample_rate: A[int, Numerical(min=8000, max=64000, step=8000)] = 16000,
    duration: A[float, Numerical(min=1.0, max=10.0)] = 3.0,
) -> A[tuple[list, Decimal], Name("PROMPT")]:
    """Get audio from mic"""

    precision = duration * sample_rate
    audio_stream = [0]
    audio_stream = sd.rec(int(precision), samplerate=sample_rate, channels=1)
    sd.wait()
    sample_length = Decimal(str(float(len(audio_stream) / sample_rate)))
    return audio_stream, sample_length


# async def play_audio(self) -> None:
#     """Playback audio recordings"""
#     try:
#         sd.play(self.audio_stream, samplerate=self.frequency)
#         sd.wait()
#     except TypeError as error_log:
#         dbuq(error_log)


# async def erase_audio(self) -> None:
#     """Clear audio graph and recording"""
#     self.audio_stream = [0]
#     self.frequency = 0.0
#     self.sample_length = 0.0

# @node(path="prompt", name="Image Prompt")
# def image_prompt(
#     image: Optional[Image] = None,
# ) -> A[Image, Name("IMAGE")]:
#     return image
