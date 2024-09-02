from sdbx import config
from sdbx.nodes.types import *
from sdbx.nodes.helpers import softRandom, hardRandom
from sdbx.config import get_config_location
from sdbx.nodes.helpers import getDirFiles
import torch
# from PIL import Image
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler

import os

from llama_cpp import Llama

@node(name="GGUF Loader")
def gguf_loader(
    checkpoint: Literal[*getDirFiles("models.llms")] = Literal[*getDirFiles("models.llms")],
    gpu_layers: A[int, Slider(min=-1, max=35, step=1)] = 0,
    threads: A[int, Slider(min=0, max=64, step=1)] = 8,
    max_context: A[int, Slider(min=0, max=32767, step=64)] = 2048,
    advanced_options: bool = False,
    one_time_seed: A[bool, Dependent(on="advanced_options", when=True)] = False,
    flash_attention: A[bool, Dependent(on="advanced_options", when=True)] = False,
    verbose: A[bool, Dependent(on="advanced_options", when=True)] = False,
    batch: A[int, Numerical(min=0, max=512, step=1), Dependent(on="advanced_options", when=True)] = 1,  
) -> Llama:
    print("⎆loading:GGUF")
    # debug print(os.path.join(config.get_path("models.llms"), checkpoint))
    return Llama(
        model_path=os.path.join(config.get_path("models.llms"), checkpoint),
        n_gpu_layers=gpu_layers,
        n_threads=threads,
        seed=softRandom() if one_time_seed == False else hardRandom(),
        n_ctx=max_context,
        n_batch=batch,
        flash_attn=flash_attention,
        verbose=verbose
    )

@node(name="Safetensors Loader")
def safetensors_loader(
    checkpoint: Literal[*getDirFiles("models.checkpoints")] = Literal[*getDirFiles("models.checkpoints")],
) -> Llama:
    print("⎆loading:Safetensors")
    return os.path.join(config.get_path("models.checkpoints"), checkpoint)