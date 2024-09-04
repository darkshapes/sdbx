from sdbx import config
from sdbx.nodes.types import *
from sdbx.nodes.helpers import softRandom, hardRandom
from sdbx.config import get_config_location
from sdbx.nodes.helpers import getDirFiles
import torch
# from PIL import Image
# from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
# from kolors.models.modeling_chatglm import ChatGLMModel
# from kolors.models.tokenization_chatglm import ChatGLMTokenizer

import os
from llama_cpp import Llama

@node(name="GGUF Loader")
def gguf_loader(
    checkpoint: Literal[*getDirFiles("models.llms", ".gguf")] = Literal[*getDirFiles("models.llms", ".gguf")],
    cpu_only: bool = True,
        gpu_layers: A[int, Dependent(on="cpu_only", when=False), Slider(min=-1, max=35, step=1)] = -1,
    advanced_options: bool = False,
        threads: A[int, Dependent(on="advanced_options", when=True), Slider(min=0, max=64, step=1)] = 8,
        max_context: A[int,Dependent(on="advanced_options", when=True), Slider(min=0, max=32767, step=64)] = 0,
        one_time_seed: A[bool, Dependent(on="advanced_options", when=True)] = False,
        flash_attention: A[bool, Dependent(on="advanced_options", when=True)] = False,
        verbose: A[bool, Dependent(on="advanced_options", when=True)] = False,
        batch: A[int, Dependent(on="advanced_options", when=True), Numerical(min=0, max=512, step=1), ] = 1,  
) -> Llama:
    print("loading:GGUF")
    return Llama(
        model_path=os.path.join(config.get_path("models.llms"), checkpoint),
        seed=softRandom() if one_time_seed == False else hardRandom(),
        n_gpu_layers=gpu_layers if cpu_only == False else 0,
        n_threads=threads,
        n_ctx=max_context,
        n_batch=batch,
        flash_attn=flash_attention,
        verbose=verbose
    )

@node(name="Safetensors Loader")
def safetensors_loader(
    checkpoint: Literal[*getDirFiles("models.checkpoints", ".safetensors")] = Literal[*getDirFiles("models.checkpoints", ".safetensors")],
) -> Llama:
    print("loading:Safetensors")
    return os.path.join(config.get_path("models.checkpoints"), checkpoint)