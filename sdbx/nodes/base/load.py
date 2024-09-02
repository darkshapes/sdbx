from sdbx import config
from sdbx.nodes.types import *
from sdbx.nodes.helpers import softRandom, hardRandom

import os

from llama_cpp import Llama

@node(name="GGUF Loader")
def gguf_loader(
    checkpoint: A[str, os.path.join(os.path.dirname(os.path.abspath("models.")), "llms")] = os.path.join(os.path.dirname(os.path.abspath("models.")), "llms"),
    # path: A[str, Text(multiline=False, dynamic_prompts=True)] = "codeninja-1.0-openchat-7b.Q5_K_M.gguf",
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
    return Llama(
        model_path=os.path.join(config.get_path("models.llm"), path),
        n_gpu_layers=gpu_layers,
        n_threads=threads,
        seed=softRandom() if one_time_seed == False else hardRandom(),
        n_ctx=max_context,
        n_batch=batch,
        flash_attn=flash_attention,
        verbose=verbose
    )