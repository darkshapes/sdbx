from sdbx.nodes.types import *
from sdbx.nodes.helpers import softRandom

from llama_cpp import Llama

@node
def llm_loader(
    threads: A[int, Slider(min=0, max=10, step=1)] = 8,
    max_context: A[int, Numerical(min=0, max=32767, step=64)] = 2048,
    repeat_penalty: A[float, Numerical(min=0, max=2, step=0.01)] = 1,
    temperature: A[float, Numerical(min=0, max=2, step=0.01)] = 0.2
) -> str:
    return Llama(
        model_path="/Users/Shared/ouerve/recent/darkshapes/models/llms/codeninja-1.0-openchat-7b.Q5_K_M.gguf",
        # n_gpu_layers=-1, # Uncomment to use GPU acceleration
        n_threads=threads,   # The number of CPU threads to use, tailor to your system and the resulting performance
        seed=softRandom(), # Uncomment to set a specific seed
        n_ctx=max_context, # Uncomment to increase the context window
        chat_format="openchat",
        repeat_penalty=repeat_penalty,
        temperature=temperature,
    )