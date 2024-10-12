import os
import PIL
from PIL import Image
from llama_cpp import Llama
from transformers import AutoModel, TensorType
from sdbx import config
import sdbx.config
from sdbx.nodes.types import *
from sdbx.nodes.helpers import soft_random, hard_random
from sdbx.nodes.compute import T2IPipe
from sdbx.nodes.tuner import NodeTuner
from sdbx.indexer import IndexManager

system = config.get_default("spec","data") #needs to be set by system @ launch
spec = system["devices"]
flash_attn = system["flash_attention"]
algorithms = config.get_default("algorithms","schedulers")
algorithms = config.get_default("algorithms","solvers")

llms = config.get_default("index","LLM")
diffusion_models = config.get_default("index","DIF")
lora_models = config.get_default("index","LOR")
vae_models = config.get_default("index","VAE")
transformers = config.get_default("index","TRA")
primary_models = {**llms, **diffusion_models}


import os
from llama_cpp import Llama

debug = True # eeeee

@node(name="GGUF Loader")
def gguf_loader(
     model: Literal[*llms.keys()] =  next(iter([*llms.keys()]),""),
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
    if debug == True: print("loading:GGUF")
    return Llama(
        model_path=os.path.join(config.get_path("models.llms"), model),
        seed=soft_random() if one_time_seed == False else hard_random(),
        n_gpu_layers=gpu_layers if cpu_only == False else 0,
        n_threads=threads,
        n_ctx=max_context,
        n_batch=batch,
        flash_attn=flash_attention,
        verbose=verbose
    )


@node(name="LLM Prompt")
def llm_prompt(
    llama: Llama,
    system_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "You're a guru for revealing what you know, yet wiser for revealing what you do not.",
    user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    streaming: bool = True, #triggers generator in next node? 
    advanced_options: bool = False,
        top_k: A[int, Dependent(on="advanced_options", when=True),Slider(min=0, max=100)] = 40,
        top_p: A[float, Dependent(on="advanced_options", when=True), Slider(min=0, max=1, step=0.01, round=0.01)] = 0.95,
        repeat_penalty: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 1,
        temperature: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 0.2,
        max_tokens:  A[int, Dependent(on="advanced_options", when=True),  Numerical(min=0, max=2)] = 256,
) -> str:
    print("Encoding Prompt")
    return llama.create_chat_completion(
        messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
        stream=streaming,
        repeat_penalty=repeat_penalty,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
    )

@node(name="LLM Print")
def llm_print(
    response: A[str, Text()]
) -> I[str]:
    print("Calculating Resposnse")
    for chunk in range(response):
        delta = chunk['choices'][0]['delta']
        # if 'role' in delta:               # this prints assistant: user: etc
            # print(delta['role'], end=': ')
            #yield (delta['role'], ': ')
        if 'content' in delta:              # the response itself
            print(delta['content'], end='')
            yield delta['content'], ''