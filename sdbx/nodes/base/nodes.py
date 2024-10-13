import os
import PIL
from PIL import Image
from sdbx.config import config
from sdbx.nodes.types import *
from sdbx.nodes.helpers import soft_random, hard_random

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
optimize = config.node_tuner
insta = config.t2i_pipe

import os
from llama_cpp import Llama

@node(name="Genesis Node", display=True)
def genesis_node(
     user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
     model: Literal[*primary_models.keys()] = next(iter([*primary_models.keys()]),""),
 ) -> Tuple(data, data, data, data, data):
    optimize.determine_tuning(model)
    opt_exp = optimize.opt_exp() 
    gen_exp = optimize.gen_exp(2)#clip skip
    cond_exp = optimize.cond_exp()
    pipe_exp = optimize.pipe_exp()
    vae_exp = optimize.vae_exp()   
    return opt_exp, gen_exp, cond_exp, pipe_exp, vae_exp

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

from transformers import data, TensorType

@node(name="Image Prompt", display=True)
def image_prompt(

    prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "A slice of a rich and delicious chocolate cake presented on a table in a palace reminiscent of Versailles",
    seed: A[int, Numerical(min=0, max=0xFFFFFFFFFFFFFF, step=1, randomizable=True)] = soft_random(),
    # type: ignore
) -> str:
    return insta.queue_manager(prompt,seed)

@node(name="Load Vision Model", display=True)
def load_vision_model(
    transformer: Literal[*llms.keys()] = next(iter([*llms.keys()]),""),
    batch: A[int, Numerical(min=0, max=512, step=1)] = 1,
    device: Literal[*spec] = next(iter(*spec), "cpu"),
    cpu_only: bool = (True if next(iter(*system), "cpu") == "cpu" else False), 
    gpu_layers: A[int, Dependent(on="cpu_only", when=False), Slider(min=-1, max=35, step=1)] = -1,
    flash_attention: bool = flash_attn, #autodetect
    threads: A[int, Slider(min=0, max=64, step=1)] = 8,
    max_context: A[int, Slider(min=0, max=32767, step=64),] = None, #let ollama do its thing
    verbose: bool = False,
    one_time_seed: bool = False,
) -> TensorType:
    Tensor = insta.declare_encoders(gen_exp)
    return Tensor

@node(name="Load LoRA", display=True)
def load_lora(
    lora: Literal[*lora_models.keys()] = next(iter([*lora_models.keys()]),""),
    lora_scale: A[float, Numerical(min=0.0, max=1.0, step=0.01)] = 1.0,
    fuse_lora: bool = True,
) -> TensorType:
    return insta.add_lora(lora, fuse_lora ,lora_scale)

@node(name="Device", display=True)
def device_name(
    device: A[str, Literal[*system]] = next(iter([*system]), "cpu"),
) -> str:
    return device

@node(name="Load Diffusion Model", display=True)
def load_diffusion_model(
    model: Literal[*diffusion_models.keys()],
    vae: TensorType,
    device: str,
    precision: str, = pipe_exp["variant"],
) ->  TensorType:
    exp = precision
    pipe = AutoPipelineForText2Image.from_pretrained(model, vae=vae, **exp).to(device)
    return pipe

@node(name="Load Vae Model", display=True)
def load_vae_model(
    model: Literal[*vae_models.keys()] =  next(iter([*vae_models.keys()]),""),
    device: Literal[*spec] = next(iter(*spec), "cpu"),
    vae_slice: bool = False,
    vae_tile: bool = True,
    upcast: bool = False
) -> TensorType:
    vae_model = insta.add_vae(model)
    return vae_model

insta.construct_pipe(pipe_exp, vae_exp)

@node(name="Encode Prompt", display=True)
def encode_prompt(
    text_encoder: TensorType,
    text_encoder_2: TensorType = None,
    text_encoder_3: TensorType = None,
    lora: TensorType = None,
    gpu_id: A[int, Dependent(on="device", when=(not "cpu")), Slider(min=0, max=100)] = 0,
) -> TensorType:
    return insta.encode_prompt(cond_exp)


insta.diffuse_latent(gen_exp)