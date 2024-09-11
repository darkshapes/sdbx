import gc
import os
from time import perf_counter
import datetime
import time
from diffusers import AutoPipelineForText2Image, AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, FromOriginalModelMixin
from diffusers.schedulers import AysSchedules
from sdbx.sdbx.nodes.computations import Text2ImageGenerator, llm_request, timecode as tc
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from sdbx.nodes.types import node, Literal, Text, Slider, Any, I, Numerical, Annotated as A
import sdbx.config
from sdbx.config import config
from sdbx.nodes.helpers import seed_planter, soft_random
from sdbx.nodes.types import *
from llama_cpp import Llama
import sdbx.sdbx.nodes.computations
import safetensors

def tc(): print(str(datetime.timedelta(seconds=time.process_time())), end="")

@node(name="Diffusion Prompt", display=True)
def diffusion_prompt(
    text_encoder: torch.Tensor,
    text_encoder_2: torch.Tensor,
    text_encoder_gguf: Llama,
    text_encoder_2_gguf: Llama,
    prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "A rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles",
    seed: A[int, Numerical(min=0, max=0xFFFFFFFFFFFFFF, step=1,randomizable=True)]= int(soft_random()),
    override_device: Literal[*get_gpus()] = get_gpus()[0], # type: ignore
) -> Tuple[torch.Tensor, torch.Tensor]:
    queue = push_prompt(prompt, seed)
    embeddings = start_encoding(queue, model, text_encoder, text_encoder_2)
    cleanup()
    return embeddings


@node(name="Safetensors Loader")
def safetensors_loader(
    file: Literal[config.get_path_contents("models.diffusers", extension="safetensors")] = "stabilityai/stable-diffusion-xl-base-1.0",
    model_type: Literal["diffusion", "autoencoder" ,"super_resolution", "token_encoder"] = "diffusion",
        # safety: A[bool, Dependent(on="model_type", when="diffusion")] = False,
    advanced_options: bool = False,    
        override_device: (Literal[*get_device(), "cpu"], Dependent(on="advanced_options", when=True)) = "", # type: ignore
        precision: A[int, Dependent(on="device", when=(not "cpu")), Slider(min=16, max=64, step=16)] = 16, 
        bfloat: A[bool, Dependent(on="precision", when="16")] = False,
) -> Tuple[*TensorData]:
    tc()
    if model_type=="diffusion":
        processor = load_pipeline(file, precision, bfloat) 
    elif model_type=="autoencoder":
        processor = load_autoencoder(file)
    elif model_type=="token_encoder":
        processor = load_token_encoder(file)
    tc()
    return processor

@node(name="Diffusion", display=True)
def diffusion(
    model: TensorData,
    queue: TensorData,
    # lora needs to be explicitly declared
    lora: Literal[config.get_path_contents(
        "models.loras", extension="safetensors") or pcm_default] = "pcm_sdxl_normalcfg_8step_converted_fp16.safetensors",
   scheduler: Literal[config.get_default(
        "algorithms", "schedulers") or "EulerDiscreteScheduler"] = "EulerDiscreteScheduler",
    # only appears if Lora isnt a PCM/LCM and scheduler isnt "AysScheduler". lower to increase speed
    inference_steps: A[int, Numerical(min=0, max=250, step=1)] = 8,
    # default for sdxl-architecture. raise for sd-architecture, drop to 0 (off) to increase speed with turbo, etc. auto mode?
    guidance_scale: A[float, Numerical(
        min=0.00, max=50.00, step=.01, round=".01")] = 5,
    # lora2 = next(iter(m for m in os.listdir(config.get_path("models.loras")) if "adfdfd" in m and m.endswith(".safetensors")), "a default lora.safetensors")
    # text_inversion = next((w for w in get_dir_files("models.embeddings"," ")),"None")
    advanced_options: bool = False,
    dynamic_guidance: A[bool, Dependent(on="advanced_options", when=True)] = False,
) -> latents :
    latent = run_inference(queue, steps, guidance_scale, dynamic_guidance, scheduler, lora)
    cleanup()
    return latent

@node(name="Autodecode", display=True)
def autodecode(
    # USER OPTIONS  : VAE/SAVE/PREVIEW
    latent: Any,
    vae: Literal[config.get_path_contents(
        "models.vae", extension="safetensors")] = None,
    file_prefix: A[str, Text(multiline=False, dynamic_prompts=True)] = "Shadowbox-",
    # optional png compression,
    compress_level: A[int, Slider(min=1, max=9, step=1)] = 4
) -> I[Image]:
    batch = autodecode(self, latent, file_prefix)
    for image in range(batch):
        yield image

@node(name="GGUF Loader", display=True)
def gguf_loader(
    checkpoint: Literal[*get_dir_files("models.llms", ".gguf")] = get_dir_files("models.llms", ".gguf")[0], # type: ignore
    advanced_options: bool = False,
        cpu_only: A[bool, Dependent(on="advanced_options", when=True)] = True,
        gpu_layers: A[int, Dependent(on="cpu_only", when=False), Slider(min=-1, max=35, step=1)] = -1,
        threads: A[int, Dependent(on="advanced_options", when=True), Slider(min=0, max=64, step=1)] = 8,
        max_context: A[int,Dependent(on="advanced_options", when=True), Slider(min=0, max=32767, step=64)] = 8192,
        one_time_seed: A[bool, Dependent(on="advanced_options", when=True)] = False,
        flash_attention: A[bool, Dependent(on="advanced_options", when=True)] = False,
        verbose: A[bool, Dependent(on="advanced_options", when=True)] = False,
        batch: A[int, Dependent(on="advanced_options", when=True), Numerical(min=0, max=512, step=1), ] = 1,  
) -> TensorData:
    tc()
    llama = llm_load(checkpoint,threads,max_context,verbose)
    return llama
    
@node(name="LLM Prompt", display=True)
def llm_prompt(
    llama: TensorData,
    system_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "You're a guru for revealing what you know, yet wiser for revealing what you do not.",
    user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    advanced_options: bool = False,
        top_k: A[int, Dependent(on="advanced_options", when=True),Slider(min=0, max=100)] = 40,
        top_p: A[float, Dependent(on="advanced_options", when=True), Slider(min=0, max=1, step=0.01, round=0.01)] = 0.95,
        repeat_penalty: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 1,
        temperature: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 0.2,
        max_tokens:  A[int, Dependent(on="advanced_options", when=True),  Numerical(min=0, max=2)] = 256,
        streaming: bool = True, #triggers generator in next node? 
) -> str:
    tc()
    llm_request(system_prompt, user_prompt,)
    tc()
    return llama

@node(name="LLM Print", display=True)
def llm_print(
    response: A[str, Text()]
) -> I[str]:
    tc()
    print("Calculating Resposnse")
    for chunk in range(response):
        delta = chunk['choices'][0]['delta']
        # if 'role' in delta:               # this prints assistant: user: etc
            # print(delta['role'], end=': ')
            #yield (delta['role'], ': ')
        if 'content' in delta:              # the response itself
            print(delta['content'], end='')
            yield delta['content'], ''

