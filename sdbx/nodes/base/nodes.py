import time
import datetime

from llama_cpp import Llama

import sdbx.nodes.computations

from sdbx import config
from sdbx.nodes.types import *
from sdbx.nodes.helpers import seed_planter, soft_random
from sdbx.nodes.compute import Inference, get_device

# from time import perf_counter diagnostics

# AUTOCONFIGURE OPTIONS : this should autodetec
token_encoder_default = "stabilityai/stable-diffusion-xl-base-1.0"
lora_default = "pcm_sdxl_normalcfg_8step_converted_fp16.safetensors"
pcm_default = "Kijai/converted_pcm_loras_fp16/tree/main/sdxl/"
vae_default = "madebyollin/sdxl-vae-fp16-fix.safetensors"
model_default = "ponyFaetality_v11.safetensors"
llm_default = "codeninja-1.0-openchat-7b.Q5_K_M.gguf"
scheduler_default = "EulerAncestralDiscreteScheduler"


def tc():
    print(str(datetime.timedelta(seconds=time.process_time())), end="")


@node(name="Text Prompt", display=True)
def llm_prompt(
    llama: Llama,
    system_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "You're a guru for knowing what you know, yet wiser for knowing what you do not.",
    user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    advanced_options: bool = False,
    top_k: A[int, Dependent(on="advanced_options", when=True), Slider(min=0, max=100)] = 40,
    top_p: A[float,Dependent(on="advanced_options", when=True),Slider(min=0, max=1, step=0.01, round=0.01)] = 0.95,
    repeat_penalty: A[float,Dependent(on="advanced_options", when=True),Numerical(min=0.0, max=2.0, step=0.01, round=0.01),] = 1,
    temperature: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01),] = 0.2,
    max_tokens: A[int, Dependent(on="advanced_options", when=True), Numerical(min=0, max=2)] = 256,
    streaming: bool = True,  # triggers generator in next node?
) -> str:
    tc()
    Inference.llm_request(system_prompt, user_prompt, streaming=True)
    tc()
    return llama


@node(name="Image Prompt", display=True)
def image_prompt(
    text_encoder: Llama,
    text_encoder_2: Llama = None,
    text_encoder_gguf: Llama = None,
    text_encoder_2_gguf: Llama = None,
    prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "A rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles",
    seed: A[int, Numerical(min=0, max=0xFFFFFFFFFFFFFF, step=1, randomizable=True)] = int(soft_random()),
    # type: ignore
    override_device: Literal[*next(iter(get_device()), "cpu")] = "",
) -> Tuple[Llama, Llama]:
    queue = Inference.push_prompt(prompt, seed)
    embeddings = Inference.start_encoding(queue, text_encoder, text_encoder_2)
    return embeddings


@node(name="Load Model", display=True)
def Load(
    model: Literal[*config.get_path_contents("models.llms", extension="safetnesors", base_name=True)] = None,
    # safety: A[bool, Dependent(on="model_type", when="diffusion")] = False,
    advanced_options: bool = False,
    override_device: A[Literal[*get_device(), "cpu"], Dependent(on="advanced_options", when=True)] = "",  # type: ignore
    precision: A[int, Dependent(on="device", when=(not "cpu")), Slider(min=16, max=64, step=16)] = 16,
    bfloat: A[bool, Dependent(on="precision", when="16")] = False,
    gpu_layers: A[int, Dependent(on="cpu_only", when=False), Slider(min=-1, max=35, step=1)] = -1,
    threads: A[int, Dependent(on="advanced_options", when=True), Slider(min=0, max=64, step=1)] = 8,
    one_time_seed: A[bool, Dependent(on="advanced_options", when=True)] = False,
    flash_attention: A[bool, Dependent(on="advanced_options", when=True)] = False,
    verbose: A[bool, Dependent(on="advanced_options", when=True)] = False,
    max_context: A[int,Dependent(on="advanced_options", when=True),Slider(min=0, max=32767, step=64),] = 8192,
) -> Tuple[Llama]:
    tc()
    metadata = ReadMeta(model).data
    model_type = EvalMeta(metadata).data
    if "LLM" in model_type:
        llama = Inference.gguf_load(model, threads, max_context, verbose)
    else:
        if "VAE-" in model_type:
            processor = Inference.load_vae(model)
        if "CLI-" in model_type:
            processor = Inference.load_token_encoder(model)
        if "LORA-" in model_type:
            processor = Inference.load_pipeline(model, precision, bfloat)   
        if "LLM-" in model_type:
            processor = Inference.load_llm(model) 

    else:




    tc()
    return [*processor]


@node(name="GGUF Loader", display=True)
def gguf_loader(
    gguf: Literal[
        *config.get_path_contents("models.llms", extension="gguf", base_name=True)] = llm_default,  # type: ignore
    advanced_options: bool = False,
    cpu_only: A[bool, Dependent(on="advanced_options", when=True)] = True,
    gpu_layers: A[int, 
        Dependent(on="cpu_only", when=False), 
        Slider(min=-1, max=35, step=1)] = -1,
    threads: A[int, 
               Dependent(on="advanced_options", when=True), Slider(min=0, max=64, step=1)] = 8,

    one_time_seed: A[bool, Dependent(on="advanced_options", when=True)] = False,
    flash_attention: A[bool, Dependent(on="advanced_options", when=True)] = False,
    verbose: A[bool, Dependent(on="advanced_options", when=True)] = False,
    batch: A[int,
        Dependent(on="advanced_options", when=True),
        Numerical(min=0, max=512, step=1),
    ] = 1,
) -> Llama:
    tc()
    llama = Inference.gguf_load(gguf, threads, max_context, verbose)
    return llama


@node(name="Generate", path=None, display=True)
def diffusion(
    model: Llama,
    queue: Llama,
    # lora needs to be explicitly declared
    lora: Literal[
        *config.get_path_contents("models.loras", extension="safetensors", base_name=True)] = pcm_default,
    scheduler: Literal[
        *config.get_default("algorithms", "schedulers")] = scheduler_default,
    # only appears if Lora isnt a PCM/LCM and scheduler isnt "AysScheduler". lower to increase speed
    inference_steps: A[int, Numerical(min=0, max=250, step=1)] = 8,
    # default for sdxl-architecture. raise for sd-architecture, drop to 0 (off) to increase speed with turbo, etc. auto mode?
    guidance_scale: A[
        float, Numerical(min=0.00, max=50.00, step=0.01, round=".01")
    ] = 5,
    # lora2 = next(iter(m for m in os.listdir(config.get_path("models.loras")) if "adfdfd" in m and m.endswith(".safetensors")), "a default lora.safetensors")
    # text_inversion = next((w for w in get_dir_files("models.embeddings"," ")),"None")
    advanced_options: bool = False,
    dynamic_guidance: A[bool, Dependent(on="advanced_options", when=True)] = False,
    override_device: A[
        Literal[*get_device(), "cpu"], Dependent(on="advanced_options", when=True)
    ] = "",  # type: ignore
) -> Llama:
    latent = Inference.run_inference(
        queue,
        inference_steps,
        guidance_scale,
        dynamic_guidance,
        scheduler,
        lora=lora_default,
    )
    Inference.clear_memory_cache()
    return latent


@node(name="Show & Save Image", display=True)
def autodecode(
    # USER OPTIONS  : VAE/SAVE/PREVIEW
    latent: Llama,
    vae: Literal[
        *config.get_path_contents("models.vae", extension="safetensors", base_name=True)] = vae_default,
    file_prefix: A[str, Text(multiline=False, dynamic_prompts=True)] = "Shadowbox-",
    # advanced_options: bool = False,
    # temp: bool = False,
    # format: A[Literal, Dependent(on:"temp", when="False"), "png","jpg","optimize"]] = "optimize",
    # compress_level: A[int, Slider(min=1, max=9, step=1),  Dependent(on:"format", when=(not "optimize"))] = 7,
) -> I[Any]:
    # tempformat="optimize", compress_level="7"
    batch = autodecode(vae, latent, file_prefix)
    for image in range(batch):
        yield image


@node(name="LLM Print", display=True)
def llm_print(response: A[str, Text()]) -> I[str]:
    tc()
    print("Calculating Resposnse")
    for chunk in range(response):
        delta = chunk["choices"][0]["delta"]
        # if 'role' in delta:               # this prints assistant: user: etc
        # print(delta['role'], end=': ')
        # yield (delta['role'], ': ')
        if "content" in delta:  # the response itself
            print(delta["content"], end="")
            yield delta["content"], ""
