from sdbx.config import config
from sdbx.nodes.types import *
from sdbx.nodes.helpers import soft_random, hard_random, get_gpus
from sdbx.nodes.helpers import get_dir_files
from diffusers import AutoPipelineForText2Image, AutoencoderKL
import torch

from transformers import CLIPTokenizer, CLIPTextModel
import os
from llama_cpp import Llama

@node(name="GGUF Loader")
def gguf_loader(
    checkpoint: Literal[*get_dir_files("models.llms", ".gguf")] = next(iter(get_dir_files("models.llms", ".gguf")), None),
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
    print(f"loading:GGUF{os.path.join(config.get_path('models.llms'), checkpoint)}")
    return Llama(
        model_path=os.path.join(config.get_path("models.llms"), "codeninja-1.0-openchat-7b.Q5_K_M.gguf"),
        seed=soft_random(), #if one_time_seed == False else hard_random(),
        #n_gpu_layers=gpu_layers if cpu_only == False else 0,
        n_threads=threads,
        n_ctx=max_context,
        #n_batch=batch,
        #flash_attn=flash_attention,
        verbose=verbose,
    )

@node(name="Safetensors Loader")
def safetensors_loader(
    checkpoint: Literal[*get_dir_files("models.checkpoints", ".safetensors")] = next(iter(get_dir_files("models.checkpoints", ".safetensors")), None),
    model_type: Literal["diffusion", "autoencoder" ,"super_resolution", "token_encoder"] = "diffusion",
    safety: A[bool, Dependent(on="model_type", when="diffusion")] = False,
    device: Literal[*get_gpus()] = next(iter(get_gpus()), None),
    float_32: A[bool, Dependent(on="device", when="cpu")] = False,
    bfloat: A[bool, Dependent(on="float_32", when="False")] = False,
) -> Any:
    print("loading:Safetensors:" + checkpoint)
    if model_type == "token_encoder":
        print("init tokenizer & text encoder")
        tokenizer = CLIPTokenizer.from_pretrained(
            checkpoint,
            subfolder='tokenizer',
        )
        text_encoder = CLIPTextModel.from_pretrained(
            checkpoint,
            subfolder='text_encoder',
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant='fp16',
        ).to(device)

        vectors = [{
            "tokenizer": tokenizer, 
            "text_encoder": text_encoder 
            }]
        return vectors

    elif model_type == "diffusion":
        print("create pipeline")
        pipe_args = {
                'use_safetensors': True,
                'tokenizer':None,
                'text_encoder':None,
                'tokenizer_2':None,
                'text_encoder_2':None,
            } 
        if float_32 != True:
            pipe_args.extend({
                'torch_btype': 'torch.float16' if bfloat == False else 'torch.bfloat16',
                'variant': 'fp16' if bfloat == False else 'bf16',
            })

        if safety is None: pipe_args.extend({'safety_checker': 'None',})

        print("apply pipeline")
        # Load the model on the graphics card
        pipe = AutoPipelineForText2Image.from_pretrained(checkpoint,**pipe_args).to(device)
        return pipe
        
    elif model_type == "vae":
        print("setup vae")
        checkpoint = AutoencoderKL.from_pretrained(
            'madebyollin/sdxl-vae-fp16-fix',
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(device)
        return autoencoder