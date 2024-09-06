from sdbx import config
from sdbx.nodes.types import *
from sdbx.nodes.helpers import softRandom, hardRandom, getGPUs
from sdbx.config import config
from sdbx.nodes.helpers import getDirFiles
from diffusers import AutoPipelineForText2Image
import torch

import os
from llama_cpp import Llama

debug = True # eeeee

@node(name="GGUF Loader")
def gguf_loader(
    checkpoint: Literal[*getDirFiles("models.llms", ".gguf")] = getDirFiles("models.llms", ".gguf")[0],
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
    checkpoint: Literal[*getDirFiles("models.checkpoints", ".safetensors")] = getDirFiles("models.checkpoints", ".safetensors")[0],
    model_type: Literal["diffusion", "autoencoder" ,"super_resolution", "token_encoder"] = "diffusion",
    safety: A[bool, Dependent(on="model_type", when="diffusion")] = False,
    device: Literal[*getGPUs()] = getGPUs()[0],
    # precision: [int, Slider(min=16, max=32, step=16), Dependent(on="getGPUs()", when=f"{not 'cpu'}")] = 16, 
    bfloat: A[bool, Dependent(on="precision", when="16")] = False,
) -> torch.Tensor:
    print("loading:Safetensors:" + checkpoint)
    if model_type == "token_encoder":
        if debug==True: print("init tokenizer & text encoder")
        tokenizer = CLIPTokenizer.from_pretrained(
            token_encoder,
            subfolder='tokenizer',
        )
        text_encoder = CLIPTextModel.from_pretrained(
            token_encoder,
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
        if debug==True: print("create pipeline")
        pipe_args = {
                'use_safetensors': True,
                'tokenizer':None,
                'text_encoder':None,
                'tokenizer_2':None,
                'text_encoder_2':None,
            } 
        if precision != 32:
            pipe_args.extend({
                'torch_btype': 'torch.float16' if bfloat == False else 'torch.bfloat16',
                'variant': 'fp16' if bfloat == False else 'bf16',
            })

        if safety is None: pipe_args.extend({'safety_checker': 'None',})

        if debug==True: print("apply pipeline")
        # Load the model on the graphics card
        pipe = AutoPipelineForText2Image.from_pretrained(model,**pipe_args).to(device)
        return pipe
        
    elif model_type == "vae":
        if debug==True: print("setup vae")
        autoencoder = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix',
        use_safetensors=True,
        torch_dtype=torch.float16,
        ).to(device)
        return autoencoder



