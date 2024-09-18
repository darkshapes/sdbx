from time import process_time_ns
# print(f'end: {process_time_ns()/1e6} ms') debug   

import os
import json
import struct
import platform

from pathlib import Path
from functools import cache
from dataclasses import dataclass
from collections import defaultdict

import networkx as nx
from networkx import MultiDiGraph


class NodeTuner:
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.info.fname

    @cache
    def get_tuned_parameters(self, widget_inputs, model_types, metadata):
        max_value = max(metadata.values())
        largest_keys = [k for k, v in metadata.items() if v == max_value] # collect the keys of the largest pairs
        ReadMeta.full_data.get("size", 0)/psutil.virtual_memory().total

        torch.device(0)
        torch.get_default_dtype() (default float)
        torch.cuda.mem_get_info(device=None) (3522586215, 4294836224)

        # get system memory
        # get cpu generation
        # get gpu type
        # get gpu generation
        # get gpu memory
        # goal - high, but stable resource allocation
        # intent - highest quality at reasonable speed
        # gpu > cpu


        # match loader code to model type
        # check lora availability
        # if no lora, check if AYS or other optimization available
        # MATCH LORA TO MODEL TYPE BY PRIORITY
        #     PCM
        #     SPO
        #     HYP
        #     LCM
        #     RT
        #     TCD
        #     FLA
        # match loader code to model type
        # ram calc+lora
        # decide sequential offload on 75% util/ off 50% util
        # match dtype to lora load
        # MATCH VAE TO MODEL TYPE
        # match dtype to vae load


        # except:
        #    pass
        # else:
        #    each = "klF8Anime2VAE_klF8Anime2VAE.safetensors" #ae.safetensors #noosphere_v42.safetensors #luminamodel .safetensors #mobius-fp16.safetensors

        #check variable against available
        # gpu mem
        # cpu mem
        # size of model
        # if cpu know proc speed?
        # bf16 = bf16
        # fp16 = fp16
        # fp16 = 
        # bf16
        # fp16
        # fp32
        # fp64
        # ays
        # pcm
        # dyg
        # gpu
        # cpu
        # mtcache
        # cache
        # compile
        # nocompile
        # bf16>fp16>fp32
        # xl? ays>pcm>dynamic cfg
        # tokenizer? allocate gpu layers
        # dump cache!!!, full allocation every step
        # try to compile
        # except, skip

        # flash-attn/xformers

### AUTOCONFIGURE OPTIONS  : TOKEN ENCODER
token_encoder_default = "stabilityai/stable-diffusion-xl-base-1.0" # this should autodetect

# AUTOCONFIG OPTIONS : INFERENCE
# [universal] lower vram use (and speed on pascal apparently!!)
sequential_offload = True
precision = '16'  # [universal], less memory for trivial quality decrease
# [universal] half cfg @ 50-75%. sdxl architecture only. playground, vega, ssd1b, pony. bad for pcm
dynamic_guidance = True
# [compatibility for alignyoursteps to match model type
model_ays = "StableDiffusionXLTimesteps"
# [compatibility] only for sdxl
pcm_default = "pcm_sdxl_normalcfg_8step_converted_fp16.safetensors"
pcm_default_dl = "Kijai/converted_pcm_loras_fp16/tree/main/sdxl/"
cpu_offload = False  # [compatibility] lower vram use by pushing to cpu
# [compatibility] certain types of models need this, it influences determinism as well
bf16 = False
timestep_spacing = "trailing"  # [compatibility] DDIM, PCM "trailing"
clip_sample = False  # [compatibility] PCM False
set_alpha_to_one = False,  # [compatibility]PCM False
rescale_betas_zero_snr = True  # [compatibility] DDIM True
disk_offload = False  # [compatibility] last resort, but things work
compile_unet = False #[performance] compile the model for speed, slows first gen only, doesnt work on my end

# AUTOCONFIG OPTIONS  : VAE
# pipe.upcast_vae()
vae_tile = True  # [compatibility] tile vae input to lower memory
vae_slice = False  # [compatibility] serialize vae to lower memory
# [compatibility] this should be detected by model type
vae_default = "madebyollin/sdxl-vae-fp16-fix.safetensors"
vae_config_file = "ssdxlvae.json"  # [compatibility] this too


    lora=pcm_default

# Device and memory setup
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"

def clear_memory_cache(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# SYS IMPORT
device = ""
compile_unet = ""
queue = ""
clear_cache = ""
linux = ""

        # tuned parameters & hyperparameters only!! pcm parameters here 

        # return {
        #     "function name of node": {
        #         "parameter name": "parameter value",
        #         "parameter name 2": "parameter value 2"
        #     }

        # > 1mb embeddings
        # up to 50mb?
        # pt and safetensors
        # <2 tensor params

        # pt files

    def tuned_parameters(self, node_manager, graph: MultiDiGraph, node_id: str):
        predecessors = graph.predecessors(node_id)

        node = graph.nodes[node_id]

        tuned_parameters = {}
        for p in predecessors:
            pnd = graph.nodes[p]  # predecessor node data
            pfn = node_manager.registry[pnd['fname']]  # predecessor function

            p_tuned_parameters = pfn.tuner.get_tuned_parameters(pnd['widget_inputs'])[node['fname']]

            tuned_parameters |= p_tuned_parameters
        
        return tuned