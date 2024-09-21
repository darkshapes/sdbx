import os
import json
import sdbx.indexer as indexer
from sdbx import config, logger
from sdbx.config import config_source_location

path_name =  config.get_path("models.download") #multi read

def write_index():
    all_data = []  # Collect all data to write at once
    for each in os.listdir(path_name):  # SCAN DIRECTORY
        full_path = os.path.join(path_name, each)
        if os.path.isfile(full_path):  # Check if it's a file
            metareader = indexer.ReadMeta(full_path).data()
            if metareader is not None:
                all_data.append(indexer.EvalMeta(metareader).data())
            else:
                print(f"No data: {each}.")
    if all_data:
        try:
            os.remove(os.path.join(config_source_location,".index.json"))
        except FileNotFoundError as error_log:
            logger.debug(f"'Config file absent at index write: 'index.json'.'{error_log}", exc_info=True)
            pass
        with open(os.path.join(config_source_location, ".index.json"), "a", encoding="UTF-8") as index:
            json.dump(all_data, index, ensure_ascii=False, indent=4, sort_keys=True)
    else:
        print("Empty model directory, or no data to write.")

evaluate = write_index()

# print(f'end: {process_time_ns()*1e-6} ms')

# #import struct
# #import platform

#from pathlib import Path
#from functools import cache
#from dataclasses import dataclass
#from collections import defaultdict

#import networkx as nx
#from networkx import MultiDiGraph

# from sdbx import config, indexer
#from sdbx.config import get_default


# class NodeTuner:
#     def __init__(self, fn):
#         self.fn = fn
#         self.name = fn.info.fname

#     @cache
#     def get_tuned_parameters(self, widget_inputs, model_types, metadata):
#         max_value = max(metadata.values())
#         largest_keys = [k for k, v in metadata.items() if v == max_value] # collect the keys of the largest pairs
#         ReadMeta.full_data.get("size", 0)/psutil.virtual_memory().total

#         torch.device(0)
#         torch.get_default_dtype() (default float)
#         torch.cuda.mem_get_info(device=None) (3522586215, 4294836224)



#     def tuned_parameters(self, node_manager, graph: MultiDiGraph, node_id: str):
#         predecessors = graph.predecessors(node_id)

#         node = graph.nodes[node_id]

#         tuned_parameters = {}
#         for p in predecessors:
#             pnd = graph.nodes[p]  # predecessor node data
#             pfn = node_manager.registry[pnd['fname']]  # predecessor function

#             p_tuned_parameters = pfn.tuner.get_tuned_parameters(pnd['widget_inputs'])[node['fname']]

#             tuned_parameters |= p_tuned_parameters
        
#         return tuned
            
            
        # ### AUTOCONFIGURE OPTIONS  : TOKEN ENCODER
        # token_encoder_default = "stabilityai/stable-diffusion-xl-base-1.0" # this should autodetect

        # # AUTOCONFIG OPTIONS : INFERENCE
        # # [universal] lower vram use (and speed on pascal apparently!!)
        # sequential_offload = True
        # precision = '16'  # [universal], less memory for trivial quality decrease
        # # [universal] half cfg @ 50-75%. sdxl architecture only. playground, vega, ssd1b, pony. bad for pcm
        # dynamic_guidance = True
        # # [compatibility for alignyoursteps to match model type
        # model_ays = "StableDiffusionXLTimesteps"
        # # [compatibility] only for sdxl
        # pcm_default = "pcm_sdxl_normalcfg_8step_converted_fp16.safetensors"
        # pcm_default_dl = "Kijai/converted_pcm_loras_fp16/tree/main/sdxl/"
        # cpu_offload = False  # [compatibility] lower vram use by pushing to cpu
        # # [compatibility] certain types of models need this, it influences determinism as well
        # bf16 = False
        # timestep_spacing = "trailing"  # [compatibility] DDIM, PCM "trailing"
        # clip_sample = False  # [compatibility] PCM False
        # set_alpha_to_one = False,  # [compatibility]PCM False
        # rescale_betas_zero_snr = True  # [compatibility] DDIM True
        # disk_offload = False  # [compatibility] last resort, but things work
        # compile_unet = False #[performance] compile the model for speed, slows first gen only, doesnt work on my end

        # # AUTOCONFIG OPTIONS  : VAE
        # # pipe.upcast_vae()
        # vae_tile = True  # [compatibility] tile vae input to lower memory
        # vae_slice = False  # [compatibility] serialize vae to lower memory
        # # [compatibility] this should be detected by model type
        # vae_default = "madebyollin/sdxl-vae-fp16-fix.safetensors"
        # vae_config_file = "ssdxlvae.json"  # [compatibility] this too


        # # SYS IMPORT
        # device = ""
        # compile_unet = ""
        # queue = ""
        # clear_cache = ""
        # linux = ""

        # 20+ step little scheduler deviation
        # if xl
        #     try retrieve lora-pcm 16 step
        #     try retrieve ays 10 step
        #     try retrieve lora-lcm 4 step
        #     try retrieve lora-spo 
        #     try retrieve lora-tcd?

        #     load_in_4bit=True

        #     else set ays to xl type
        #     LCMScheduler
        # if sd
        #     try retrieve lora-pcm
        #     try retrieve lora-spo
        #     try retrieve lora-tcd?
        #     else set ays to xl type

        # if flux1
        #     try n4
            

        # Queue number = user set, >9 triggers lora fuse, torch compile
        # Frame/Queue = Dynamic, 75 nearing 1/mb  50 rising from 1/mb
        # Slice Queue  - dynamic/ 75 on, 50 off



        # item[0][:3]  #VAE,CLI,LOR,LLM, anything else should be treated like a Diff model


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


# find model name in [models]

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
