from time import process_time_ns
# print(f'end: {process_time_ns()/1e6} ms') debug   

import json
import struct
from pathlib import Path
import os
from collections import defaultdict
import platform

from functools import cache
from dataclasses import dataclass
import networkx as nx
from networkx import MultiDiGraph


# AUTOCONFIGURE OPTIONS : this should autodetec
token_encoder_default = "stabilityai/stable-diffusion-xl-base-1.0"
lora_default = "pcm_sdxl_normalcfg_8step_converted_fp16.safetensors"
pcm_default = "Kijai/converted_pcm_loras_fp16/tree/main/sdxl/"
vae_default = "madebyollin/sdxl-vae-fp16-fix.safetensors"
model_default = "ponyFaetality_v11.safetensors"
llm_default = "codeninja-1.0-openchat-7b.Q5_K_M.gguf"
scheduler_default = "EulerAncestralDiscreteScheduler"



class NodeTuner:
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.info.fname

    @cache
    def get_tuned_parameters(self, widget_inputs, model_types, metadata):
        max_value = max(metadata.values())
        largest_keys = [k for k, v in metadata.items() if v == max_value] # collect the keys of the largest pairs
        ReadMeta.full_data.get("size", 0)/psutil.virtual_memory().total
        # get system memory
        # get cpu generation
        # get gpu type
        # get gpu generation
        # get gpu memory
        # goal - high, but stable resource allocation
        # intent - highest quality at reasonable speed
        # gpu > cpu


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

        #metareader = single_run(config_path, search_path, search_name)


        # user select model
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


        #except:
        #    pass
        #else:
        #    each = "klF8Anime2VAE_klF8Anime2VAE.safetensors" #ae.safetensors #noosphere_v42.safetensors #luminamodel .safetensors #mobius-fp16.safetensors



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
        

    def collect_tuned_parameters(self, node_manager, graph: MultiDiGraph, node_id: str):
        predecessors = graph.predecessors(node_id)

        node = graph.nodes[node_id]

        tuned_parameters = {}
        for p in predecessors:
            pnd = graph.nodes[p]  # predecessor node data
            pfn = node_manager.registry[pnd['fname']]  # predecessor function

            p_tuned_parameters = pfn.tuner.get_tuned_parameters(pnd['widget_inputs'])[node['fname']]

            tuned_parameters |= p_tuned_parameters
        
        return tuned