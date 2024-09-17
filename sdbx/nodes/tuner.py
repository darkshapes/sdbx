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

        # check filename for clues
        # check file size
        # check tensor size

        # 3072,3072 shape flux
        # 11891971136 offset flux
        # > 11901525888 flux
        # 780 params flux
        # flux flux

        # 49408 shape x value - sdxl
    
        6938041280 6938040682 6938040706
        # ~6,778,430-632KB size - fp16 sdxl
        # 2,519,599-841kb size - fp16 sd1.5

        #lightning - no vae?

        # vae
        # 163mb size sd/xl vae f16
        # 326mb size sd/xl vae f32
        # size 50mb-850mb - vae
        # [512, 512, 3, 3] shape vae
        # unet - vae
        # ~75mb size cascade stage a
        # i64 dtype cascade stage a
        # [8192,4] shape

        # taesd
        # 5mb taesd/xl decoder/encoder
        # 10mb taesd sd3

        # lora
        # 20-500mb  upscale
        # pth - upscale 
        # bin
        # 2gb aura sr* 
        # safetensors aurasr

        # lora
        # 10240 shape x value - sdxl lora
        # tensor params 700-4000 lora
        # lightning lora 5000 params
        # fp16, fp32, bf16 lora
        # ss_base_model_version sdxl_base_v1.0 lora
        # (metadata byte size)

        #pcm lora
        # fp16
        # 2364 params
        # 10240
        # ~393854624 bytes
        # lean on filename

        #lcm lora
        # fp16
        # 2364 params
        # 10240
        # rt ssd1b lcm
        # 100-400 mb
        # 393854624
        # 393854592
        # 393855224
        # lean on filename here
        
        # sd lora 134621556


        # unet 
        # > 2000 tensor params

        # transformers just shows up as transformers
        # i64 transformers, open clip
        # 
        # 49408 shape clip
        # 200-2000 params clip
        # clip g mislabeled as diffusers lora!!!

        # mmdit -
        # flux -
        # pixxart s

        # sd - enable pcm, ays
        # if hunyuan - 
        # if diffusers - image interpreter (diffusers)
        # if sdxl - enable pcm, ays option, force upcast vae if default vae, otherwise use fp16

        # if transformers - text model
        # compare the values and assign sensible variables
        # generate dict of tuned parameters like below:
        # return the dict

        #control net
        # 844 params
        # 10240 x shape
        # unet model type

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