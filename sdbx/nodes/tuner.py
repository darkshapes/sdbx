import os
import json
import sdbx.indexer as indexer
from sdbx import config, logger
from sdbx.config import config_source_location
from collections import defaultdict

path_name =  config.get_path("models.download") #multi read

from time import process_time_ns
print(f'begin: {process_time_ns()*1e-6} ms')

class IndexManager:
    
    def write_index(self, index_file="index.json"):
        all_data = {
            "DIF": defaultdict(dict),
            "LLM": defaultdict(dict),
            "LOR": defaultdict(dict),
            "TRA": defaultdict(dict),
            "VAE": defaultdict(dict),
                    }  # Collect all data to write at once
        for each in os.listdir(path_name):  # SCAN DIRECTORY
            full_path = os.path.join(path_name, each)
            if os.path.isfile(full_path):  # Check if it's a file
                self.metareader = indexer.ReadMeta(full_path).data()
                if self.metareader is not None:
                    self.eval_data = indexer.EvalMeta(self.metareader).data()
                    if self.eval_data != None:
                        tag = self.eval_data[0]
                        filename = self.eval_data[1][0]
                        compatability = self.eval_data[1][1:2][0]
                        data = self.eval_data[1][2:5]
                        all_data[tag][filename][compatability] = (data)
                    else:
                        logger.debug(f"No eval: {each}.", exc_info=True)

                else:
                    log = f"No data: {each}."
                    logger.debug(log, exc_info=True)
                    print(log)
        if all_data:
            index_file = os.path.join(config_source_location, index_file)
            print(index_file)
            try:
                os.remove(index_file)
            except FileNotFoundError as error_log:
                logger.debug(f"'Config file absent at write time: {index_file}.'{error_log}", exc_info=True)
                pass
            with open(os.path.join(config_source_location, index_file), "a", encoding="UTF-8") as index:
                json.dump(all_data, index, ensure_ascii=False, indent=4, sort_keys=True)
        else:
            log = "Empty model directory, or no data to write."
            logger.debug(f"{log}{error_log}", exc_info=True)
            print(log)

    def fetch_compatible(self, data, query, path=None, index=False):
        if path is None: path = []

        if isinstance(data, dict):
            for key, self.value in data.items():
                self.current = path + [key]
                if self.value == query:
                    return self.__unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self.fetch_compatible(self.value, query, self.current)
                    if self.match:
                        return self.match
        elif isinstance(data, list):
            for key, self.value in enumerate(data):
                self.current = path if not index else path + [key]
                if self.value == query:
                    return self.__unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self.fetch_compatible(self.value, query, self.current)
                    if self.match:
                        return self.match
                    
    def __unpack(self): 
        iterate = []  
        self.match = self.current, self.value           
        for i in range(len(self.match)-1):
            for j in (self.match[i]):
                iterate.append(j)
        iterate.append(self.match[len(self.match)-1])
        return iterate

#write = IndexManager().write_index()
clip_data = config.get_default("tuning", "clip_data") 
query = 'STA-XL'


root, *path = IndexManager().fetch_compatible(clip_data, query)
print(root)
print(*path[:len(path)-1] if not None else """ignore""")
vae_index = config.get_default("index", "VAE")
print(next((key for key, value in vae_index.items() if query in value), None))
tra_index = config.get_default("index", "TRA")
print(next((key for key, value in tra_index.items() if root in value), None))
for each in path:
    if each != query:
        print(next((key for key, value in tra_index.items() if each in value), None))

#print(  next( ( value for key, value in vae_index.items() if query in value), None)  )
#print(next((value for value in vae_index.values() if query in value), None))
#next_query = root[0]
#tra_index = config.get_default("index", "TRA") #
#print((next((i for i in tra_index.values() if next_query in i), None)))


print(f'end: {process_time_ns()*1e-6} ms')

# check if user has these models
# if so, add loaders for them and attach
# if not, use main model

# model_types = ["DIF", "LLM", "LOR", "TRA", "VAE"]

# value_2 = IndexManager().fetch("STA-15",value, sub_string=True)
# print(value)
# # try:
# #     result.append(index[key]) 
# # except KeyError as error_log:
# #     logger.debug(f"{log}{error_log}", exc_info=True)
# #     pass
# #obj = json.loads('{[{"key": ["foo"]}]}')


# import psutil
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

# model str
# lora
# pcm
# spo
# tcd
# hyp
# fla
# lcm
# dmd
# ays
# vae
# mem
# quant
# dtype 
# sequential bool
# cpu int
# disk bool
# batch limit int
# upcast bool
# cache eject bool
# pipeline str
# compile bool
# scheduler str
# scheduler properties str
# cfg/dynamite guide str
# steps int


# filename = "noosphere_v42.safetensors"
# fetch = IndexManager().fetch(filename,"index.json", sub_string=False)
# print(fetch)

# fetch_map = config.get_default("index",1)

# # fetch_map = IndexManager().fetch_map(0,"index.json")
# print((fetch_map))




#obj = json.loads('{[{"key": ["foo"]}]}')

# model_code, model_size, model_path, model_precision = fetch[:4]
# short_code = model_code[0:3] 
# ver_code = [len(model_code)[-2:]]

# lora_types = { 
#     "PCM",
#     "SPO",
#     "TCD",
#     "HYP",
#     "LCM",
#     "FLA"
#     }

# for types in lora_types:
#     type_code = f"LOR-{types}-{model_code}"
#     fetch = IndexManager().fetch(type_code,"index.json", sub_string=True)

# if fetch, set lora to fetch else set ays if model_code == sta-xl, pla, 10

    # timestep_spacing = "trailing"  # [compatibility] DDIM, PCM "trailing"
    # clip_sample = False  # [compatibility] PCM False
    # set_alpha_to_one = False,  # [compatibility]PCM False
    # rescale_betas_zero_snr = True  # [compatibility] DDIM True 
    # steps x # [compatibility] LCM, LIG, FLA, DPO, PCM, HYP, TCD True
    # cfg x # [compatibility] LCM, LIG, FLA, DPO, PCM, HYP, TCD True
    # dynamic_guidance = True [universal] half cfg @ 50-75%. xl only.no lora/pcm

#find compatible vae
    # fetch vae
    # small vae if >75
    # fetch vaeconfig.json
    # pipe.upcast_vae() if sdxl

# # overhead = model_size+lora_size+vae+size
# #cpu_ceiling = overhead/psutil.virtual_memory().total
# import torch

# avail_vid_ram = overhead/torch.cuda.mem_get_info()
# gpu_ceiling = model_size/avail_vid_ram[1]

#calc ram-specific params
    # try independent unet
    # try independent clips
    # token_encoder_default = "TRA"-found[1] if "TRA"=found[1] else: found[0]# this should autodetect
    # look for quant, load_in_4bit=True >75
    # dtypes auto <50
    # no batch limit <50

    # dtypes 16 or less >75
    # vae_tile = True  >75 # [compatibility] tile vae input to lower memory
    # vae_slice = False >75  # [compatibility] serialize vae to lower memory
    # batch = 1 > 75, 
    # sequential_offload = True >75 # [universal] lower vram use (and speed on pascal apparently!!)
    # cpu_offload = False  >90 # [compatibility] lower vram use by forcing to cpu
    # disk_offload = False >90  # [compatibility] last resort, but things work
    # compile_unet = False if unet #[performance] unet only, compile the model for speed, slows first gen only, doesnt work on my end
    # cache jettison - True > 75

# choose pipeline
    # set pipeline params
    #     scheduler (if not already changed)
    #     inference steps (if not already changed)
    #     cfg (if not already changed) # dynamic cfg (rarely)
    #     resolution

# refiner
    #     high_noise_frac = 0.8
    #     image = base(
    #     image = refiner(
    #     high_noise_frac = 0.8
    #     denoising_end=high_noise_frac,
    #     num_inference_steps=n_steps,
    #     denoising_start=high_noise_frac,

#     compile? pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)

# strands [
#     upscale
#     hiresfix
#     zero conditioning
#     prompt injection
#     x/y map
# ]

# if short_code == "LLM":
#     """
#     context_len = model_precision
#     """
# elif short_code == "STA":

#     if ver_code == "15":

#         model_ays = "StableDiffusionTimesteps"
#         vae_code = f"VAE-{model_code}"
#         fetch = IndexManager().fetch(vae_code,"index.json",value=True)
#         vae_file = os.path.basename(fetch[2])
#         tra_code = f"TRA-CLI-VL"
#         fetch = IndexManager().fetch(tra_code,"index.json",value=True)
#         tra_file = os.path.basename(fetch[2])
#         lor_code = f"VAE-{lor_code}"
#         fetch = IndexManager().fetch(lor_code,"index.json",value=True)
#         lor_file = os.path.basename(fetch[2])
#         print(fetch)
#         print(fetch)
#     elif ver_code =="XL":
#         """
#         vae_code - f"VAE-{model_code}"
#         model_ays = "StableDiffusionXLTimesteps
#         """
#     elif ver_code =="3M" or ver_code == "3Q" or ver_code == "3C":
#         """
#         model_ays = "StableDiffusion3Timesteps
#         """"
# elif short_code == "FLU": #FLUX
#     """
#     1D 1S 01"
#     """
   


# class NodeTuner:
#     def __init__(self, fn):
#         self.fn = fn
#         self.name = fn.info.fname

#     @cache
#     def get_tuned_parameters(self, widget_inputs, model_types, metadata):
#         max_value = max(metadata.values())
#         largest_keys = [k for k, v in metadata.items() if v == max_value] # collect the keys of the largest pairs
#         ReadMeta.full_data.get("model_size", 0)/psutil.virtual_memory().total

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
