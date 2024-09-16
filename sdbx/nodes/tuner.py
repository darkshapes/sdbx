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

class ReadMeta: # instance like so - ReadMeta(filename, full_path).data(filename, full_path)
    full_data, meta, count_dict = {}, {}, {} # level of certainty, counts tensor block type matches
    occurrence_counts = defaultdict(int)
    model_tag = {  # measurements and metadata detected from the model
            "filename": "",
            "size": "",
            "dtype": "",
            "tensor_params": 0,
            "shape": "",
            "data_offsets": "",
            "__metadata__": "",
            "info.files_metadata": "",
            "file_metadata": "",
            "name": "",
            "info.sharded": "",
            "info.metadata": "",
            "file_metadata.tensors": "",
            "modelspec.title": "",
            "modelspec.architecture": "",
            "modelspec.author": "",
            "modelspec.hash_sha256": "",
            "modelspec.resolution": "",
            "resolution": "",
            "ss_resolution": "",
            "ss_mixed_precision": "",
            "ss_base_model_version": "",
            "ss_network_module": "",
            "model.safetensors": "",
            "ds_config": "",
        }

    known = { #known model blocks and their use
            "adaLN_modulation":"mmdit",
            "mlp.fc":"mmdit",
            "mlpX":"mmdit",
            "w1q":"mmdit",
            "self_attn.out_proj":"mmdit",
            "w1q":"mmdit",
            "w1o":"mmdit",
            "mlpX.c_proj":"mmdit",
            "mlpC.c_proj":"mmdit",
            "w2q.":"mmdit",
            "w2k.":"mmdit",
            "w2v.":"mmdit",
            "w2o.":"mmdit",
            "w1k.":"mmdit",
            "w1v.":"mmdit",
            "mlpX.c_fc":"mmdit",
            "mlpX.c_proj.":"mmdit",
            "mlpC.c_fc":"mmdit",
            "modC.":"mmdit",
            "modX.":"mmdit",
            "model.register_tokens":"auraflow",
            "model.positional_encoding":"auraflow",
            "model.init_x_linear":"auraflow",
            "model.t_embedder.mlp.":"auraflow",
            "model.t_embedder.mlp.":"auraflow",
            "modCX":"flux",
            "img_attn.proj":"flux",
            "time_in.in_layer":"flux",
            "time_in.out_layer":"flux",
            "vector_in.in_layer":"flux",
            "vector_in.out_layer":"flux",
            "guidance_in.in_layer":"flux",
            "guidance_in.in_layer":"flux",
            "txt_in":"flux",
            "img_in":"flux",
            "img_mod.lin":"flux",
            "txt_mod.lin":"flux",
            "img_attn.qkv":"flux",
            "txt_attn.qkv":"flux",
            "t_embedder.mlp":"pixart_s",
            "y_embedder.embedding_table":"pixart_s",
            "wkqv.":"hunyuan",
            "wqkv.":"hunyuan",
            "q_norm":"hunyuan",
            "k_norm":"hunyuan",
            "out_proj":"hunyuan",
            "kq_proj":"hunyuan",
            "default_modulation.":"hunyuan",
            "pooler":"hunyuan",
            "t_embedder":"hunyuan",
            "x_embedder":"hunyuan",
            "mlp_t5":"hunyuan",
            "time_extra_emb.extra_embedder":"hunyuan",
            "to_q":"diffusers",
            "to_k":"diffusers",
            "to_v":"diffusers",
            "norm_q":"diffusers",
            "norm_k":"diffusers",
            "to_out":"diffusers",
            "norm1.norm":"diffusers",
            "norm1.linear":"diffusers",
            "ff.net.0":"diffusers",
            "ff.net.2":"diffusers",
            "time_extra_emb":"diffusers",
            "time_embedding":"diffusers",
            "pos_embd":"diffusers",
            "text_embedder":"diffusers",
            "extra_embedder":"diffusers",
            "attn.norm_added_q":"diffusers",
            "skip.connection":"sdxl",
            "upsamplers":"sdxl",
            "downsamplers":"sdxl",
            "op":"sdxl",
            "in.layers.2":"sdxl",
            "out.layers.3":"sdxl",
            "in_layers":"sd",
            "out_layers":"sd",
            "emb_layers":"sd",
            "skip_connection":"sd",
            "text_model":"diffusers_lora",
            "self_attn":"diffusers_lora",
            "to_q_lora":"diffusers_lora",
            "to_k_lora":"diffusers_lora",
            "to_v_lora":"diffusers_lora",
            "to_out_lora":"diffusers_lora",
            "text_projection":"diffusers_lora",
            "to.q.lora":"unet_lora",
            "to.k.lora":"unet_lora",
            "to.v.lora":"unet_lora",
            "to.out.0.lora":"unet_lora",
            "proj.in":"unet_lora",
            "proj.out":"unet_lora",
            "emb.layers":"unet_lora",
            "proj_in.":"transformers",
            "proj_out.":"transformers",
            "norm.":"transformers",
            "norm1.":"unet",
            "norm2.":"unet",
            "norm3.":"unet",
            "attn1.to_q.":"unet",
            "attn1.to_k.":"unet",
            "attn1.to_v.":"unet",
            "attn1.to_out.0.":"unet",
            "attn2.to_q.":"unet",
            "attn2.to_k.":"unet",
            "attn2.to_v.":"unet",
            "attn2.to_out.0.":"unet",
            "ff.net.0.proj.":"unet",
            "ff.net.2.":"unet",
        }
    
    @classmethod
    def __init__(
        self, filename, full_path
    ):

        self.full_path = full_path #the path of the file
        self.filename = filename #the title of the file only
        if not os.path.exists(self.full_path): #be sure it exists, then proceed
            raise RuntimeError(f"Not found: {self.filename}")
        else:
            self.model_tag["filename"] = self.filename
            self.model_tag["size"] = os.path.getsize(self.full_path)

    @classmethod
    def _parse_safetensors_metadata(self, full_path):
        with open(full_path, "rb") as json_file:  # open the model file header, anticipating json structured data
            header = struct.unpack("<Q", json_file.read(8))[0]
            try:
                return json.loads(json_file.read(header), object_hook=self._search_dict) # hook search
            except:
                return print(f"error loading {full_path}")

    @classmethod
    def data(self, filename, full_path):
        if Path(filename).suffix in {".pt", ".pth", ".ckpt"}: #process torch formats elsewhere
            return
        elif Path(filename).suffix in {".safetensors" or ".sft"}:
            self.occurrence_counts.clear() #empty values
            self.full_data.clear() #prepare dict
            self.meta = self._parse_safetensors_metadata(full_path) #analyse file contents
            self.full_data.update((k,v) for k,v in self.model_tag.items() if v != "") #make a new dict with all attributes
            self.full_data.update((k,v) for k,v in self.count_dict.items() if v != 0) #part 2 dict boogaloo
            self.count_dict.clear() #empty counter
            self.model_tag.clear() # clean up lingering values
            self.meta.clear() #dump file contents
           # for  k, v in self.full_data.items():            #diagnostics
           #     print(k,v)
           # return self.full_data

        elif Path(filename).suffix in {".gguf"}:
            meta = ""  # placeholder - parse gguf metadata(path) using llama lib
        elif Path(filename).suffix in {".bin"}:
            meta = ""  # placeholder - parse bin metadata(path) using ...???
        else:
            raise RuntimeError(f"Unknown file format: {filename}")

    @classmethod
    def _search_dict(self, meta):
        self.meta = meta
        for num in list(self.meta): #list chunks of the metadata
            if self.model_tag.get(num, "not_found") != "not_found": #handle inevitable exceptions invisibly
                self.model_tag[num] = meta.get(num) #drop it like its hot 
            if "dtype" in num: #counting these keys reveals tensor count
                 self.model_tag["tensor_params"] += 1 # count tensors
            elif "shape" in num:
                if meta.get(num) > self.model_tag["shape"]: #measure tensor shape 
                     self.model_tag["shape"] = self.meta.get(num) # keep largest dimension values
            if "data_offsets" in num: #not sure what to do with this yet
                pass
            else:
                if ("shapes" or "dtype") not in num: # block names only
                    for block, model_type in self.known.items():  #model type, dict data
                        if block in num:     #if value matches one of our key values
                            self.occurrence_counts[model_type] += 1 #count matches
                            self.count_dict[model_type] = self.occurrence_counts.get(model_type, 0) #pair match count with type of model  
        return meta   
# print(f'end: {process_time_ns()/1e6} ms')  debig
# metareader = ReadMeta(each, full_path).data(each, full_path)


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