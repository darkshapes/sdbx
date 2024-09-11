import inspect
import logging
import json
import struct
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import os

from functools import cache
from typing import Callable, Dict
from dataclasses import dataclass
from sdbx.config import DTYPE_T, TensorData
import networkx as nx
from networkx import MultiDiGraph

# @dataclass
source = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_source_location = os.path.join(source, "config")

class ReadMeta:
    def __init__(self, file, meta):
        model_tag = {
                    'filename': '',
                    'size':'',
                    'dtype': '',
                    'tensor_params': 0,
                    'shape': '',
                    "__metadata__": '',
                    'info.files_metadata': '',
                    'file_metadata':'',
                    'name': '',
                    'info.sharded': '',
                    'info.metadata': '',
                    'file_metadata.tensors':'',
                    'modelspec.title':'',
                    'modelspec.architecture':'',
                    'modelspec.author':'',
                    'modelspec.hash_sha256':'',
                    'modelspec.resolution':'',
                    'resolution': '',
                    'ss_resolution':'',
                    'ss_mixed_precision': '',
                    'ss_base_model_version':'',
                    'ss_network_module': '',
                    'model.safetensors':'',
                    'ds_config': '',
                    }
                            
        if not os.path.exists(file):
            raise RuntimeError(f"Not found: {file}")
        else:
            model_tag['filename'] = os.path.basename(file)
            model_tag['size'] = os.path.getsize(file)

    def data(self, file):

        if Path(file).suffix in {'.pt', '.pth'}: #yes, this will be slow
            model = torch.load(file, map_location='cpu') #Access model's metadata (e.g., layer names and parameter shapes)
            for p in model.named_parameters():
                print(f"Data: {p}, Size: {p.size()}")

        elif Path(file).suffix in {".safetensors" or ".sft"}:
            meta = self._parse_safetensors_metadata(file)
            for key, value in self.model_tag.items(): print(key,value) #debug
            #with open(f'{t}.json', 'w') as f: json.dump(meta, f)
            return print(f"saved for {file}")
        
        elif Path(file).suffix in {".gguf"}:
            meta = "" #parse gguf metadata(path)
        else:
            raise RuntimeError(f"Unknown file format: {file}")
                   
    def _parse_safetensors_metadata(self, file):
        with open(file, 'rb') as f:
            header = struct.unpack('<Q', f.read(8))[0]
            return json.loads(f.read(header), object_hook=self._search_dict)

    def _search_dict(self, meta):
        try: self.model_tag['__metadata__'] = (meta.get('__metadata__')) 
        except: pass
        
        # get path 
        # model_class = Path(os.path.join(config_source_location, "model_classes.toml"))
        # with open(model_class, "rb") as f: fd = tomllib.load(f)
        # name = model_class.stem
        # d[name] = fd
        # with open(model_class, "rb") as f: data = tomllib.load(f)
        # for each in config.model_classes.toml:
        # meta.get()

        for key, value in self.model_tag.items():
            if meta.get(key, 'not_found') != 'not_found':
                if self.model_tag[key] == '':
                    self.model_tag[key] = meta.get(key)
                if key == 'dtype': 
                    self.model_tag['tensor_params'] += 1
                if key == 'shape':
                    if meta.get(key) > self.model_tag['shape']: self.model_tag['shape'] = self.meta.get(key) 
        return meta
        
#path = os.path.join("c:",os.sep,"users","public","models","loras")
#for t in os.listdir(path):
#    if ".safetensors" in t:
#        ReadMeta.data(os.path.normpath(os.path.join(path, t)))

class NodeTuner:
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.info.fname

    @cache
    def get_tuned_parameters(self, widget_inputs):

        #if no metadata os.path.getsize(file_path)

        # if self.name is "load gguf" use this routine to pull metadata

        # NOTE: self.name comparisons are against function name, not declared (@node) name


        # fetch dict of system specifications - inside config.calibration later
        # fetch safetensors metadata

        
        # if there is no metadata
        # fall back to filename and guess
        
        # if no filename
        # pick a generic config.json from models.metadata and cross ur fingers

        # if self.name is "load gguf"
        # new dependency - > gguf>=0.10.0
        # fetch gguf data
        # check

        # if self.name is "diffusion"
        
        # if self.name is "load safetensors"

        # if self.name is "prompt"
        # if llm model then # show temp
        # if self.name is save??????????


        
        # This needs to return a dictionary of all of the tuned parameters for any given
        # node given the current widget inputs. For example, if this is for the Loader node,
        # and the current widget input is { "model": "pcm8_model" }, it should return:

        # Key is the function name of the node whose parameters you want to change.
        # Value is all of the parameters you would like to change and their value. For example:


        # compare the values and assign sensible variables
        # generate dict of tuned parameters like below:
        # return the dict

        # tuned parameters & hyperparameters only!! pcm parameters here 

        # return {
        #     "function name of node": {
        #         "parameter name": "parameter value",
        #         "parameter name 2": "parameter value 2"
        #     }
        co

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