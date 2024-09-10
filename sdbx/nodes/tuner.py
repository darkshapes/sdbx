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
    
# @dataclass
class ReadMeta:
    # pass a file to this class like so : ReadMeta.data(file)
    def data(file: Path,):
        dict_ref = {
                    '__metadata__': '',
                    'dtype': '',
                    'ds_config': '',
                    'modelspec.title':'',
                    'modelspec.architecture':'',
                    'modelspec.author':'',
                    }
            
        def _search_dict(meta):
            for key, value in dict_ref.items():
                if meta.get(key, 'not_found') != 'not_found':
                    if dict_ref[key] == '':
                       dict_ref[key] = meta.get(key)

            return meta
        
        def _parse_safetensors_metadata(file: Path):
            with open(file, 'rb') as f:
                header = struct.unpack('<Q', f.read(8))[0]
                return json.loads(f.read(header).decode('utf-8'), object_hook=_search_dict)

        if not os.path.exists(file):
            raise RuntimeError(f"Not found: {file}")

        if file.suffix in {'.pt', '.pth', '.ckpt'}: #yes, this will be slow
            model = torch.load(file, map_location='cpu') #Access model's metadata (e.g., layer names and parameter shapes)
            for x in model.named_parameters():
                print(f"Data: {x}, Size: {x.size()}")

        elif file.suffix in {".safetensors" or ".sft"}:
            with open('sd3clip.json', 'w') as f: 
                    json.dump(_parse_safetensors_metadata(file),f)
            return print("saved")
        
        elif file.suffix in {".gguf"}:
            meta = "" #parse gguf metadata(path)
        else:
            raise RuntimeError(f"Unknown file format: {file}")