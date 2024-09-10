import inspect
import logging
import json

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

        
        if self.name is "safetensors_loader": 
        data_offsets: Tuple[int, int]
        parameter_count: int = field(init=False)

        def __post_init__(self) -> None:  #  https://stackoverflow.com/a/13840436
        try:
            self.parameter_count = functools.reduce(operator.mul, self.shape)
        except TypeError:
            self.parameter_count = 1  # scalar value has no shape


        metadata: dict[str, str]
        tensors: dict[str, TensorData]
        parameter_count: dict[DTYPE_T, int] = field(init=False)

        def __post_init__(self) -> None:
            parameter_count: dict[DTYPE_T, int] = defaultdict(int)
            for tensor in self.tensors.values():
                parameter_count[tensor.dtype] += tensor.parameter_count
            self.parameter_count = dict(parameter_count)
    
        parse_safetensors_file_metadata(
            "HuggingFaceH4/zephyr-7b-beta", "model-00003-of-00008.safetensors")


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
@dataclass
class SafetensorsMetadata: # https://huggingface.co/docs/safetensors/index#format.

    metadata: Dict[str, str] # The metadata contained in the file
    tensors: Dict[TENSOR_NAME_T, TensorInfo] # tensors map. Keys are tensor names and values are information about the corresponding tensor, as a parameter_count (`Dict[str, int]`):
    parameter_count: Dict[DTYPE_T, int] = field(init=False) # parameter num per datatype map. data types:  parameters,
            of that data type.

    def get_data(self, model) -> #not sure what type this returns:
        info = self._api.get_safetensors_metadata(model)    hmmm
    _api: "HfApi" = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        parameter_count: Dict[DTYPE_T, int] = defaultdict(int)
        for tensor in self.tensors.values():
            parameter_count[tensor.dtype] += tensor.parameter_count
        self.parameter_count = dict(parameter_count)

_parameters

    def test_get_safetensors_metadata_sing, modelle_file(seloomz-560m")
        assert isinstance(info, SafetensorsRepoMetadata)

        assert not info.sharded
        assert info.metadata is None  # Never populated on non-sharded model
        assert len(info.files_metadata) == 1
        assert "model.safetensors" in info.files_metadata

    def test_whoami_with_passing_token(self):
        info = self._api.whoami(token=self._token)
        file_metadata = info.files_metadata["model.safetensors"]

        assert isinstance(file_metadata, SafetensorsFileMetadata)
        assert file_metadata.metadata == {"format": "pt"}
        assert len(file_metadata.tensors) == 293

        assert isinstance(info.weight_map, dict)
        assert info.weight_map["h.0.input_layernorm.bias"] == "model.safetensors"

        assert info.parameter_count == {"F16": 559214592}

def test_parse_safetensors_metadata(self) -> None:
        info = self._api.parse_safetensors_file_metadata(
            "HuggingFaceH4/zephyr-7b-beta", "model-00003-of-00008.safetensors"
        )
        assert isinstance(info, SafetensorsFileMetadata)

        assert info.metadata == {"format": "pt"}
        assert isinstance(info.tensors, dict)
        tensor = info.tensors["model.layers.10.input_layernorm.weight"]

        assert tensor == TensorInfo(dtype="BF16", shape=[4096], data_offsets=(0, 8192))

        assert tensor.parameter_count == 4096
        assert info.parameter_count == {"BF16": ter_count)

