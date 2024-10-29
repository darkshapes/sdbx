import os
import struct
import json
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union, Optional, Any
from pydantic import BaseModel, Field, model_validator, ValidationError
import logging
from typing import Dict, Any, Optional, Union, Set
from pydantic import BaseModel, Field, model_validator






class Model:

    counts: Dict[str, int] = Field(default_factory=lambda: {
        "diffusion": 0, "lora": 0, "vae": 0, "x4mer": 0, "taesd": 0, "llm": 0
    })  # Holds counts for various block types

    dtype: Set[str, str] = Field(default_factory=set)
    tensor_parameters: int = Field(default_factory=int)


class Block(Model):

    def __init__(self, current_tensor):
        self.blocks = current_tensor

    def max_count(self):
        return max(self.counts , key=lambda k: len(self.counts[k]))

    def label(self, model):
        for index, key, value in enumerate(model.items()):
            if "model.diffusion_model" in key: # diffusion models
                self.blocks.counts += 1
        # etc
            if isinstance(value, torch.Tensor):
                self.blocks.dtype
            self.blocks.tensor_parameters = index

1024



for current_tensor in tensor_stack:
    File1 = StateDict(current_tensor)


# class BlockData(BaseModel):
#     """
#     This model represents the data structure that holds both counts and detailed information.
#     """
#     counts: Dict[str, int] = Field(default_factory=lambda: {
#         "diffusion": 0, "lora": 0, "vae": 0, "x4mer": 0, "taesd": 0, "llm": 0
#     })  # Holds counts for various block types

#     details: Dict[str, Union[str, Dict[str, Any]]] = Field(default_factory=dict)  # Holds block details like dtype, len(value), etc.

#     # Example extra field that could be added for tensor metadata
#     tensor_count: int = Field(default_factory=int)
#     dtype: Dict[str, Any] = Field(default_factory=dict)

#     @model_validator(mode="after")
#     def check_dtype(cls, values):
#         """
#         This method validates that certain keys exist in the data (e.g., dtype).
#         """
#         model_dtype = values.get('dtype')
#         if not model_dtype:
#             raise ValueError("dtype cannot be empty")
#         return values

#     def tally_block(self, block_name: str):
#         """
#         Method to increment the tally for a specific block.
#         """
#         if block_name in self.counts:
#             self.counts[block_name] += 1
#         else:
#             raise ValueError(f"Block '{block_name}' not found in counts.")

#     def add_detail(self, block_name: str, detail: Any):
#         """
#         Method to add details for a specific block.
#         """
#         self.details[block_name] = detail

# class BlockIndexMeta(type):
#     def __new__(cls, name, bases, attrs):
#         cls_obj = super().__new__(cls, name, bases, attrs)
#         cls_obj.logger = logging.getLogger(name)
#         return cls_obj

# class BlockIndex(metaclass=BlockIndexMeta):

#     def main(self, file_name: str, path: str):  # Heavy-lifting method
#         self.logger.info(f"Loading: {os.path.basename(file_name)}")
#         self.extension = Path(file_name).suffix.lower()
#         raw_blocks = defaultdict(dict)

#         if self.extension in [".safetensors", ".sft"]:
#             raw_blocks = self.load_safetensors(file_name)
#         elif self.extension == ".gguf":
#             raw_blocks = self.load_gguf(file_name)
#         elif self.extension in [".pt", ".pth"]:
#             raw_blocks = self.load_pickletensor(file_name)
#         else:
#             raw_blocks = False

#         if raw_blocks:  #only return parse results if there were blocks to be found
#             block_match = self._search_dict(raw_blocks)
#             send_to_file = defaultdict(dict)

#             block_match_comparison = defaultdict(dict)
#             for key in block_match.keys():
#                 block_match_comparison[key] = block_match[key]
#                 self.logger.info(f"Number of {key} modules: {len(block_match[key])}")

#             block_max = "Unknown model type!"
#             if len(block_match_comparison):
#                 if block_match.get("TAESD VAE", 0) == len(raw_blocks):
#                     block_max = "TAESD VAE"
#                 else:
#                     block_max = max(block_match_comparison, key=lambda k: len(block_match_comparison[k]))

#             # Create BlockData model instance to validate data
#             try:
#                 block_data_model = BlockData(
#                     data=send_to_file["data"],
#                     dtype=list(block_match.dtype),
#                     block_max=block_max
#                 )
#                 filename = path + os.path.basename(file_name) + ".json"
#                 self.save_to_file(filename, block_data_model)
#             except ValidationError as e:
#                 self.logger.error(f"Data validation failed: {e}")
#             else:
#                 self.logger.info(f"Total module number: {len(raw_blocks)}")
#                 self.logger.info(f"Dtypes detected: {block_data_model.dtype}")
#                 self.logger.info(f"Model detected: {block_max}")
#         else:
#             self.logger.info("No Data.")


#             # Example usage:
#             block_data = BlockData()

#             # Increment block counts
#             block_data.tally_block("diffusion")
#             block_data.tally_block("lora")

#             # Add details (could include length, dtype, or any other block-specific information)
#             block_data.add_detail("diffusion", {"len": 100, "dtype": "float32"})
#             block_data.add_detail("lora", {"len": 50, "dtype": "int64"})

#             print(block_data.json(indent=2))



#     def load_safetensors(self, file_name: str) -> dict:
#         from safetensors.torch import load_file
#         try:
#             return load_file(file_name)
#         except Exception as error_log:
#             self.error_handler("retry", error_log, ".safetensors", file_name)
#             try:
#                 with open(file_name, "rb") as file:
#                     header = struct.unpack("<Q", file.read(8))[0]
#                     return json.loads(file.read(header), object_hook=self._search_dict)
#             except Exception as error_log:
#                 self.error_handler("fail", error_log, self.extension, file_name)
#         return {}

#     def load_gguf(self, file_name: str) -> dict:
#         from gguf import GGUFReader
#         try:
#             reader = GGUFReader(file_name)
#             raw_blocks = reader.fields
#             for tensor in reader.tensors:
#                 raw_blocks.setdefault(tensor.name, f"{tensor.data_offset}{tensor.shape}")
#             return raw_blocks
#         except Exception as error_log:
#             self.error_handler("retry", error_log, self.extension, file_name)
#             try:
#                 from llama_cpp import Llama
#                 with open(file_name, "rb") as file_obj:
#                     magic = file_obj.read(4)
#                     if magic != b"GGUF":
#                         logger.debug(f"Invalid GGUF magic number in '{file_name}'")
#                         raw_blocks = False
#                         return
#                     else:
#                         version = struct.unpack("<I", file_obj.read(4))[0]
#                         if version < 2:
#                             logger.debug(f"Unsupported GGUF version {version} in '{file_name}'")
#                             raw_blocks = False
#                             return
#                         else:
#                             parser = Llama(model_path=file_name, vocab_only=True, verbose=False)
#                             raw_blocks = parser.metadata
#             except ValueError as error_log: # We tried...
#                 self.error_handler("fail", error_log, self.extension , file_name)
#                 raw_blocks = False
#         return {}

#     def load_pickletensor(self, file_name: str) -> dict:
#         try:
#             return torch.load(file_name, map_location="cpu")
#         except Exception as error_log:
#             self.error_handler("retry", error_log, ".pt/.pth", file_name)
#         return {}

#     def save_to_file(self, filename: str, block_data_model: BlockData):
#         with open(filename, "w", encoding="UTF-8") as index:
#             json.dump(block_data_model.model_dump(), index, ensure_ascii=False, indent=4, sort_keys=True)

#     def error_handler(self, nature: str, error_log: str, identity: str = None, reference: str = None):
#         if nature == "retry":
#             self.logger.debug(f"Error reading {reference}, switching read method for '{identity}': {error_log}")
#         elif nature == "fail":
#             self.logger.debug(f"{reference} parse attempts exhausted for '{identity}' [{error_log}]")

#     def _search_dict(self, model: dict) -> dict:
#         block_match = BlockMatch()
#         for key, value in model.items():
#             if "model.diffusion_model" in key: # diffusion models
#                 block_match.diffusion.add(key)
#             elif "general.architecture" in key: #llm model key
#                 block_match.llm.add(key)
#             elif "lora_down" in key or "lora_up" in key: # lora model key
#                 block_match.lora.add(key)
#             elif "decoder.up" in key or "encoder.down" in key: # vae model key
#                 block_match.vae.add(key)
#             elif "decoder.layers" in key or "encoder.layers" in key: #taesd specific
#                 block_match.taesd.add(key)
#             elif "self_attention" in key or "SelfAttention" in key or "self_attn" in key: # x4mer chekc
#                 block_match.x4mer.add(key)
#             if isinstance(value, torch.Tensor):
#                 dtype = f"{str(tuple(value.size())).replace(', ', '-')}"
#                 match = model.get(key).real.real.dtype
#                 block_match.dtype.add(match)
#                 block_data.tensor_count +=1
#             else:
#                 block_data = str(value)
#         return block_match.model_dump()

# if __name__ == "__main__":
#     # Initial logger setup
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     file = "c:\\Users\\Public\\models\\image\\"  # Replace with actual file path
#     blocks = BlockIndex()
#     save_location = "C:\\Users\\Public\\code\\metadata_scans_v2"  # Replace with actual save location
#     if Path(file).is_dir():
#         path_data = os.listdir(file)
#         for each in path_data:
#             file_path = os.path.join(file, each)
#             blocks.main(file_path, save_location)
#     else:
#         blocks.main(file, save_location)




open the file
get