import os
import struct
import json
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter

from sdbx import logger
from sdbx.config import config

class Domain:
    """Represents a top-level domain like nn, info, or dev."""

    def __init__(self, domain_name):
        self.domain_name = domain_name
        self.architectures = {}  # Stores Architecture objects

    def add_architecture(self, architecture_name, architecture_obj):
        self.architectures[architecture_name] = architecture_obj

    def flatten(self, prefix):
        """Flattens the block format to a dict."""
        flat_dict = {}
        for arc_name, arc_obj in self.architectures.items():
            path = f"{prefix}.{arc_name}"
            flat_dict.update(arc_obj.flatten(path))
        return flat_dict


    class Architecture:
        """Represents model architecture like sdxl, flux."""

        def __init__(self, architecture):
            self.architecture = architecture
            self.components = {}  # Stores Component objects

        def add_component(self, component_name, component_obj):
            self.components[component_name] = component_obj

        def flatten(self, prefix):
            """Flattens the architecture to a dict."""
            flat_dict = {}
            for comp_name, comp_obj in self.components.items():
                path = f"{prefix}.{comp_name}"
                flat_dict[path] = comp_obj.to_dict()
            return flat_dict

        class Component:
            """Represents individual model components like vae, lora, unet."""
            def __init__(self, component_name, **kwargs):
                self.component_name = component_name

                if 'dtype' in kwargs: self.dtype = kwargs['dtype']
                if 'file_size' in kwargs: self.file_size = kwargs['file_size']
                if 'distinction' in kwargs: self.distinction = kwargs['distinction']

            def to_dict(self):
                """Serializes the Component object to a dictionary."""
                return {
                    'component_name': self.component_name,
                    'dtype': self.dtype,
                    'file_size': self.file_size,
                    'distinction': self.distinction
                }

class BlockIndex:
    def main(self, file_name: str, path: str): # 重みを確認するモデルファイル

        logger.info(f"loading: {os.path.basename(file_name)}")

        self.identifying_values = defaultdict(dict)
        file_suffix = Path(file_name).suffix
        if file_suffix == "": return
        self.identifying_values["extension"] = Path(file_name).suffix.lower()
        deserialized_model = defaultdict(dict)


        if self.identifying_values["extension"] in [".safetensors", ".sft"]: deserialized_model: dict = self.__unsafetensors(file_name, self.identifying_values["extension"])
        elif self.identifying_values["extension"] == ".gguf": deserialized_model: dict = self.__ungguf(file_name, self.identifying_values["extension"])
        elif self.identifying_values["extension"] in [".pt", ".pth"]: deserialized_model: dict = self.__unpickle(file_name, self.identifying_values["extension"])


        if deserialized_model:
            self.neural_net = Domain("nn")
            parsed_blocks = self.__skim_metadata(deserialized_model)


#SAFETENSORS
    def __unsafetensors(self, file_name, extension):
        self.identifying_values["extension"] = "safetensors"
        self.identifying_values["file_size"] = os.path.getsize(file_name)
        #from safetensors.torch import load_file
        with open(file_path, 'rb') as file:
            try:
                first_8_bytes = file.read(8)
                length_of_header = struct.unpack('<Q', first_8_bytes)[0]
                header_bytes = file.read(length_of_header)
                header = json.loads(header_bytes.decode('utf-8'))
                if header.get("__metadata__",0 ) != 0:
                    header.pop("__metadata__")
                return header
            except Exception as error_log:  #couldn't open file
                self.error_handler(kind="fail", error_log=error_log, identity=extension, reference=file_name)

# GGUF
    def __ungguf(self, file_name, extension):
        self.identifying_values["file_size"] = os.path.getsize(file_name)
        from gguf import GGUFReader
        try: # Method using gguf library, better for Latent Diffusion conversions
            reader = GGUFReader(file_name)
            file_data = defaultdict(dict)
            self.identifying_values["dtype"] = reader.data.dtype.name
            arch = reader.fields["general.architecture"]
            self.identifying_values["name"] = str(bytes(arch.parts[arch.data[0]]), encoding='utf-8')
            if len(arch.types) > 1:
                self.identifying_values["name"] = arch.types
            for tensor in reader.tensors:
                file_data[tensor.name] = f"{tensor.shape, tensor.data_offset}"
            return file_data
        except ValueError as error_log:
            self.error_handler(kind="fail", error_log=error_log, identity=extension, reference=file_name)

# PICKLETENSOR FILE
    def __unpickle(self, file_name, extension):
        self.identifying_values["file_size"] = os.path.getsize(file_name)
        import mmap
        import pickle
        try:
            return torch.load(file_name, map_location="cpu") #this method seems outdated
        except TypeError as error_log:
            self.error_handler(kind="retry", error_log=error_log, identity=extension, reference=file_name)
            try:
                with open(file_name, "r+b") as file_obj:
                    mm = mmap.mmap(file_obj.fileno(), 0)
                    return pickle.loads(memoryview(mm))
            except Exception as error_log: #throws a _pickle error
                self.error_handler(kind="fail", error_log=error_log, identity=extension, reference=file_name)

    def __skim_metadata(self, deserialized_model):

        MODEL_FORMATS    = config.get_default("tuning","model_formats")
        COMPVIS_FORMAT   = config.get_default("tuning","compvis_format")
        DIFFUSERS_FORMAT = config.get_default("tuning","diffusers_format")
        LLAMA_FORMAT     = config.get_default("tuning","llama_format")
        PDXL_FORMAT      = config.get_default("tuning","pdxl_format")

        key_string_list = MODEL_FORMATS | COMPVIS_FORMAT | DIFFUSERS_FORMAT | LLAMA_FORMAT

        self.identifying_values["tensors"] = len(deserialized_model)
        self.count_values                  = defaultdict(int)

        def count_matches(source_key, string_list):
            for label, block_key in string_list.items():
                block_key = block_key if isinstance(block_key, list) else [block_key]
                for each in block_key:
                    match_count = sum(str(each) in k for k in source_key)
                    if match_count != 0: self.count_values[label] = match_count

        count = count_matches(deserialized_model.keys(), key_string_list)
        count = deserialized_model.values()
        for value in count:
            if isinstance(value, dict):
                if value.get("dtype",0) != 0: self.identifying_values["dtype"] = value.get("dtype",0)
                offsets = value.get("data_offsets",0)
                if offsets != 0:
                    if offsets in PDXL_FORMAT["pdxl"]:
                        self.count_values["pdxl"] +=1


        data = self.identifying_values | self.count_values
        print(data)
        return data
    def error_handler(self, kind:str, error_log:str, identity:str=None, reference:str=None):
        if kind == "retry":
            logger.info(f"Error reading metadata,  switching read method for '{identity}' type  {reference}, [{error_log}]", exc_info=True)
        elif kind == "fail":
            logger.info(f"Metadata read attempts exhasted for:'{identity}' type {reference}  [{error_log}]")
        return


if __name__ == "__main__":

    file = config.get_path("models.image")
    blocks = BlockIndex()
    save_location = os.path.join(config.get_path("models.dev"),"metadata")
    if Path(file).is_dir() == True:
        path_data = os.listdir(file)
        for each in tqdm(path_data, total=len(path_data)):
            file_path = os.path.join(file,each)
            blocks.main(file_path, save_location)
    else:
        blocks.main(file, save_location)
