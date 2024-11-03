import os
import struct
import json
import torch
import re
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

        self.id_values = defaultdict(dict)
        file_suffix = Path(file_name).suffix
        if file_suffix == "": return
        self.id_values["extension"] = Path(file_name).suffix.lower()
        deserialized_model = defaultdict(dict)


        if self.id_values["extension"] in [".safetensors", ".sft"]: deserialized_model: dict = self.__unsafetensors(file_name, self.id_values["extension"])
        elif self.id_values["extension"] == ".gguf": deserialized_model: dict = self.__ungguf(file_name, self.id_values["extension"])
        elif self.id_values["extension"] in [".pt", ".pth"]: deserialized_model: dict = self.__unpickle(file_name, self.id_values["extension"])


        if deserialized_model:
            self.neural_net = Domain("nn")

            self.MODEL_FORMAT = config.get_default("tuning","model_format")
            self.PDXL_FORMAT  = config.get_default("tuning","pdxl_format")

            self.id_values["tensors"] = len(deserialized_model)
            self.block_count(deserialized_model, self.MODEL_FORMAT) # Check for matches in the keys of deserialized_model against MODEL_FORMAT
            if self.id_values.get("dtype") is None:
                self.block_details(deserialized_model) # Check for matches in the keys of deserialized_model against MODEL_FORMAT

            key_value_length = len(self.id_values.values())
            info_format = "{:<5} | " * key_value_length
            value = tuple(self.id_values.keys())
            print(info_format.format(*value))
            print("-" * 80)
            data = tuple(self.id_values.values())
            print(info_format.format(*data))
            filename = os.path.join(path, os.path.basename(file_name) + ".json")
            with open(filename, "w", encoding="UTF-8") as index: # todo: make 'a' type before release
                data = self.id_values | deserialized_model
                json.dump(data ,index, ensure_ascii=False, indent=4, sort_keys=True)

    def error_handler(self, kind:str, error_log:str, file_name:str=None, parse_type:str=None):
        if kind == "retry":
            logger.info(f"Error reading metadata, switching read method ", exc_info=False)
        elif kind == "fail":
            logger.info(f"Metadata read attempts exhasted for:'{file_name}'", exc_info=False)
        logger.debug(f"Could not read : '{file_name}' Recognized as {parse_type}: {error_log}", exc_info=True)
        return

#SAFETENSORS
    def __unsafetensors(self, file_name:str, extension: str):
        self.id_values["extension"] = "safetensors"
        self.id_values["file_size"] = os.path.getsize(file_name)
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
                self.error_handler(kind="fail", error_log=error_log, file_name=file_name, parse_type=extension)

# GGUF
    def __ungguf(self, file_name:str, extension:str):
        self.id_values["file_size"] = os.path.getsize(file_name)
        file_data = defaultdict(dict)
        from llama_cpp import Llama
        try:
            with open(file_name, "rb") as file:
                magic = file.read(4)
                if magic != b"GGUF":
                    logger.debug(f"Invalid GGUF magic number in '{file_name}'")
                    return
                version = struct.unpack("<I", file.read(4))[0]
                if version < 2:
                    logger.debug(f"Unsupported GGUF version {version} in '{file_name}'")
                    return
            parser = Llama(model_path=file_name, vocab_only=True, verbose=False)
            arch = parser.metadata.get("general.architecture")
            name = parser.metadata.get("general.name")
            self.id_values["name"] = name if name is not None else arch
            self.id_values["dtype"] = parser.scores.dtype.name #outputs as full name eg: 'float32 rather than f32'
            return
        except ValueError as error_log:
            self.error_handler(kind="retry", error_log=error_log, file_name=file_name, parse_type=extension)
        from gguf import GGUFReader
        try: # Method using gguf library, better for LDM conversions
            reader = GGUFReader(file_name, 'r')
            self.id_values["dtype"] = reader.data.dtype.name
            arch = reader.fields["general.architecture"]
            self.id_values["name"] = str(bytes(arch.parts[arch.data[0]]), encoding='utf-8')
            if len(arch.types) > 1:
                self.id_values["name"] = arch.types
            for tensor in reader.tensors:
                file_data[str(tensor.name)] = f"{str(tensor.shape), str(tensor.tensor_type.name)}"
            return file_data
        except ValueError as error_log:
            self.error_handler(kind="fail", error_log=error_log, file_name=file_name, parse_type=extension)

# PICKLETENSOR FILE
    def __unpickle(self, file_name:str, extension:str):
        self.id_values["file_size"] = os.path.getsize(file_name)
        import mmap
        import pickle
        try:
            return torch.load(file_name, map_location="cpu") #this method seems outdated
        except TypeError as error_log:
            self.error_handler(kind="retry", error_log=error_log, file_name=file_name, parse_type=extension)
            try:
                with open(file_name, "r+b") as file_obj:
                    mm = mmap.mmap(file_obj.fileno(), 0)
                    return pickle.loads(memoryview(mm))
            except Exception as error_log: #throws a _pickle error
                self.error_handler(kind="fail", error_log=error_log, file_name=file_name, parse_type=extension)

    def block_count(self, source_data:dict, format_dict:dict):
        """
        Generalized function to check either the keys or values of a source dictionary
        against a format dictionary. If check_values is True, match against the values, otherwise keys.
        """
        target_data    = source_data.keys()  # model dict to process
        flattened_data = ' '.join(map(str, target_data)) # list to string for easier processing
        for label, block_key in format_dict.items(): #  count occurrences of each block_key in the flattened data
            block_key = block_key if isinstance(block_key, list) else [block_key]
            for each in block_key:
                if str(each).startswith("r'"):
                    raw_block_name = (str(each)
                        .replace("d+", r"\d+")  # Replace 'd+' with '\d+' for digits
                        .replace(".", r"\.")    # Escape literal dots with '\.'
                        .strip("r'")            # Strip the 'r' and quotes from the string
                    )
                    block_name = re.compile(raw_block_name)
                    total_matches = len(list(block_name.finditer(flattened_data)))
                else:
                    total_matches = flattened_data.count(str(each))
                if total_matches > 0:
                    self.id_values[label] = self.id_values.get(label, 0) + total_matches # Update count_values with the result

    def block_details(self, deserialized_model:dict):
        for value in deserialized_model.values():         # Additionally, handle any specific logic for "dtype" and "data_offsets"
            self.id_values["dtype"] = []
            self.id_values["shape"] = []
            if isinstance(value, dict):
                search_items = ["dtype", "shape"]
                for field_name in search_items:
                    field_value = value.get(field_name)
                    if isinstance(field_value, list):
                        field_value = ", ".join(map(str, field_value))
                    elif field_value is not None:
                        field_value = str(field_value)
                    self.id_values[field_name].append(field_value)
                    self.id_values[field_name] = self.id_values[field_name][0]

                # if str(value.get("data_offsets")) in self.PDXL_FORMAT.get("pdxl"):
                #         self.id_values["pdxl"] = self.id_values.get("pdxl", 0) + 1

if __name__ == "__main__":
    file = config.get_path("models.image")
    blocks = BlockIndex()
    save_location = os.path.join(config.get_path("models.dev"),"metadata")
    if Path(file).is_dir() == True:
        path_data = os.listdir(file)
        for each_file in tqdm(path_data, total=len(path_data)):
            file_path = os.path.join(file,each_file)
            blocks.main(file_path, save_location)
    else:
        blocks.main(file, save_location)
