import os
import struct
import json
import torch
import re
import sys
from tqdm.auto import tqdm
from pathlib import Path
from collections import defaultdict, Counter

from sdbx import logger
from sdbx.config import config

class Domain:
    """Represents a top-level domain like nn, info, or dev."""

    def __init__(self, domain_name):
        self.domain_name = domain_name
        self.architectures = {}  # Achitecture objects

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
            self.components = {}  # Component objects

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
    # 重みを確認するモデルファイル
    def main(self, file_name: str, path: str):

        self.id_values = defaultdict(dict)
        file_suffix = Path(file_name).suffix
        if file_suffix == "": return
        self.id_values["extension"] = Path(file_name).suffix.lower() # extension is metadata
        model_header = defaultdict(dict)

        # Process file by method indicated by extension, usually struct unpacking, except for pt files which are memmap
        if self.id_values["extension"] in [".safetensors", ".sft"]: model_header: dict = self.__unsafetensors(file_name, self.id_values["extension"])
        elif self.id_values["extension"] == ".gguf": model_header: dict = self.__ungguf(file_name, self.id_values["extension"])
        elif self.id_values["extension"] in [".pt", ".pth"]: model_header: dict = self.__unpickle(file_name, self.id_values["extension"])


        if model_header:
            self.neural_net = Domain("nn") #create the domain only when we know its a model

            self.MODEL_FORMAT = config.get_default("tuning","model_format")
            self.PDXL_FORMAT  = config.get_default("tuning","pdxl_format")

            self.id_values["tensors"] = len(model_header)
            self.block_count(model_header, self.MODEL_FORMAT) # Check for matches in the keys of model_header against MODEL_FORMAT
            self._pretty_output(file_name)
            filename = os.path.join(path, os.path.basename(file_name) + ".json")
            with open(filename, "w", encoding="UTF-8") as index: # todo: make 'a' type before release
                data = self.id_values | model_header
                json.dump(data ,index, ensure_ascii=False, indent=4, sort_keys=True)

    def error_handler(self, kind:str, error_log:str, obj_name:str=None, error_source:str=None):
        if kind == "retry":
            self.id_progress("Error reading metadata, switching read method")
        elif kind == "fail":
            self.id_progress("Metadata read attempts exhasted for:'", obj_name)
        logger.debug(f"Could not read : '{obj_name}' In {error_source}: {error_log}", exc_info=True)
        return

#SAFETENSORS
    def __unsafetensors(self, file_name:str, extension: str):
        self.id_values["extension"] = "safetensors"
        self.id_values["file_size"] = os.path.getsize(file_name)
        with open(file_path, 'rb') as file:
            try:
                first_8_bytes    = file.read(8)
                length_of_header = struct.unpack('<Q', first_8_bytes)[0]
                header_bytes     = file.read(length_of_header)
                header           = json.loads(header_bytes.decode('utf-8'))
                if header.get("__metadata__",0 ) != 0:  # we want to remove this metadata so its not counted as tensors
                    header.pop("__metadata__")  # it is usally empty on safetensors ._.
                return header
            except Exception as error_log:  #couldn't open file
                self.error_handler(kind="fail", error_log=error_log, obj_name=file_name, error_source=extension)

# GGUF
    def __ungguf(self, file_name:str, extension:str):
        self.id_values["file_size"] = os.path.getsize(file_name) # how big will be important for memory management
        file_data = defaultdict(dict)
        from llama_cpp import Llama
        try:
            with open(file_name, "rb") as file:
                magic = file.read(4)
                if magic != b"GGUF":
                    logger.debug(f"Invalid GGUF magic number in '{file_name}'") # uh uh uh, you didn't say the magic word
                    return
                version = struct.unpack("<I", file.read(4))[0]
                if version < 2:
                    logger.debug(f"Unsupported GGUF version {version} in '{file_name}'")
                    return
            parser                  = Llama(model_path=file_name, vocab_only=True, verbose=False) #  fails image quants, but dramatically faster vs ggufreader
            arch                    = parser.metadata.get("general.architecture") # with gguf we can directly request the model name but it isnt always written in full
            name                    = parser.metadata.get("general.name") # sometimes this field is better than arch
            self.id_values["name"]  = name if name is not None else arch
            self.id_values["dtype"] = parser.scores.dtype.name #outputs as full name eg: 'float32 rather than f32'
            return # todo: handle none return better
        except ValueError as error_log:
            self.error_handler(kind="retry", error_log=error_log, obj_name=file_name, error_source=extension) # the aforementioned failing
        from gguf import GGUFReader
        try: # method using gguf library, better for LDM conversions
            reader                  = GGUFReader(file_name, 'r')
            self.id_values["dtype"] = reader.data.dtype.name # get dtype from metadata
            arch                    = reader.fields["general.architecture"] # model type category, usually prevents the need  toblock scan for llms
            self.id_values["name"]  = str(bytes(arch.parts[arch.data[0]]), encoding='utf-8') # retrieve model name from the dict data
            if len(arch.types) > 1:
                self.id_values["name"] = arch.types #if we get a result, save it
            for tensor in reader.tensors:
                file_data[str(tensor.name)] = {"shape": str(tensor.shape), "dtype": str(tensor.tensor_type.name)} # create dict similar to safetensors/pt results
            return file_data
        except ValueError as error_log:
            self.error_handler(kind="fail", error_log=error_log, obj_name=file_name, error_source=extension) # >:V

# PICKLETENSOR FILE
    def __unpickle(self, file_name:str, extension:str):
        self.id_values["file_size"] = os.path.getsize(file_name) #
        import mmap
        import pickle
        try:
            return torch.load(file_name, map_location="cpu") #this method seems outdated
        except TypeError as error_log:
            self.error_handler(kind="retry", error_log=error_log, obj_name=file_name, error_source=extension)
            try:
                with open(file_name, "r+b") as file_obj:
                    mm = mmap.mmap(file_obj.fileno(), 0)
                    return pickle.loads(memoryview(mm))
            except Exception as error_log: #throws a _pickle error (so salty...)
                self.error_handler(kind="fail", error_log=error_log, obj_name=file_name, error_source=extension)

    def block_count(self, source_data:dict, format_dict:dict):
        """
        Generalized function to check either the keys or values of a source dictionary
        against a format dictionary.
        """
        target_data    = source_data.keys()  # The model dict keys to process
        flattened_data = ' '.join(map(str, target_data)) # Convert the key list to a string for easier processing
        for label, known_block in format_dict.items(): #  Begin to count all occurrences of each known key inside the flattened model data
            known_block = known_block if isinstance(known_block, list) else [known_block] # Normalize known_blocks as list
            for list_string in known_block:  # Process tuning.json as either regex or literal string
                list_string = str(list_string)
                if list_string.startswith("r'"): # Regex conversion
                    raw_block_name = (list_string
                        .replace("d+", r"\d+")  # Replace 'd+' with '\d+' for digits
                        .replace(".", r"\.")    # Escape literal dots with '\.'
                        .strip("r'")            # Strip the 'r' and quotes from the string
                    )
                    block_name = re.compile(raw_block_name)
                    match      = block_name.search(flattened_data)
                    if match:  # If a match is found, Find the corresponding key in source_data by searching for the actual key substring
                        matched_key = next((key for key in source_data.keys() if block_name.search(key)), None)
                        self.block_details(source_data[matched_key], label)  # Send matched key's data to block_details
                else: # When our block string is not regex
                     if list_string in flattened_data:  # Check for the string directly in flattened_data
                        matched_key = next((key for key in source_data.keys() if list_string in key), None)
                        self.block_details(source_data[matched_key], label)  # Send matched key's data to block_details

    def block_details(self, model_header:dict, label: str):
        self.id_values[label] = self.id_values.get(label, 0) + 1  # Increment the count for the match
        search_items = ["dtype", "shape"]
        for field_name in search_items:
            field_value = model_header.get(field_name)
            if field_value:
                if isinstance(field_value, list):
                    field_value = str(field_value).strip("\r").strip("\n")  # We only need the first two numbers of 'shape'
                if field_value not in self.id_values.get(field_name,""):
                    self.id_values[field_name] = " ".join([self.id_values.get(field_name, ""),field_value]) # Prevent data duplication

    def _pretty_output(self, file_name): #pretty printer
        key_value_length = len(self.id_values.values())  # number of items detected in the scan
        info_format      = "{:<1} | " * key_value_length # shrink print columns to data width
        value            = tuple(self.id_values.keys()) # use to create table
        horizontal_bar   = ("  " + "-" * (10*key_value_length)) # horizontal divider of arbitrary length. could use shutil to dynamically create but eh. already overkill
        formatted_data   = tuple(self.id_values.values()) # data extracted from the scan
        self.id_progress(file_name, info_format.format(*value), horizontal_bar, info_format.format(*formatted_data)) #send to print function

    def id_progress(*args):
        sys.stdout.write("\033[F" *len(args))  # ANSI escape codes to move the cursor up 3 lines
        for line_data in args:
            sys.stdout.write(" " * 175 + "\x1b[1K\r")
            sys.stdout.write(f"{line_data}\n")  # Print the lines
        sys.stdout.flush()              # Empty output buffer to ensure the changes are shown


if __name__ == "__main__":
    file = config.get_path("models.image")
    blocks = BlockIndex()
    save_location = os.path.join(config.get_path("models.dev"),"metadata")
    if Path(file).is_dir() == True:
        path_data = os.listdir(file)
        print("\n\n\n\n")
        for each_file in tqdm(path_data, total=len(path_data), position=0, leave=True):
            file_path = os.path.join(file,each_file)
            blocks.main(file_path, save_location)
    else:
        blocks.main(file, save_location)
