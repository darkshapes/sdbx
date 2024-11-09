import os
import struct
import json
import torch
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from math import isclose
from functools import reduce
import hashlib

from tqdm.auto import tqdm

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


class MatchMetadata:

    def extract_tensor_data(self, source_data_item, id_values):
        """
        Extracts shape and key data from the source data.
        This would extract whatever additional information is needed when a match is found.
        """
        TENSOR_TOLERANCE = 4e-2
        search_items = ["dtype", "shape"]
        for field_name in search_items:
            field_value = source_data_item.get(field_name)
            if field_value:
                if isinstance(field_value, list):
                    field_value = str(field_value)  # We only need the first two numbers of 'shape'
                if field_value not in id_values.get(field_name,""):
                    id_values[field_name] = " ".join([id_values.get(field_name, ""),field_value]).lstrip() # Prevent data duplication

        return {
            "tensors": id_values.get("tensors", 0),
            'shape': id_values.get('shape', 0),
        }

    def find_matching_metadata(self, known_values, source_data, id_values, depth=[]):
        """
        Recursively traverse the criteria and source_data to check for matches.
        Track comparisons with id_values, using a list to track the recursion depth.

        known_values: the original hierarchy of known values from a .json file
        source_data: the original model state dict
        id_values: information matching our needs extracted from the source data
        depth: current level inside the dict
        """

        id_values = id_values

        # Get the dict position indicated in depth
        get_nested = lambda d, keys: reduce(lambda d, key: d.get(key, None) if isinstance(d, dict) else None, keys, d)
        # Return the previous position indicated in depth
        backtrack_depth = lambda depth: depth[:-1] if depth and depth[-1] in ["block_names", "tensors", "shape", "file_size", "hash"] else depth

        def advance_depth(depth: list, lateral: bool = False) -> list:
            """
            Attempts to advance through the tuning dict laterally (to the next key at the same level),
            failing which retraces vertically (first to the parent level, then the next key at parent level).
            """
            if not depth:
                return None  # Stop if we've reached the root or there's no further depth
            parent_dict = get_nested(known_values, depth[:-1]) # Prior depth

            if not isinstance(parent_dict, dict):    # We look for dicts, and no other types
                return None  # Invalid state if we can't get the parent dict

            parent_keys = list(parent_dict.keys())  #  Keys from above
            previous_depth = depth[-1] # Current level

            if previous_depth in parent_keys: # Lateral movement check
                current_index = parent_keys.index(previous_depth)

                if current_index + 1 < len(parent_keys): #Lateral/next movement, same level
                    new_depth = depth[:-1]  # Get the parent depth
                    new_depth.append(parent_keys[current_index + 1])  # Add the next key in sequence
                    return new_depth

            if len(depth) > 1: # If no lateral movement is possible, try  vertical/backtracking if there is more than one level
                return advance_depth(depth[:-1])  # Move to parent and retry

            return None  # Traversal complete

        criteria = get_nested(known_values, depth)
        if criteria is None:  # Cannot advance, stop
            return id_values

        if isinstance(criteria, str): criteria = [criteria]
        if isinstance(criteria, dict):
            for name in criteria: # Descend dictionary structure
                depth.append(name) # Append the current name to depth list
                self.find_matching_metadata(known_values, source_data, id_values, depth)
                if depth is None:  # Cannot advance, stop
                    return id_values
                else:
                    depth = backtrack_depth(depth)
                    current_depth = get_nested(known_values, depth)
                    if current_depth[-1] ==
                        if len(current_depth) == id_values.get(depth[-1],0):
                            id_values.get("type", set()).add(depth[-1])

                        known_values[next(iter(known_values), "nn")].pop(depth[-1])
                    advance_depth(depth)
                    self.find_matching_metadata(known_values, source_data, id_values, depth)
            return id_values

        elif isinstance(criteria, list): # when fed correct datatype, we check for matches
            for checklist in criteria:
                if not isinstance(checklist, list): checklist = [checklist]  # normalize scalar to list
                for list_entry in checklist: # the entries to match
                    if depth[-1] == "hash":
                         id_values["hash"] = hashlib.sha256(open(id_values["file_name"],'rb').read()).hexdigest()
                    list_entry = str(list_entry)
                    if list_entry.startswith("r'"): # Regex conversion
                        expression = (list_entry
                            .replace("d+", r"\d+")  # Replace 'd+' with '\d+' for digits
                            .replace(".", r"\.")    # Escape literal dots with '\.'
                            .strip("r'")            # Strip the 'r' and quotes from the string
                        )
                        regex_entry = re.compile(expression)
                        match = next((regex_entry.search(k) for k in source_data), False)
                    else:
                        match = next((k for k in source_data if list_entry in k), False)
                    if match: # Found a match, based on keys
                        previous_depth = depth[-1]
                        depth = backtrack_depth(depth)
                        found = depth[-1] if depth else "unknown"    # if theres no header or other circumstances
                        id_values[found] = id_values.get(found, 0) + 1

                        shape_key_data = self.extract_tensor_data(source_data[match], id_values)
                        id_values.update(shape_key_data)

                        depth.append(previous_depth) #if length depth = 2
                        depth = advance_depth(depth, lateral=True)
                        if depth is None:  # Cannot advance, stop
                            return id_values
                        self.find_matching_metadata(known_values, source_data, id_values, depth)  # Recurse


            return id_values


class BlockIndex:
    # 重みを確認するモデルファイル
    def main(self, file_name: str, path: str):

        self.id_values = defaultdict(dict)
        file_suffix = Path(file_name).suffix
        if file_suffix == "": return
        self.id_values["file_name"] = file_name
        self.id_values["extension"] = Path(file_name).suffix.lower() # extension is metadata
        model_header = defaultdict(dict)

        # Process file by method indicated by extension, usually struct unpacking, except for pt files which are memmap
        if self.id_values["extension"] in [".safetensors", ".sft"]: model_header: dict = self.__unsafetensors(file_name, self.id_values["extension"])
        elif self.id_values["extension"] == ".gguf": model_header: dict = self.__ungguf(file_name, self.id_values["extension"])
        elif self.id_values["extension"] in [".pt", ".pth"]: model_header: dict = self.__unpickle(file_name, self.id_values["extension"])

        if model_header:
            self.neural_net = Domain("nn") #create the domain only when we know its a model

            self.MODEL_FORMAT = config.get_default("tuning","formats")
            self.id_values["tensors"] = len(model_header)
            instance = MatchMetadata()
            self.id_values = instance.find_matching_metadata(known_values=self.MODEL_FORMAT, source_data=model_header, id_values=self.id_values)

            self._pretty_output(file_name)
            filename = os.path.join(path, os.path.basename(file_name) + ".json")
            with open(filename, "w", encoding="UTF-8") as index: # todo: make 'a' type before release
                data = self.id_values | model_header
                json.dump(data ,index, ensure_ascii=False, indent=4, sort_keys=False)

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
                    header.pop("__metadata__")  # it is usually empty on safetensors ._.
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

    def _pretty_output(self, file_name): #pretty printer
        print_values = self.id_values.copy()
        if (k := next(iter(print_values), None)) is not None:
            print_values.pop(k)  # Only pop if a valid key is found
        key_value_length = len(print_values)  # number of items detected in the scan
        info_format      = "{:<5} | " * key_value_length # shrink print columns to data width
        header_keys      = tuple(print_values) # use to create table
        horizontal_bar   = ("  " + "-" * (10*key_value_length)) # horizontal divider of arbitrary length. could use shutil to dynamically create but eh. already overkill
        formatted_data   = tuple(print_values.values()) # data extracted from the scan
        return self.id_progress(self.id_values.get("file_name", None), info_format.format(*header_keys), horizontal_bar, info_format.format(*formatted_data)) #send to print function

    def id_progress(self, *formatted_data):
        sys.stdout.write("\033[F" * len(formatted_data))  # ANSI escape codes to move the cursor up 3 lines
        for line_data in formatted_data:
            sys.stdout.write(" " * 175 + "\x1b[1K\r")
            sys.stdout.write(f"{line_data}\r\n")  # Print the lines
        sys.stdout.flush()              # Empty output buffer to ensure the changes are shown

if __name__ == "__main__":
    file = config.get_path("models.dev")
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
