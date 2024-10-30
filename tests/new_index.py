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


import json

class Domain:
    """Represents a top-level domain like nn, info, or dev."""

    def __init__(self, domain_name):
        self.domain_name = domain_name
        self.block_formats = {}  # Stores BlockFormat objects

    def add_block_format(self, block_format_name, block_format_obj):
        self.block_formats[block_format_name] = block_format_obj

    def flatten(self, prefix=""):
        """Flattens the hierarchy into a dict with path as the key."""
        flat_dict = {}
        for bf_name, bf_obj in self.block_formats.items():
            path = f"{prefix}.{bf_name}" if prefix else bf_name
            flat_dict.update(bf_obj.flatten(path))
        return flat_dict

    class BlockFormat:
        """Represents the block format like compvis, diffusers."""

        def __init__(self, block_format):
            self.block_format = block_format
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

                def __init__(self, component_name, dtype, file_size, distinction):
                    self.component_name = component_name
                    self.dtype = dtype
                    self.file_size = file_size
                    self.distinction = distinction

                def to_dict(self):
                    """Serializes the Component object to a dictionary."""
                    return {
                        'component_name': self.component_name,
                        'dtype': self.dtype,
                        'file_size': self.file_size,
                        'distinction': self.distinction
                    }

# Example Usage

# Create a domain (e.g., nn)
nn_domain = Domain("nn")

# Create block formats (e.g., compvis, diffusers)
compvis_format = Domain.BlockFormat("compvis")
diffusers_format = Domain.BlockFormat("diffusers")

# Create architectures (e.g., sdxl, flux)
sdxl_architecture = Domain.BlockFormat.Architecture("sdxl")
flux_architecture = Domain.BlockFormat.Architecture("flux")

# Create components with attributes (e.g., vae, lora, unet)
vae_component = Domain.BlockFormat.Architecture.Component("vae", "fp16", "50MB", "LCM")
unet_component = Domain.BlockFormat.Architecture.Component("unet", "fp32", "120MB", "INSTRUCT")
flux_encoder = Domain.BlockFormat.Architecture.Component("encoder", "fp32", "70MB", "T5")

# Build the hierarchy
sdxl_architecture.add_component("vae", vae_component)
sdxl_architecture.add_component("unet", unet_component)
flux_architecture.add_component("encoder", flux_encoder)

compvis_format.add_architecture("sdxl", sdxl_architecture)
diffusers_format.add_architecture("flux", flux_architecture)

nn_domain.add_block_format("compvis", compvis_format)
nn_domain.add_block_format("diffusers", diffusers_format)

# Flatten the domain object to a dictionary with path as the key
nn_domain_flat = nn_domain.flatten()

# Print flattened structure
print(json.dumps(nn_domain_flat, indent=4))

# Write to JSON file
with open('index.json', 'w') as f:
    json.dump(nn_domain_flat, f, indent=4)

def rebuild_hierarchy(flat_dict):
    """Rebuilds the nested hierarchy from a flat dict."""
    nested_dict = {}

    for full_path, component_data in flat_dict.items():
        parts = full_path.split('.')
        current = nested_dict

        # Traverse or create the structure
        for part in parts[:-1]:  # All parts except the last (component)
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final component
        current[parts[-1]] = component_data

    return nested_dict

# Example usage of rebuilding
nested_hierarchy = rebuild_hierarchy(nn_domain_flat)
print(json.dumps(nested_hierarchy, indent=4))





class BlockIndex:
    def main(self, file_name: str, path: str): # 重みを確認するモデルファイル

        logger.info(f"loading: {os.path.basename(file_name)}")

        extension = Path(file_name).suffix.lower()
        deserialized_model = defaultdict(dict)
        self.identifying_values = defaultdict(dict)

        #infer, indicate, hint

        if extension in [".safetensors", ".sft"]: deserialized_model: dict = self.__unsafetensors(file_name, extension)
        elif extension == ".gguf": deserialized_model: dict = self.__ungguf(file_name, extension)
        elif extension in [".pt", ".pth"]: deserialized_model: dict = self.__unpickle(file_name, extension)


        if len(deserialized_model) != 0:
            self.neural_net = Domain("nn")
            self.__skim_metadata(deserialized_model)


        # self._search_dict(deserialized_model)

#SAFETENSORS
    def __unsafetensors(self, file_name, extension):
        from safetensors.torch import load_file
        try:
            return load_file(file_name)
        except Exception as error_log:  # not gonna do it
            self.error_handler(kind="retry", error_log=error_log, identity=extension, reference=file_name)
            try: #
                with open(self.path, "rb") as file:
                    header_size = struct.unpack("<Q", file.read(8))[0]
                    header = file.read(header_size)
                    return json.loads(header)
            except Exception as error_log:  #couldn't open file
                self.error_handler(kind="fail", error_log=error_log, identity=extension, reference=file_name)

# GGUF
    def __ungguf(self, file_name, extension):
        from gguf import GGUFReader
        try: # Method using gguf library, better for Latent Diffusion conversions
            reader = GGUFReader(file_name)
            file_data = defaultdict(dict)
            file_data[reader.data.dtype.name] = reader.data.dtype.name
            file_data["fields"] = reader.fields
            for tensor in reader.tensors:
                file_data[tensor.name] = f"{tensor.shape, tensor.data_offset}"
            return file_data
        except Exception as error_log:
            self.error_handler(kind="retry", error_log=error_log, identity=extension, reference=file_name)
        try: # Method using Llama library, better for LLMs
            from llama_cpp import Llama
            with open(file_name, "rb") as file_obj:
                magic = file_obj.read(4)
                if magic != b"GGUF":
                    logger.debug(f"Invalid GGUF magic number in '{file_name}'")
                    return
                version = struct.unpack("<I", file_obj.read(4))[0]
                if version < 2:
                    logger.debug(f"Unsupported GGUF version {version} in '{file_name}'")
                    return
            parser = Llama(model_path=file_name, vocab_only=True, verbose=False)
            return parser.metadata
        except ValueError as error_log: # We tried...
            self.error_handler(kind="fail", error_log=error_log, identity=extension, reference=file_name)

# PICKLETENSOR FILE
    def __unpickle(self, file_name, extension):
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
        self.hint_values
        self.hint_values["dtype"] = set()
        for key in deserialized_model:
            if "down_blocks." in key or "up_blocks." in key:
                # Create block formats (e.g., compvis, diffusers)
                diffusers_format = Domain.BlockFormat("diffusers")
                self.neural_net.add_block_format("diffusers", compvis_format)
            elif "input_blocks." in key or "out_blocks." in key:
                diffusers_format = Domain.BlockFormat("compvis")
                self.neural_net.add_block_format("compvis", compvis_format)

    # Example usage to create nn.compvis.sdxl
    add_entry(nn_domain, "nn.compvis.sdxl")

    # Example usage to add a component
    add_entry(nn_domain, "nn.compvis.sdxl", component_name="vae", dtype="fp16", file_size="50MB", distinction="LCM")


                        if deserialized_model.key.dtype is not None:

# Create components for 'sdxl' architecture
vae_component = Domain.BlockFormat.Architecture.Component("vae", "fp16", "50MB", "LCM")
unet_component = Domain.BlockFormat.Architecture.Component("unet", "fp32", "120MB", "INSTRUCT")

# Add components to the 'sdxl' architecture
sdxl_architecture.add_component("vae", vae_component)
sdxl_architecture.add_component("unet", unet_component)

# Flatten the nn_domain to check the structure
nn_domain_flat = nn_domain.flatten()

# Print flattened structure
print(json.dumps(nn_domain_flat, indent=4))



# Now 'nn.compvis.sdxl' exists in your hierarchy
    def error_handler(self, kind:str, error_log:str, identity:str=None, reference:str=None):
        if kind == "retry": logger.debug(f"Error reading metadata,  switching read method for '{identity}' type  {reference}, [{error_log}]", exc_info=True)
        elif kind == "fail": logger.debug(f"Metadata read attempts exhasted for:'{identity}' type {reference}  [{error_log}]")
        return

    def add_entry(domain, path, component_name=None, dtype=None, file_size=None, distinction=None):
        """Adds a new entry given a path like 'nn.compvis.sdxl'.

        Optionally, adds a component to the architecture if component details are provided.
        """
        parts = path.split('.')

        # Create or fetch the domain
        current_domain = domain

        # Create or fetch the block format
        block_format_name = parts[1]
        if block_format_name not in current_domain.block_formats:
            current_domain.add_block_format(block_format_name, Domain.BlockFormat(block_format_name))
        current_block_format = current_domain.block_formats[block_format_name]

        # Create or fetch the architecture
        architecture_name = parts[2]
        if architecture_name not in current_block_format.architectures:
            current_block_format.add_architecture(architecture_name, Domain.BlockFormat.Architecture(architecture_name))
        current_architecture = current_block_format.architectures[architecture_name]

        # Optionally, add the component
        if component_name and dtype and file_size and distinction:
            component = Domain.BlockFormat.Architecture.Component(component_name, dtype, file_size, distinction)
            current_architecture.add_component(component_name, component)


if __name__ == "__main__":


    log_level = "INFO"
    msg_init = None
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.logging import RichHandler
    handler = RichHandler(console=Console(stderr=True))

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)  # same as print
        handler.propagate = False

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)

    if msg_init is not None:
        logger = logging.getLogger(__name__)
        logger.info(msg_init)

    log_level = getattr(logging, log_level)
    logger = logging.getLogger(__name__)

    file = "C:\\Users\\woof\\AppData\\Local\\Shadowbox\\models\\image"
    blocks = BlockIndex()
    save_location = "C:\\Users\\Public\\code\\metadata_scans_v2\\"
    if Path(file).is_dir() == True:
        path_data = os.listdir(file)
        for each in tqdm(path_data, total=len(path_data)):
            file_path = os.path.join(file,each)
            blocks.main(file_path, save_location)
    else:
        blocks.main(file, save_location)
