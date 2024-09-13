import json
import struct
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import os
from os.path import basename as basedname
import tomllib
from collections import defaultdict
from sdbx import config
from sdbx.config import config, config_source_location

full_path = os.path.join(config.get_path("models.diffusers"), "elan-mt-bt-en-ja.safetensors")
filename = basedname(full_path)
print(full_path)

#source = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #windows box
#config_source_location = os.path.join(source, "config")

class ReadMeta:
    empty = {}
    full_data = {}
    known = {}
    meta = {}
    occurrence_counts = defaultdict(int)
    count_dict = {} # level of certainty, counts tensor block type matches
    model_tag = {  # measurements and metadata detected from the model
                "filename": "",
                "size": "",
                "dtype": "",
                "tensor_params": 0,
                "shape": "",
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
        with open(full_path, "rb") as f:
            header = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(header), object_hook=self._search_dict)

    @classmethod
    def data(self, filename, full_path):
        if Path(filename).suffix in {".pt", ".pth"}:  # yes, this will be slow
            # Access model's metadata (e.g., layer names and parameter shapes)
            model = torch.load(full_path, map_location="cpu")
            for p in model.named_parameters():
                print(f"Data: {p}, Size: {p.size()}")

        elif Path(filename).suffix in {".safetensors" or ".sft"}:
            self.occurrence_counts.clear()
            self.count_dict.clear()
            self.full_data.clear()
            self.known.clear()
            self.meta = self._parse_safetensors_metadata(full_path)
            self.full_data.update(self.model_tag)
            self.full_data.update(self.count_dict)
            class_key = list(self.count_dict.keys())
            for key, value in self.full_data.items():
                if value != "":
                    print(key,value) #debug
            print(class_key)
            self.model_tag.clear()
            self.meta.clear()
            return class_key, self.full_data

        elif Path(filename).suffix in {".gguf"}:
           self.meta = ""  # parse gguf metadata(path)
        else:
            raise RuntimeError(f"Unknown file format: {filename}")

    @classmethod
    def _search_dict(self, meta):
        keycount = 0
        self.meta = meta
        if self.meta.get("__metadata__", "not_found") != "not_found":
            self.model_tag["__metadata__"] = meta.get("__metadata__")
        model_classes = Path(os.path.join(config_source_location, "classify.toml"))
        with open(model_classes, "rb") as f:
            self.types = tomllib.load(f)  # Load the TOML contents into 'types'
        for key, value in self.types.items(): # Check if the value is a multiline string
            if isinstance(value, str): # Split  multiline string into lines and strip whitespace
                self.items = [item.strip() for item in value.strip().split('\n') if item.strip()] #get rid of newlines and whitespace
                self.known[key] = {i: item for i, item in enumerate(self.items)} # Create dictionary
        for key, values in self.known.items():  #model type, dict data
            for i, value in values.items():      #dict data to key pair
                for num in self.meta:         #extracted metadata as values
                    if value in num:     #if value matches one of our key values
                        self.occurrence_counts[value] += 1 #count matches
                self.count_dict[key] = self.occurrence_counts.get(value, 0) #pair match count with type of model

        for key, value in self.model_tag.items():
            if self.meta.get(key, "not_found") != "not_found": #handle inevitable exceptions invisibly
                if self.model_tag[key] == "":  #be sure the value isnt empty
                    self.model_tag[key] = meta.get(key) #drop it like its hot
                if key == "dtype":
                    self.model_tag["tensor_params"] += 1 # count tensors
                if key == "shape":
                    if meta.get(key) > self.model_tag["shape"]: #measure first shape size thats returned
                        self.model_tag["shape"] = self.meta.get(key)  # (room for improvement here, would prefer to be largest shape, tbh)
        return meta


folder = "checkpoints"
for each in os.listdir(os.path.join("c:",os.sep,"users","public","models",folder)):
    filename = each
    full_path = os.path.join("c:",os.sep,"users","public","models",folder, filename)
    metareader = ReadMeta(os.path.basename(full_path), full_path).data(os.path.basename(full_path), full_path)
    


metareader = ReadMeta(full_path).data(os.path.basename(full_path), full_path)
