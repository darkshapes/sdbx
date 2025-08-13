import os
import json
import struct
from enum import Enum
from math import isclose
from pathlib import Path
from collections import defaultdict

from llama_cpp import Llama
from pydantic import BaseModel, RootModel, ValidationError, root_validator

from sdbx import logger
from sdbx import config
from sdbx.config import config_source_location

# Constants for criteria thresholds
MODEL_TENSOR_PCT = 2e-3
MODEL_BLOCK_PCT = 1e-4
VAE_PCT = 5e-3
VAE_XL_PCT = 1e-8
TRA_PCT = 1e-4
TRA_LEEWAY = 0.03
LORA_PCT = 0.05

# Load configuration data
peek = config.get_default("tuning", "peek")
known = config.get_default("tuning", "known")

model_peek = peek.get('model_peek', {})
vae_peek_12 = peek.get('vae_peek_12', {})
vae_peek = peek.get('vae_peek', {})
vae_peek_0 = peek.get('vae_peek_0', {})
tra_peek = peek.get('tra_peek', {})
lora_peek = peek.get('lora_peek', {})

class ModelType(Enum):
    DIFFUSION = 'DIF'
    LANGUAGE = 'LLM'
    LORA = 'LOR'
    TRANSFORMER = 'TRA'
    VAE = 'VAE'

class ModelCodeData(BaseModel):
    size: int
    path: str
    dtype_or_context_length: str

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_list

    @classmethod
    def validate_list(cls, value):
        if isinstance(value, list) and len(value) == 3:
            size, path, dtype_or_context_length = value
            return cls(size=size, path=path, dtype_or_context_length=dtype_or_context_length)
        else:
            raise ValueError(f"Invalid value for ModelCodeData: {value}")

class IndexData(RootModel):
    root: dict

class EvalMeta:
    """
    Determines the identity of an unknown tensor and returns a tag to identify it.
    """

    def __init__(self, extract):
        self.tag = None
        self.code = None
        self.extract = extract
        self.clip_inside = False
        self.vae_inside = False
        self.tag_dict = {}

        keys_int = ["unet", "diffusers", "transformers", "sdxl", "tensor_params",
                    "mmdit", "flux", "diffusers_lora", "hunyuan", "size"]
        for key in keys_int:
            setattr(self, key, int(self.extract.get(key, 0)))

        # Extract shape
        self.shape = self.extract.get("shape", [0])

        # Extract string values
        keys_str = ["filename", "extension", "path", "dtype", "general.name",
                    "general.architecture", "tokenizer.chat_template", "context_length"]
        for key in keys_str:
            attr_name = key.split(".")[-1]
            setattr(self, attr_name, self.extract.get(key, ""))

        self.arch = self.architecture.upper() if hasattr(self, 'architecture') and self.architecture else ""

    def tks(self, tag, key, sub_key):
        self.tag = tag
        self.key = key
        self.sub_key = sub_key

    def process_vae(self):
        if self.shape[:1] == [32]:
            self.tks("0", '114560782', '248')
        elif self.shape[:1] == [512]:
            self.tks("0", "335304388", "244")
        elif self.sdxl == 12:
            if self.mmdit == 4:
                self.tks("0", "167335342", "248")
            elif any(isclose(self.size, val, rel_tol=VAE_XL_PCT) for val in [167335343, 167666902]):
                self.tks(
                    '12' if "vega" in self.filename.lower() else "0", 
                    '167335344' if "vega" in self.filename.lower() else "167335343", 
                    "248"
                )
            else:
                self.tks("12", "334643238", "248")
        elif self.mmdit == 8:
            if isclose(self.size, 404581567, rel_tol=VAE_XL_PCT):
                self.tks("0", "404581567", "304")
            else:
                self.tks("v", "167333134", "248")
        elif isclose(self.size, 334641190, rel_tol=VAE_XL_PCT):
            self.tks("v", "334641190", "250")
        else:
            self.tks("v", "334641162", "250")

    def process_lor(self):
        for size, attributes in lora_peek.items():
            size_int = int(size)
            if any(isclose(self.size, size_int * factor, rel_tol=LORA_PCT) for factor in [1, 2, 0.5]):
                for tensor_params, desc in attributes.items():
                    if isclose(self.tensor_params, int(tensor_params), rel_tol=LORA_PCT):
                        if any(desc_item.lower() in self.filename.lower() for desc_item in desc):
                            self.set_key("l", size, tensor_params)
                            self.value = desc_item
                            return

    def process_tra(self):
        for tensor_params, attributes in tra_peek.items():
            if isclose(self.tensor_params, int(tensor_params), rel_tol=TRA_LEEWAY):
                for shape, name in attributes.items():
                    if isclose(self.transformers, name[0], rel_tol=TRA_PCT):
                        self.set_key("t", tensor_params, shape)
                        return

    def process_model(self):
        for tensor_params, attributes in model_peek.items():
            if isclose(self.tensor_params, int(tensor_params), rel_tol=MODEL_TENSOR_PCT):
                for shape, name in attributes.items():
                    if self.shape and any(
                        isclose(val, name[0], rel_tol=MODEL_BLOCK_PCT)
                        for val in [int(self.shape[0]), self.diffusers, self.mmdit, self.flux, self.diffusers_lora, self.hunyuan]
                    ):
                        self.set_key("m", tensor_params, shape)
                        logger.debug(f"{self.tag}, VAE-{self.tag}:{self.vae_inside}, CLI-{self.tag}:{self.clip_inside}")
                        return
                logger.debug(f"No shape key for model '{self.extract}'.", exc_info=True)

    def data(self):
        if getattr(self, 'name', None):
            self.set_key("c", "", "")
        elif self.unet > 96:
            self.vae_inside = True
        elif self.unet == 96:
            self.process_vae()
        elif self.diffusers >= 256:
            self.process_lor()
        elif self.transformers >= 2:
            self.clip_inside = True
            self.process_tra()
        elif self.size > 1e9:
            self.process_model()

        tag_lookup_map = {
            "0": vae_peek_0,
            "12": vae_peek_12,
            "v": vae_peek,
            "t": tra_peek,
            "l": lora_peek,
            "m": model_peek
        }

        if self.tag in ["0", "12", "v"]:
            self.code = ModelType.VAE
            self.lookup = getattr(self, f'vae_peek_{self.tag}', {}).get(self.key, {}).get(self.sub_key)
        elif self.tag in ["t", "l", "m"]:
            code_mapping = {"t": ModelType.TRANSFORMER, "l": ModelType.LORA, "m": ModelType.DIFFUSION}
            self.code = code_mapping.get(self.tag)
            peek_attr = getattr(self, f'{self.tag}_peek', {})
            name = peek_attr.get(self.key, {}).get(self.sub_key)
            if self.tag == "l":
                self.lookup = f"{getattr(self, 'value', '')}-{name[-1][0]}"
            else:
                self.lookup = f"{name[-1][0]}"
        elif self.tag == "c":
            self.code = ModelType.LANGUAGE
            self.lookup = self.arch
        else:
            logger.debug(f"Could not determine id '{self.extract}'. Unknown type: '{self.filename}'.", exc_info=True)
            return None

        if not self.tag:
            logger.debug(f"Not indexed. 'No eval error' should follow: '{self.extract}'.", exc_info=True)
            return None

        logger.debug(f"{self.code}, {self.filename}, {self.size}, {self.path}")

        return self.code, (self.filename, self.lookup, self.size, self.path, getattr(self, 'context_length', '') or self.dtype)

class ReadMeta:
    """
    Reads metadata from model files and extracts useful information.
    """

    def __init__(self, path):
        self.path = path
        self.full_data = {}
        self.meta = {}
        self.count_dict = {}

        self.known = known

        self.model_tag = {
            "filename": "",
            "size": "",
            "path": "",
            "dtype": "",
            "torch_dtype": "",
            "tensor_params": 0,
            "shape": "",
            "data_offsets": "",
            "general.name": "",
            "general.architecture": "",
            "tokenizer.chat_template": "",
            "context_length": "",
            "block_count": "",
            "attention.head_count": "",
            "attention.head_count_kv": "",
        }
        self.occurrence_counts = defaultdict(int)
        self.filename = os.path.basename(self.path)
        self.ext = Path(self.filename).suffix.lower()

        if not os.path.exists(self.path):
            logger.debug(f"File not found: '{self.filename}'.", exc_info=True)
            raise FileNotFoundError(f"File not found: {self.filename}")
        else:
            self.model_tag["filename"] = self.filename
            self.model_tag["extension"] = self.ext.replace(".", "")
            self.model_tag["path"] = self.path
            self.model_tag["size"] = os.path.getsize(self.path)

    def _parse_safetensors_metadata(self):
        try:
            with open(self.path, "rb") as file:
                header_size = struct.unpack("<Q", file.read(8))[0]
                header = file.read(header_size)
                metadata = json.loads(header)
                self._search_dict(metadata)
        except Exception as e:
            logger.debug(f"Error reading safetensors metadata from '{self.path}': {e}", exc_info=True)

    def _parse_gguf_metadata(self):
        try:
            with open(self.path, "rb") as file:
                magic = file.read(4)
                if magic != b"GGUF":
                    logger.debug(f"Invalid GGUF magic number in '{self.path}'")
                    return
                version = struct.unpack("<I", file.read(4))[0]
                if version < 2:
                    logger.debug(f"Unsupported GGUF version {version} in '{self.path}'")
                    return
            parser = Llama(model_path=self.path, vocab_only=True, verbose=False)
            self.meta = parser.metadata
            self._search_dict(self.meta)
        except Exception as e:
            logger.debug(f"Error parsing GGUF metadata from '{self.path}': {e}", exc_info=True)
    
    def _parse_metadata(self):
        self.full_data.update({k: v for k, v in self.model_tag.items() if v})
        self.full_data.update({k: v for k, v in self.count_dict.items() if v})
        for k, v in self.full_data.items(): 
            logger.debug(f"{k}: {v}")
        self.count_dict.clear()
        self.model_tag.clear()
        self.meta = {}

    def data(self):
        if self.ext in [".pt", ".pth", ".ckpt"]:
            # Placeholder for future implementation
            pass
        elif self.ext in [".safetensors", ""]:
            self._parse_safetensors_metadata()
            self._parse_metadata()
        elif self.ext == ".gguf":
            self._parse_gguf_metadata()
            self._parse_metadata()
        else:
            logger.debug(f"Unrecognized file format: '{self.filename}'", exc_info=True)
        return self.full_data

    def _search_dict(self, meta):
        self.meta = meta
        if self.ext == ".gguf":
            for key, value in self.meta.items():
                logger.debug(f"{key}: {value}")
                if key in self.model_tag:
                    self.model_tag[key] = value
                if "general.architecture" in self.model_tag and self.model_tag["general.architecture"]:
                    prefix = self.model_tag["general.architecture"]
                    if key.startswith(f"{prefix}."):
                        prefixless_key = key.replace(f"{prefix}.", "")
                        if prefixless_key in self.model_tag:
                            self.model_tag[prefixless_key] = value
        elif self.ext in [".safetensors", ""]:
            for key in self.meta:
                if key in self.model_tag:
                    self.model_tag[key] = self.meta.get(key)
                if "dtype" in key:
                    self.model_tag["tensor_params"] += 1
                elif "shape" in key:
                    shape_value = self.meta.get(key)
                    if shape_value > self.model_tag.get("shape", 0):
                        self.model_tag["shape"] = shape_value
                if "data_offsets" not in key and not any(x in key for x in ["shapes", "dtype"]):
                    for block, model_type in self.known.items():
                        if block in key:
                            self.occurrence_counts[model_type] += 1
                            self.count_dict[model_type] = self.occurrence_counts[model_type]
        return self.meta

    def __repr__(self):
        return f"ReadMeta(data={self.data()})"

class ModelIndexer:
    def __init__(self):
        self.index = {}

        index_file = os.path.join(config.get_path("models"), "index.json")
        spec_file = os.path.join(config.get_path("models"), "spec.json")
        
        if os.path.exists(index_file):
            self.load_index(index_file)
        else:
            self.write_index(index_file)
    
    def load_index(self, index_file):
        index_data = IndexData.parse_file(index_file)
        self.index = index_data.root

    def write_index(self, index_file):
        self.index = {model_type.value: defaultdict(dict) for model_type in ModelType}

        def extract_files(tree):
            for node in tree:
                if 'path' in node:
                    yield node['path']
                elif 'children' in node:
                    yield from extract_files(node['children'])

        for full_path in list(extract_files(config.get_path_tree("models", file_callback=lambda path: {"path": path}))):
            metareader_data = ReadMeta(full_path).data()
            if not metareader_data:
                logger.debug(f"No metadata found for '{full_path}'.", exc_info=True)
                continue

            eval_data = EvalMeta(metareader_data).data()
            if not eval_data:
                logger.debug(f"No evaluation data for '{full_path}'.", exc_info=True)
                continue

            code_enum = eval_data[0]
            if not isinstance(code_enum, ModelType):
                logger.debug(f"Invalid code type: {code_enum}", exc_info=True)
                continue

            filename, lookup, size, path, context_dtype = eval_data[1]
            self.index[code_enum.value][filename][lookup] = [size, path, context_dtype]

        if self.index:
            try:
                with open(index_file, "w+", encoding="utf8") as index:
                    json.dump(self.index, index, ensure_ascii=False, indent=4, sort_keys=True)
            except Exception as e:
                logger.debug(f"Error writing index file '{index_file}': {e}", exc_info=True)
        else:
            logger.debug("No data to write to index.", exc_info=True)

    def fetch_id(self, search_item):
        for tag_enum, data in self.index.items():
            if search_item in data:
                for category, value in data[search_item].items():
                    return tag_enum, category, value
        return None, None, None
        
    def fetch_compatible(self, query):
        clip_data = config.get_default("tuning", "clip_data")
        lora_priority = config.get_default("algorithms", "lora_priority")

        model_indexes = {
            mt: self.index.get(mt.value, {}) for mt in [ModelType.VAE, ModelType.TRANSFORMER, ModelType.LORA]
        }

        # Initialize empty dictionaries
        lora_sorted = {}
        tra_sorted = {}
        vae_sorted = {}

        try:
            tra_req = self._fetch_txt_enc_types(clip_data, query)
            if tra_req is None:
                tra_sorted = None
                logger.debug(f"No external text encoder found compatible with '{query}'.", exc_info=True)
            else:
                tra_match = {}
                for item in tra_req[:-1]:
                    matches = self._filter_compatible(item, model_indexes[ModelType.TRANSFORMER])
                    if matches:
                        for match in matches:
                            key, value = match
                            tra_match[key[1]] = value
                tra_sorted = tra_match or None
        except TypeError as error_log:
            logger.debug(f"No match found for {query}: {error_log}", exc_info=True)
            tra_sorted = None

        vae_sorted = self._filter_compatible(query, model_indexes[ModelType.VAE]) or None
        lora_match = self._filter_compatible(query, model_indexes[ModelType.LORA]) or []

        lora_sorted = {}
        for priority in lora_priority:
            for key, val in lora_match:
                if priority in key[1]:
                    lora_sorted[key] = val
        lora_sorted = lora_sorted or None

        return vae_sorted, tra_sorted, lora_sorted

    def fetch_refiner(self):
        dif_index = self.index.get(ModelType.DIFFUSION.value, {})
        return dif_index.get("STA-XR", None)

    def _fetch_txt_enc_types(self, data, query, path=None):
        path = path or []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = path + [key]
                if value == query:
                    return current_path
                elif isinstance(value, (dict, list)):
                    result = self._fetch_txt_enc_types(value, query, current_path)
                    if result:
                        return result
        elif isinstance(data, list):
            for index, value in enumerate(data):
                current_path = path + [index]
                if value == query:
                    return current_path
                elif isinstance(value, (dict, list)):
                    result = self._fetch_txt_enc_types(value, query, current_path)
                    if result:
                        return result
        return None

    def _filter_compatible(self, query, index):
        pack = defaultdict(dict)
        for k, v in index.items():
            for code in v.keys():
                if query in code:
                    pack[(k, code)] = v[code]
        return sorted(pack.items(), key=lambda item: item[1]) if pack else None