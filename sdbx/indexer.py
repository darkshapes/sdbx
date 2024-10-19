import os
import json
import struct
from enum import Enum
from math import isclose
from pathlib import Path
from collections import defaultdict
from llama_cpp import Llama
from sdbx import logger
from sdbx import config
from sdbx.config import config_source_location

peek = config.get_default("tuning", "peek") # block and tensor values for identifying
known = config.get_default("tuning", "known") # raw block & tensor data

class EvalMeta:
    # determines identity of unknown tensor
    # return a tag to identify
    # BASE MODEL STA-XL   STAble diffusion xl. always use the first 3 letters
    # VAE MODEL  VAE-STA-XL
    # TEXT MODEL CLI-VL  CLIP ViT/l
    #            CLI-VG  Oop en CLIP ViT/G
    # LORA MODEL PCM, SPO, LCM, HYPER, ETC  performance boosts and various system improvements
    # to be added
    # S2T, A2T, VLLM, INPAINT, CONTROL NET, T2I, PHOTOMAKER, SAMS. WHEW

    # CRITERIA THRESHOLDS
    model_tensor_pct = 2e-3  # fine tunings
    model_block_pct = 1e-4   # % of relative closeness to a known checkpoint value
    model_size_pct = 3e-3    
    vae_pct = 5e-3           # please do not disrupt
    vae_xl_pct = 1e-8
    tra_pct = 1e-4
    tra_leeway = 0.03
    lora_pct = 0.05

    model_peek = peek['model_peek']
    vae_peek_12 = peek['vae_peek_12']
    vae_peek = peek['vae_peek']
    vae_peek_0 = peek['vae_peek_0']
    tra_peek = peek['tra_peek']
    lora_peek = peek['lora_peek']

    def __init__(self, extract, verbose=False):
        self.tag = ""
        self.code = ""
        self.extract = extract
        self.clip_inside = False
        self.vae_inside = False
        self.verbose = verbose

        # model measurements
        #integer
        self.unet_value = int(self.extract.get("unet", 0))
        self.diffuser_value = int(self.extract.get("diffusers", 0))
        self.transformer_value = int(self.extract.get("transformers", 0))
        self.sdxl_value = int(self.extract.get("sdxl", 0))
        self.tensor_value = int(self.extract.get("tensor_params", 0))
        self.mmdit_value = int(self.extract.get("mmdit", 0))
        self.flux_value = int(self.extract.get("flux", 0))
        self.diff_lora_value = int(self.extract.get("diffusers_lora", 0))
        self.hunyuan = int(self.extract.get("hunyuan", 0))
        self.size = int(self.extract.get("size", 0))
        self.shape_value = self.extract.get("shape", 0)
        if self.shape_value: self.shape_value = self.shape_value[0:1]

        #string value
        self.filename = self.extract.get("filename", "")
        self.ext = self.extract.get("extension", "")
        self.path = self.extract.get("path", "")
        self.dtype = self.extract.get("dtype", "") if not "" else self.extract.get("torch.dtype", "")

        # model supplied metadata
        self.name_value = self.extract.get("general.name","")
        self.arch = self.extract.get("general.architecture","").upper()
        self.tokenizer = self.extract.get("tokenizer.chat_template", "")
        self.context_length = self.extract.get("context_length","")

    def process_vae(self):
        if [32] == self.shape_value:
            self.tag = "0"
            self.key = '114560782'
            self.sub_key = '248' # sd1 hook
        elif [512] == self.shape_value:
            self.tag = "0"
            self.key = "335304388"
            self.sub_key = "244" # flux hook
        elif self.sdxl_value == 12:
            if self.mmdit_value == 4:
                self.tag = "0"
                self.key = "167335342"
                self.sub_key = "248"  # auraflow
            elif (isclose(self.size, 167335343, rel_tol=self.vae_xl_pct)
            or isclose(self.size, 167666902, rel_tol=self.vae_xl_pct)):
                if "vega" in self.filename.lower():
                    self.tag = '12'
                    self.key = '167335344'
                    self.sub_key = '248'  #vega
                else:
                    self.tag = "0"
                    self.key = "167335343"
                    self.sub_key = "248"  #kolors
            else:
                self.tag = "12"
                self.key = "334643238"
                self.sub_key = "248" #pixart
        elif self.mmdit_value == 8:
            if isclose(self.size, 404581567, rel_tol=self.vae_xl_pct):
                self.tag = "0"
                self.key = "404581567"
                self.sub_key = "304" #sd1 hook
            else:
                self.tag = "v"
                self.key = "167333134"
                self.sub_key = "248" #sdxl hook
        elif isclose(self.size, 334641190, rel_tol=self.vae_xl_pct):
            self.tag = "v"
            self.key = "334641190"
            self.sub_key = "250" #sd1 hook
        else:
            self.tag = "v"
            self.key = "334641162"
            self.sub_key = "250" #sdxl hook

    def process_lor(self):
        if self.size != 0:
            for size, attributes in self.lora_peek.items():
                if (
                    isclose(self.size, int(size),  rel_tol=self.lora_pct) or
                    isclose(self.size, int(size)*2, rel_tol=self.lora_pct) or
                    isclose(self.size, int(size)/2, rel_tol=self.lora_pct)
                ):
                    for tensor_params, desc in attributes.items():
                        if isclose(self.tensor_value, int(tensor_params), rel_tol=self.lora_pct):
                            for each in next(iter([desc, 'not_found'])):
                                title = self.filename.upper()
                                if each in title:
                                    self.tag = "l"
                                    self.key = size
                                    self.sub_key = tensor_params
                                    self.value = each #lora hook                               
                                        # found lora

    def process_tra(self):
        for tensor_params, attributes in self.tra_peek.items():
            if isclose(self.tensor_value, int(tensor_params), rel_tol=self.tra_leeway):
                for shape, name in attributes.items():
                    if isclose(self.transformer_value, name[0], rel_tol=self.tra_pct):
                            self.tag = "t"
                            self.key = tensor_params
                            self.sub_key = shape # found transformer

    def process_model(self):
        if isclose(self.size, 5135149760, rel_tol=self.model_size_pct):
            self.tag = "m"
            self.key = "1468"
            self.sub_key = "320" #found model
        else:
            for tensor_params, attributes, in self.model_peek.items():
                if isclose(self.tensor_value, int(tensor_params), rel_tol=self.model_tensor_pct):

                    for shape, name in attributes.items():
                        num = self.shape_value[0:1]
                        if num:
                            if (isclose(int(num[0]), int(shape), rel_tol=self.model_block_pct)
                            or isclose(self.diffuser_value, name[0], rel_tol=self.model_block_pct)
                            or isclose(self.mmdit_value, name[0], rel_tol=self.model_block_pct)
                            or isclose(self.flux_value, name[0], rel_tol=self.model_block_pct)
                            or isclose(self.diff_lora_value, name[0], rel_tol=self.model_block_pct)
                            or isclose(self.hunyuan, name[0], rel_tol=self.model_block_pct)):
                                    self.tag = "m"
                                    self.key = tensor_params
                                    self.sub_key = shape #found model
                        else:
                            logger.debug(f"'[No shape key for model '{self.extract}'.", exc_info=True)
                            self.tag = "m"
                            self.key = tensor_params
                            self.sub_key = shape               ######################################DEBUG

    def data(self):
        if "" not in self.name_value or self.context_length: # check LLM
            self.tag = "c"
            self.key = ""
            self.sub_key = ""
        else:
            if self.unet_value > 96:
                self.vae_inside = True
            if self.unet_value == 96:  # Check VAE
                self.code = self.process_vae()
            if self.diffuser_value >= 256:  # Check LoRA
                self.code = self.process_lor()
            if self.transformer_value >= 2:  # Check CLIP
                self.clip_inside = True
                self.code = self.process_tra()
            if self.size > 1e9:  # Check model
                self.code = self.process_model()


        self.tag_dict = {}
        # 0 = vae_peek_0, 12 = vae_peek_12, v = vae_peek
        # these are separated because file sizes are otherwise too similar
        # please do not disrupt
        if self.tag == "0" or self.tag == "12" or self.tag == "v":
            self.code = "VAE"
            if self.tag == "0":
                self.lookup = f"{self.vae_peek_0[self.key][self.sub_key]}"
            elif self.tag == "12":
                self.lookup = f"{self.vae_peek_12[self.key][self.sub_key]}"
            elif self.tag == "v":
                self.lookup = f"{self.vae_peek[self.key][self.sub_key]}"
        elif self.tag == "t":
            self.code = f"TRA"
            name = self.tra_peek[self.key][self.sub_key]
            self.lookup = f"{name[len(name)-1:][0]}" # type name is a list item
        elif self.tag == "l":   
            self.code = f"LOR"
            name = self.lora_peek[self.key][self.sub_key]
            self.lookup = f"{self.value}-{name[len(name)-1:][0]}" # type name is a list item
        elif self.tag == "m":
            self.code = f"DIF"
            name = self.model_peek[self.key][self.sub_key]
            self.lookup = f"{name[len(name)-1:][0]}"
        elif self.tag == "c": 
            self.code = f"LLM"
            self.lookup = f"{self.arch}"
        else:
            logger.debug(f"Unknown type:'{self.filename}'.")
            # consider making ignore list for undetermined models
            logger.debug(f"'Could not determine id '{self.extract}'.", exc_info=True)
            pass

        if self.tag == "":
            logger.debug(f"'Not indexed. 'No eval error' should follow: '{self.extract}'.", exc_info=True)
            pass
        else:   #format [ model type code, filename, compatability code, file size, full file path]
            if self.verbose is True: logger.debug(self.code, self.lookup, self.filename, self.size, self.path)
            return self.code, (
                self.filename, self.lookup, self.size, self.path, 
                (self.context_length if self.context_length else self.dtype))
                                                 
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
            "size": 0,
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
                """ try opening file """
        except Exception as log:
            logger.debug(f"Error reading safetensors metadata from '{self.path}': {log}", exc_info=True)
            logger.debug(log, exc_info=True)
        else:
            with open(self.path, "rb") as file:
                header = struct.unpack("<Q", file.read(8))[0]
                try:
                    return json.loads(file.read(header), object_hook=self._search_dict)
                except:
                    log = f"Path not found'{self.path}'''."
                    logger.debug(log, exc_info=True)
            
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
        self.full_data.update((k, v) for k, v in self.model_tag.items() if v != "")
        self.full_data.update((k, v) for k, v in self.count_dict.items() if v != 0)
        for k, v in self.full_data.items(): 
            logger.debug(f"{k}: {v}")
        self.count_dict.clear()
        self.model_tag.clear()
        self.meta = {}

    def data(self):
        if self.ext in [".pt", ".pth", ".ckpt"]:
            # Placeholder for future implementation
            pass
        elif self.ext in [".safetensors", ".sft", ""]:
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
        elif self.ext in [".safetensors", ".sft," ""]:
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

class IndexManager:

    all_data = {
        "DIF": defaultdict(dict),
        "LLM": defaultdict(dict),
        "LOR": defaultdict(dict),
        "TRA": defaultdict(dict),
        "VAE": defaultdict(dict),
    }
    
    def write_index(self, index_file="index.json"):
        # Collect all data to write at once
        self.directories =  config.get_default("directories","models") #multi read
        self.delete_flag = True
        config.write_spec()
        for each in self.directories:
            self.path_name = config.get_path(f"models.{each}")
            index_file = os.path.join(config_source_location, index_file)
            for each in os.listdir(self.path_name):  # SCAN DIRECTORY           #todo - toggle directory scan
                full_path = os.path.join(self.path_name, each)
                if os.path.isfile(full_path):  # Check if it's a file
                    self.metareader = ReadMeta(full_path).data()
                    if self.metareader is not None:
                        self.eval_data = EvalMeta(self.metareader).data()
                        if self.eval_data != None:
                            tag = self.eval_data[0]
                            filename = self.eval_data[1][0]
                            compatability = self.eval_data[1][1:2][0]
                            data = self.eval_data[1][2:5]
                            self.all_data[tag][filename][compatability] = (data)
                        else:
                            logger.debug(f"No eval: {each}.", exc_info=True)
                    else:
                        log = f"No data: {each}."
                        logger.debug(log, exc_info=True)
                        logger.debug(log)
        if self.all_data:
            if self.delete_flag:
                try:
                    os.remove(index_file)
                    self.delete_flag =False
                except FileNotFoundError as error_log:
                    logger.debug(f"'Config file absent at write time: {index_file}.'{error_log}", exc_info=True)
                    self.delete_flag =False
                    pass
            with open(os.path.join(config_source_location, index_file), "a", encoding="UTF-8") as index:
                json.dump(self.all_data, index, ensure_ascii=False, indent=4, sort_keys=True)
        else:
            log = "Empty model directory, or no data to write."
            logger.debug(f"{log}{error_log}", exc_info=True)

     #recursive function to return model codes assigned to tree keys and transformer model values
    def _fetch_txt_enc_types(self, data, query, path=None, return_index_nums=False):
        if path is None: path = []

        if isinstance(data, dict):
            for key, self.value in data.items():
                self.current = path + [key]
                if self.value == query:
                    return self._unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self._fetch_txt_enc_types(self.value, query, self.current)
                    if self.match:
                        return self.match
        elif isinstance(data, list):
            for key, self.value in enumerate(data):
                self.current = path if not return_index_nums else path + [key]
                if self.value == query:
                    return self._unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self._fetch_txt_enc_types(self.value, query, self.current)
                    if self.match:
                        return self.match
                    
    #fix the recursive list so it doesnt make lists inside itself
    def _unpack(self): 
        iterate = []  
        self.match = self.current, self.value           
        for i in range(len(self.match)-1):
            for j in (self.match[i]):
                iterate.append(j)
        iterate.append(self.match[len(self.match)-1])
        return iterate
    
    #find the model code for a single model
    def fetch_id(self, search_item):
        for each in self.all_data.keys(): 
            peek_index = config.get_default("index", each)
            if not isinstance(peek_index, dict):
                continue  # Skip if peek_index is not a dict
            if search_item in peek_index:
                break
            else:
                continue 
        if search_item in peek_index:
            for category, value in peek_index[search_item].items():
                return each, category, value  # Return keys and corresponding value
        else:
            return "∅", "∅","∅"

    #get compatible models from a specific model code
    def fetch_compatible(self, query): 
        self.clip_data = config.get_default("tuning", "clip_data") 
        self.vae_index = config.get_default("index", "VAE")
        self.tra_index = config.get_default("index", "TRA")
        self.lor_index = config.get_default("index", "LOR")
        self.model_indexes = {
            "vae": self.vae_index,
            "tra": self.tra_index, 
            "lor": self.lor_index
            }
        try:
            tra_sorted = {}
            self.tra_req = self._fetch_txt_enc_types(self.clip_data, query)
        except TypeError as error_log:
            log = f"No match found for {query}"
            logger.debug(f"{log}{error_log}", exc_info=True)
        if self.tra_req == None:
            tra_sorted =str("∅")
            logger.debug(f"No external text encoder found compatible with {query}.", exc_info=True)
        else:
            tra_match = {}
            for i in range(len(self.tra_req)-1):
                tra_match[i] = self.filter_compatible(self.tra_req[i], self.model_indexes["tra"])
                if tra_match[i] == None:
                    tra_match[i] == query
            try:
                if tra_match[0] == []:
                    logger.debug(f"No external text encoder found compatible with {query}.", exc_info=True)
                    tra_sorted = {}
                else:
                    for each in tra_match.keys():
                        tra_sorted[tra_match[each][0][0][1]] = tra_match[each][0][1]
            except IndexError as error_log:
                logger.debug(f"Error when returning encoder for {query} : {error_log}.", exc_info=True)
                tra_sorted = {}
        vae_sorted = self.filter_compatible(query, self.model_indexes["vae"])
        lora_sorted = self.filter_compatible(query, self.model_indexes["lor"])
        lora_sorted = dict(lora_sorted)
        if vae_sorted == []: 
            vae_sorted =str("∅")
            print(query)
            logger.debug(f"No external VAE found compatible with {query}.", exc_info=True)
        if lora_sorted == []: 
            lora_sorted =str("∅")
            logger.debug(f"No compatible LoRA found for {query}.", exc_info=True)

        return vae_sorted, tra_sorted, lora_sorted

    def fetch_refiner(self):
        self.dif_index = config.get_default("index", "DIF")
        for key, value in self.dif_index.items():
            if key == "STA-XR":
                return value
        return "∅"

    #within a dict of models of the same type, match model code & sort by file size
    def filter_compatible(self, query, index):
        pack = defaultdict(dict)
        if index.items():
            for k, v in index.items():
                for code in v.keys():
                    if query in code:
                        pack[k, code] = v[code]
                        
            sort = sorted(pack.items(), key=lambda item: item[1])
            return sort
        else:
            logger.debug("Compatible models not found")
            return "∅"
    

