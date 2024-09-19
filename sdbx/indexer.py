import os
import json
import struct
from time import process_time_ns
from math import isclose
from pathlib import Path
from collections import defaultdict
from llama_cpp import Llama
from sdbx import config, logger


print(f'begin: {process_time_ns()*1e-6} ms')

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
    model_tensor_pct = 1e-4  # % of relative closeness to an actual checkpoint value
    model_shape_pct = 1e-7
    model_block_pct = 1e-4
    vae_pct = 5e-3
    vae_size_pct = 1e-3
    vae_xl_pct = 1e-4
    vae_sd1_pct = 3e-2
    vae_sd1_full_pct = 1e-3
    vae_sd1_tp_pct = 9e-2
    tf_pct = 1e-4
    tf_leeway = 0.03
    lora_pct = 0.05

    model_peek = peek['model_peek']
    vae_peek_12 = peek['vae_peek_12']
    vae_peek = peek['vae_peek']
    vae_peek_0 = peek['vae_peek_0']
    tf_peek = peek['tf_peek']
    lora_peek = peek['lora_peek']

    def __init__(self, extract):
        self.extract = extract
        self.clip_inside = False
        self.vae_inside = False

    def __returner(self):
        self.size = self.extract.get("size", "")
        self.filename = self.extract.get("filename", "")
        self.ext = self.extract.get("extension", "")
        self.path = self.extract.get("path", "")
        #if self.tag[:3] == "VAE": 
        print(f"{self.tag} {self.size} {os.path.basename(self.path)}") # logger.debug
        return (self.tag, self.size, self.path,)

    def data(self):
        if int(self.extract.get("unet", 0)) > 96: self.vae_inside = True
        if int(self.extract.get("unet", 0)) == 96:  # Check VAE
            self.process_vae()
        if int(self.extract.get("diffusers", 0)) > 256:  # Check LoRA
            self.process_lora()

        if int(self.extract.get("transformers", 0)) >= 2:  # Check CLIP
            self.clip_inside = True
            self.process_tf()
        try:
            if int(self.extract.get("size", 0)) > 1e9:  # Check model
                self.process_model()
        except ValueError as error_log:
            logger.debug(f"'[Unknown model type] No model found '{self.extract}''{error_log}'.", exc_info=True)
            print(f"Unknown model type '{self.extract}'.")
        else:
            if self.extract.get("general.name","") != "":
                self.tag = f"LLM-{self.extract.get("general.architecture","").upper()}"
                return self.__returner()

    def process_vae(self):
        try:
            if 512 == self.extract.get("shape", 0)[0]:
                """
                check shape
                """
                self.tag = "VAE-"
                self.tag = f"{self.tag}{self.vae_peek_0['335304388']['244']}" # flux hook
            return self.__returner()  
        except:
            pass
        if self.extract.get("sdxl", 0) == 12:
            if self.extract.get("mmdit", 0) == 4:
                self.tag = "VAE-"
                self.tag = f"{self.tag}{self.vae_peek_0['167335342']['248']}" #auraflow
                return self.__returner()
            elif (isclose(int(self.extract.get("size", 0)), 167335343, rel_tol=self.vae_xl_pct)
                or isclose(int(self.extract.get("size", 0)), 167666902, rel_tol=self.vae_xl_pct)):
                    self.tag = "VAE-"
                    self.tag = f"{self.tag}{self.vae_peek_0['167335343']['248']}" #kolors
                    return self.__returner()
            else:
                self.process_vae_12()
        else:
            if isclose(self.extract.get("tensor_params", 0), 304, rel_tol=self.vae_xl_pct):
                try:
                    if self.extract.get("mmdit", 0) == 8:
                        self.tag = "VAE-"
                        self.tag = f"{self.tag}{self.vae_peek_0['404581567']['304']}" #sd1 hook
                        return self.__returner() 
                except:
                    pass
            else:
                try:
                    if 32 == self.extract.get("shape", 0)[0]:
                        """
                        check shape
                        """
                        self.tag = "VAE-"
                        self.tag = f"{self.tag}{self.vae_peek_0['114560782']['248']}" # sd1 hook
                    return self.__returner()  
                except:
                    pass
            self.process_vae_without_12()

    def process_vae_12(self):
        self.tag = "VAE-"
        if (isclose(int(self.extract.get("tensor_params", 0)), 167335344, rel_tol=self.vae_xl_pct)
        or"vega" in self.extract.get("filename", 0).lower()):
            self.tag = f"{self.tag}{self.vae_peek_12['167335344']['248']}"# segmind vega
            return self.__returner()
        else:
            self.tag = f"{self.tag}{self.vae_peek_12['334643238']['248']}" #pixart
            return self.__returner()

    def process_vae_without_12(self):
        self.tag = "VAE-"
        if self.extract.get("mmdit", 0) == 8:
            self.tag = f"{self.tag}{self.vae_peek['167333134']['248']}" #sxl hook
            return self.__returner()
        elif isclose(int(self.extract.get("size", 0)), 334641190, rel_tol=self.vae_xl_pct):
            self.tag = f"{self.tag}{self.vae_peek['334641190']['250']}"
        else:
            self.tag = f"{self.tag}{self.vae_peek['334641162']['250']}" #sxl hook
            return self.__returner()
            

    def process_lora(self):
        self.tag = "LORA-"

        for size, attributes in self.lora_peek.items():
            if (
                isclose(int(self.extract.get("size", 0)), int(size),   rel_tol=self.lora_pct) or
                isclose(int(self.extract.get("size", 0)), int(size)*2, rel_tol=self.lora_pct) or
                isclose(int(self.extract.get("size", 0)), int(size)/2, rel_tol=self.lora_pct)
            ):
                for tensor_params, desc in attributes.items():
                    if isclose(int(self.extract.get("tensor_params", 0)), int(tensor_params), rel_tol=self.lora_pct):
                        rep_count = 0
                        for each in next(iter([desc, 0])):
                            if rep_count <= 1:
                                if each.lower() not in str(self.extract.get('filename', 0)).lower():
                                    rep_count += 1
                                else:
                                    self.tag = f"{self.tag}{each}-"
                                    model = desc[len(desc)-1]
                                    self.tag = self.tag + model
                                    return self.__returner()  # found lora

    def process_tf(self):
        self.tag = "TRA-"

        for tensor_params, attributes in self.tf_peek.items():
            if isclose(int(self.extract.get("tensor_params", 0)), int(tensor_params), rel_tol=self.tf_leeway):
                for shape, name in attributes.items():
                    try:
                        if isclose(self.extract.get("shape", 0)[0], int(shape), rel_tol=self.tf_pct):
                            if isclose(int(self.extract.get("transformers", 0)), name[0], rel_tol=self.tf_pct):
                                self.tag = f"{self.tag}{name[1]}" # found transformer
                                return self.__returner()
                    except ValueError as error_log:
                        logger.debug(f"'[No shape key for transformer '{self.extract}''{error_log}'.", exc_info=True)
                        self.tag = f"{name[1]} estimate"
                        return self.__returner()  # estimated transformer

    def process_model(self):
        self.tag = ""
        self.unrecognized = ""
        for tensor_params, attributes, in self.model_peek.items():
            if isclose(int(self.extract.get("tensor_params", 0)), int(tensor_params), rel_tol=self.model_tensor_pct):
                for shape, name in attributes.items():
                    try:
                        if isclose(self.extract.get("shape", 0)[0], int(shape), rel_tol=self.model_block_pct):
                            if self.extract.get("diffusers", "not_found") != "not_found":
                                if isclose(self.extract.get("diffusers", 0), name[0], rel_tol=self.model_block_pct):
                                    self.tag = name[1]
                            elif self.extract.get("mmdit", "not_found") != "not_found":
                                if isclose(self.extract.get("mmdit", 0), name[0], rel_tol=self.model_block_pct):
                                    self.tag = name[1]
                            elif self.extract.get("flux", "not_found") != "not_found":
                                if isclose(self.extract.get("flux", 0), name[0], rel_tol=self.model_block_pct):
                                    self.tag = name[1]
                            elif self.extract.get("diffusers_lora", "not_found") != "not_found":
                                if isclose(self.extract.get("diffusers_lora", 0),  name[0], rel_tol=self.model_block_pct):
                                    self.tag = name[1]
                    except TypeError as error_log:
                        logger.debug(f"'[No block data '{self.extract}''{error_log}'.", exc_info=True)
                        self.unrecognized = "[]~~:"  # no block guess
                        self.tag = name[1]
                    except KeyError as error_log:
                        logger.debug(f"'[No shape key for model '{self.extract}''{error_log}'.", exc_info=True)
                        self.unrecognized = "_~~" #no model shape guess
                        self.tag = name[1]
                    else:
                            self.unrecognized = "~~"
                            self.tag = name[1]
                    finally:
                        #print(f"{self.tag}, VAE-{self.tag}:{self.vae_inside}, CLI-{self.tag}:{self.clip_inside} - {self.unrecognized}")
                        return self.__returner()  # found model

class ReadMeta:
    # ReadMeta.data(filename,full_path_including_filename)
    # scan the header of a tensor file and discover its secrets
    # return a dict of juicy info

    def __init__(self, path):
        self.path = path  # the path of the file
        self.full_data, self.meta, self.count_dict = {}, {}, {}

        # level of certainty, counts tensor block type matches
        ## DO NOT CHANGE THESE VALUES
        ## they may be labelled innacurately, ignore it
        self.known = known

        self.model_tag = {  # measurements and metadata detected from the model ggml.model imatrix.chunks_count
            #NO TOUCH!! critical values
            "filename": "", #universal
            "size": "", #file size in bytes
            "path": "",
            "dtype": "", #precision
            "tensor_params": 0, #tensor count
            "shape": "", #largest first dimension of tensors
            "general.name": "", # llm tags
            "general.architecture": "",
            "block_count":"",
            "context_length": "", #length of messages
            "type": "", 
            "attention.head_count": "",
            "attention.head_count_kv": "",
            "general.name": "",
            "tokenizer.ggml.model": "", #llm tags end here
            "__metadata__": "", #extra
            "data_offsets": "", #universal
            "general.basename": "",
            "name": "", #usually missing...

        }
        self.occurrence_counts = defaultdict(int)
        self.filename = os.path.basename(self.path)  # the title of the file only
        self.ext = Path(self.filename).suffix.lower()

        if not os.path.exists(self.path):  # be sure it exists, then proceed
            logger.debug(f"'[Not found '{self.filename}'''.", exc_info=True)
            raise RuntimeError(f"Not found: {self.filename}")

        else:
            self.model_tag["filename"] = self.filename
            self.model_tag["extension"] = self.ext.replace(".","")
            self.model_tag["path"] = self.path
            self.model_tag["size"] = os.path.getsize(self.path)

    def _parse_safetensors_metadata(self):
        try:
            with open(self.path, "rb") as json_file:
                """
                try opening file
                """
        except:
            log = f"Could not open file {self.path}"
            logger.debug(log, exc_info=True)
            print(log)
        else:
            with open(self.path, "rb") as json_file:
                header = struct.unpack("<Q", json_file.read(8))[0]
                try:
                    return json.loads(json_file.read(header), object_hook=self._search_dict)
                except:
                    log = f"Path not found'{self.path}'''."
                    logger.debug(log, exc_info=True)
                    return print(log)
            
    def _parse_gguf_metadata(self):
        try:
            with open(self.path, "rb") as llama_file:
                """
                try opening file
                """
        except:
            log = f"Could not open file {self.path}"
            logger.debug(log, exc_info=True)
            print(log)
        else:
            with open(self.path, "rb") as llama_file:
                magic = llama_file.read(4)
                if magic != b"GGUF":
                    print(f"{magic} vs. b'GGUF'. wrong magic #") # uh uh uh. you didn't say the magic word
                else:
                    llama_ver = struct.unpack("<I", llama_file.read(4))[0]
                    if llama_ver < 2:
                        print(f"{llama_ver} / needs GGUF v2+ ")
                    else:
                        try:
                            parser = Llama(model_path=self.path,vocab_only=True,verbose=False)
                            self.meta = parser.metadata
                            return self._search_dict(self.meta)
                        except ValueError as error_log:
                            logger.debug(f"'[Failed load '{self.path}''{error_log}'.", exc_info=True)
                            print(f'Llama angry! Unknown Type : {self.filename}')
                            pass
                        except OSError as error_log:
                            logger.debug(f"'[Failed access '{self.path}''{error_log}'.", exc_info=True)
                            print(f'Llama angry! :  {self.filename}')
                            pass
                        except:
                            logger.exception(Exception)
                            logger.debug(f"'[Failed unpack '{self.path}''{error_log}'.", exc_info=True)
                            return print(f"Sorry... Error loading ._. : {self.filename}")

    def data(self):
            if self.ext == ".pt" or self.ext == ".pth" or self.ext == ".ckpt":  # process closer to load
                pass
            elif self.ext == ".safetensors" or self.ext == "":
                self.occurrence_counts.clear()
                self.full_data.clear()
                self.meta = self._parse_safetensors_metadata()
                self.full_data.update((k, v) for k, v in self.model_tag.items() if v != "")
                self.full_data.update((k, v) for k, v in self.count_dict.items() if v != 0)
                #for k, v in self.full_data.items():  # uncomment to view model properties
                #   print(k, v)              ######################################DEBUG
                self.count_dict.clear()
                self.model_tag.clear()
                self.meta = ""
            elif self.ext == ".gguf":
                self.occurrence_counts.clear()
                self.full_data.clear()
                self.meta = self._parse_gguf_metadata()
                self.full_data.update((k, v) for k, v in self.model_tag.items() if v != "")
                self.full_data.update((k, v) for k, v in self.count_dict.items() if v != 0)
                #for k, v in self.full_data.items():  # uncomment to view model properties
                #    print(k, v)              ######################################DEBUG
                self.count_dict.clear()
                self.model_tag.clear()
                self.meta = ""
            elif self.ext == ".bin":  # placeholder - parse bin metadata(path) using ...???
                    self.meta = ""
                    pass
            else :
                print(f"Unrecognized file format: {self.filename}")
                pass

            return self.full_data

    def _search_dict(self, meta):
        self.meta = meta
        if self.ext == ".gguf":
            for key, value in self.meta.items():
                #print(f"{key} {value}")              ######################################DEBUG
                if self.model_tag.get(key, "not_found") != "not_found":
                    self.model_tag[key] = self.meta.get(key)  # drop it like its hot
                if self.model_tag.get("general.architecture", "") != "":
                    prefix = self.model_tag["general.architecture"]
                    prefixless_key = key.replace(f"{prefix}.","")
                    if self.model_tag.get(prefixless_key, "not_found") != "not_found":
                        self.model_tag[prefixless_key] = value  # drop it like its hot
        elif self.ext == ".safetensors" or self.ext == ".sft":
            for num in list(self.meta): # handle inevitable exceptions invisibly
                if self.model_tag.get(num, "not_found") != "not_found":
                    self.model_tag[num] = self.meta.get(num)  # drop it like its hot
                if "dtype" in num:
                        self.model_tag["tensor_params"] += 1  # count tensors
                elif "shape" in num: # measure first shape size thats returned
                    if self.meta.get(num) > self.model_tag["shape"]:
                        self.model_tag["shape"] = self.meta.get(num)
                if "data_offsets" not in num:
                    if ("shapes" or "dtype") not in num:
                        for block, model_type in self.known.items():  # model type, dict data
                            if block in num:  # if value matches one of our key values
                                self.occurrence_counts[model_type] += 1 # count matches
                                self.count_dict[model_type] = self.occurrence_counts.get(model_type, 0)  # pair match count to model type

        return self.meta


class ModelIndexer:
    def __init__(self):
        pass

