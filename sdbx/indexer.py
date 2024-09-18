import os
import json
import struct

from math import isclose
from pathlib import Path
from collections import defaultdict

from llama_cpp import Llama
from sdbx import config, logger
from sdbx.config import get_default


#### CLASS ReadMeta
#### METHODS data
#### SYNTAX instance_name = ReadMeta(full_path_to_file).data(full_path_to_file)
#### OUTPUT dict

#### CLASS EvalMeta
#### METHODS, process_vae, process_vae_no_12, process_lora, process_tf, process_model 
#### SYNTAX  instance_name = EvalMeta(dict_metadata_from_ReadMeta).data
#### OUTPUT str

peek = get_default("tuning", "peek") # block and tensor values for identifying
known = get_default("tuning", "known") # raw block & tensor data

model_peek = peek['model_peek']
vae_peek_12 = peek['vae_peek_12']
vae_peek = peek['vae_peek']
tf_peek = peek['tf_peek']
lora_peek = peek['lora_peek']

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
    vae_pct = 1e-2
    vae_size_pct = 1e-2
    vae_xl_pct = 1e-9
    vae_sd1_pct = 3e-2
    vae_sd1_tp_pct = 9e-2
    tf_pct = 1e-4
    tf_leeway = 0.03
    lora_pct = 0.05

    def __init__(self, extract):
        self.extract = extract
        self.clip_inside = False
        self.vae_inside = False

    def __returner(self, tag):
        self.tag = tag
        print(f"{self.tag} {self.extract.get("extension", "")} {self.extract.get("filename", "")}")   ###################################### DEBUG
        return self.tag

    def data(self):
        extract = self.extract
        if int(self.extract.get("unet", 0)) > 96: self.vae_inside = True
        if int(self.extract.get("unet", 0)) == 96:  # Check VAE
            self.process_vae()
        if int(self.extract.get("diffusers", 0)) > 256:  # Check LoRA
            self.process_lora()

        if int(self.extract.get("transformers", 0)) >= 2:  # Check CLIP
            self.clip_inside = True
            self.process_tf(extract, )
        try:
            if int(self.extract.get("size", 0)) > 1e9:  # Check model
                self.process_model()
        except ValueError as error_log:
            logger.exception(error_log)
            print(f'no model found {error_log}')
            print(f"Unknown model type '{self.extract}'.")
        else:
            if self.extract.get("general.name","") != "":
                model = self.extract.get("general.name","")
                return self.__returner(model)

    def process_vae(self):
        if int(self.extract.get("sdxl", 0)) == 12:
            self.process_vae_12()
        else:
            self.process_vae_no_12()

    def process_vae_12(self):
        tag = "VAE-"

        if self.extract.get("mmdit", 0) != 4:
            model = f"{tag}{self.vae_peek_12[167335343][248]}"
            return self.__returner(model)  # kolors hook
        else:
            for size, attributes in self.vae_peek_12.items():
                if (isclose(int(self.extract.get("size", 0)), size, rel_tol=self.vae_pct)
                    or isclose(int(self.extract.get("size", 0)), size*2, rel_tol=self.vae_pct)
                    or isclose(int(self.extract.get("size", 0)), size/2, rel_tol=self.vae_pct)
                        or int(self.extract.get("size", 0) == size)):
                    for tensor_params, model in attributes.items():
                        if isclose(int(self.extract.get("tensor_params", 0)), tensor_params, rel_tol=self.vae_pct):
                            try:
                                if 512 == self.extract.get("shape", 0)[0]:
                                    model = f"{
                                        tag}{self.vae_peek_12[167666902][244]}"
                                    return self.__returner(model)  # flux hook
                            except:
                                pass
                            model = f"{tag}{model}"
                            return self.__returner(model)  # found vae 12

    def process_vae_no_12(self):
        tag = "VAE-"

        if isclose(self.extract.get("size", 0), 284594513, rel_tol=self.vae_sd1_pct):
            if isclose(self.extract.get("tensor_params", 0), 276, rel_tol=self.vae_sd1_tp_pct):
                try:
                    if self.extract.get("shape", 0)[0] == 32:
                        model = f"{tag}{model}"
                        return self.__returner(model)  # sd1 hook
                except:
                    model = f"{tag}{model}"
                    return self.__returner(model)  # sd1 hook
        else:
            for size, attributes in self.vae_peek.items():
                if (
                    isclose(int(self.extract.get("size", 0)), 167333134, rel_tol=self.vae_xl_pct) or
                    isclose(int(self.extract.get("size", 0)), 334641162, rel_tol=self.vae_xl_pct)
                ):
                    for tensor_params, model in attributes.items():
                        if isclose(int(self.extract.get("tensor_params", 0)), 249, rel_tol=self.vae_pct):
                            model = f"{tag}{self.vae_peek[167333134][248]}"
                            return self.__returner(model)  # flux hook

                if (
                    isclose(int(self.extract.get("size", 0)), size,   rel_tol=self.vae_size_pct) or
                    isclose(int(self.extract.get("size", 0)), size*2, rel_tol=self.vae_size_pct) or
                    isclose(int(self.extract.get("size", 0)), size/2, rel_tol=self.vae_size_pct)
                ):
                    for tensor_params, model in attributes.items():
                        if isclose(int(self.extract.get("tensor_params", 0)), tensor_params, rel_tol=self.vae_pct):
                            try:
                                if 512 == self.extract.get("shape", 0)[0]:
                                    if self.extract.get("transformers", 0) == 4:
                                        model = f"{tag}{
                                            self.vae_peek_12[167335343][248]}"
                                        # kolors hook
                                        return self.__returner(model)
                                    else:
                                        model = f"{
                                            tag}-{self.vae_peek[335304388][244]}"
                                        # flux hook
                                        return self.__returner(model)
                            except KeyError as error_log:
                                logger.exception(error_log)
                                print(f'no shape key {error_log}')
                            else:
                                model = f"{tag}{model}"
                                return self.__returner(model)  # found vae

    def process_lora(self):
        tag = "LORA-"

        for size, attributes in self.lora_peek.items():
            if (
                isclose(int(self.extract.get("size", 0)), size,   rel_tol=self.lora_pct) or
                isclose(int(self.extract.get("size", 0)), size*2, rel_tol=self.lora_pct) or
                isclose(int(self.extract.get("size", 0)), size/2, rel_tol=self.lora_pct)
            ):
                for tensor_params, desc in attributes.items():
                    if isclose(int(self.extract.get("tensor_params", 0)), tensor_params, rel_tol=self.lora_pct):
                        rep_count = 0
                        for each in next(iter([desc, 0])):
                            if rep_count <= 1:
                                if each.lower() not in str(self.extract.get('filename', 0)).lower():
                                    rep_count += 1
                                else:
                                    tag = f"{tag}{each}"
                                    model = desc[len(desc)-1]
                                    model = tag + "-" + model
                                    return self.__returner(model)  # found lora

    def process_tf(self, extract):
        self.extract = extract
        tag = "CLI-"

        for tensor_params, attributes in self.tf_peek.items():
            if isclose(int(self.extract.get("tensor_params", 0)), tensor_params, rel_tol=self.tf_leeway):
                for shape, model in attributes.items():
                    try:
                        if isclose(self.extract.get("shape", 0)[0], shape, rel_tol=self.tf_pct):
                            if isclose(int(self.extract.get("transformers", 0)), model[0], rel_tol=self.tf_pct):
                                model = model[1]
                                model = f"{tag}{model}"
                                # found transformer
                                return self.__returner(model)
                    except ValueError as error_log:
                        logger.exception(error_log)
                        print(f'no shape key {error_log}')
                        model = model[1] or model
                        model = f"{tag}{model} estimate"
                        return self.__returner(model)  # estimated transformer

    def process_model(self):
        for tensor_params, attributes, in self.model_peek.items():
            if isclose(int(self.extract.get("tensor_params", 0)), tensor_params, rel_tol=self.model_tensor_pct):
                for shape, model in attributes.items():
                    try:
                        if isclose(self.extract.get("shape", 0)[0], shape, rel_tol=self.model_block_pct):
                            try:
                                if (isclose(self.extract.get("diffusers", 0), model[0], rel_tol=self.model_block_pct)
                                or isclose(self.extract.get("mmdit", 0), model[0], rel_tol=self.model_block_pct)
                                or isclose(self.extract.get("flux", 0), model[0], rel_tol=self.model_block_pct)
                                or isclose(self.extract.get("diffusers_lora", 0),  model[0], rel_tol=self.model_block_pct)):
                                    model = model[1]
                                else:
                                    print(f"Unrecognized model, guessing -  {model[1]}")
                                    model = model[1]
                                print( f"{model}, VAE-{model}:{self.vae_inside}, CLI-{model}:{self.clip_inside}")
                                return self.__returner(model)  # found model
                            except TypeError as error_log:
                                logger.exception(error_log)
                                print(f" Missing block data, guessing model type-  {model[1]}")
                                model = model[1]
                                print( f"{model}, VAE-{model}:{self.vae_inside}, CLI-{model}:{self.clip_inside}")
                                return self.__returner(model)  # found model
                        
                    except KeyError as error_log:
                        logger.exception(error_log)
                        print(f'no shape key {error_log}')
                        model = model[1]
                        print(
                            f"{model}, VAE-{model}:{self.vae_inside}, CLI-{model}:{self.clip_inside}")
                        return self.__returner(model)  # found model

class ReadMeta:
    # ReadMeta.data(filename,full_path_including_filename)
    # scan the header of a tensor file and discover its secrets
    # return a dict of juicy info

    def __init__(self, path):
        self.full_data, self.meta, self.count_dict = {}, {}, {}

        # level of certainty, counts tensor block type matches
        ## DO NOT CHANGE THESE VALUES
        ## they may be labelled innacurately, ignore it
        self.known = known

        self.model_tag = {  # measurements and metadata detected from the model ggml.model imatrix.chunks_count
            #NO TOUCH!! critical values
            "filename": "", #universal
            "size": "", #file size in bytes
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
        self.path = path  # the path of the file
        self.filename = os.path.basename(self.path)  # the title of the file only
        self.ext = Path(self.filename).suffix.lower()

        if not os.path.exists(self.path):  # be sure it exists, then proceed
            raise RuntimeError(f"Not found: {self.filename}")
        else:
            self.model_tag["filename"] = self.filename
            self.model_tag["extension"] = self.ext.replace(".","")
            self.model_tag["size"] = os.path.getsize(self.path)

    def _parse_safetensors_metadata(self):
        with open(self.path, "rb") as json_file:
            header = struct.unpack("<Q", json_file.read(8))[0]
            try:
                return json.loads(json_file.read(header), object_hook=self._search_dict)
            except:
                return print(f"Error loading {self.full_path}")
            
    def _parse_gguf_metadata(self):
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
                        logger.exception(error_log)
                        print(f'Llama angry! Missing data! : {error_log}')     #research   
                    except OSError as error_log:
                        logger.exception(error_log)
                        print(f'Llama angry! Bad file! :  {error_log}')     #research                                          
                    except:
                        logger.exception(Exception)
                        return print(f"Sorry... Error loading ._. : {self.path}")

    def data(self, path):
            if self.ext == ".pt" or self.ext == ".pth" or self.ext == ".ckpt":  # process closer to load
                return
            elif self.ext == ".safetensors" or self.ext == "":
                self.occurrence_counts.clear()
                self.full_data.clear()
                self.meta = self._parse_safetensors_metadata()
                self.full_data.update((k, v) for k, v in self.model_tag.items() if v != "")
                self.full_data.update((k, v) for k, v in self.count_dict.items() if v != 0)
                #for k, v in self.full_data.items():  # uncomment to view model properties
                #    print(k, v)               ###################################### DEBUG
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
                #    print(k, v)              ###################################### DEBUG
                self.count_dict.clear()
                self.model_tag.clear()
                self.meta = ""
            elif self.ext == ".bin":  # placeholder - parse bin metadata(path) using ...???
                    self.meta = ""
            else :
                print(f"Unrecognized file format: {self.filename}")
                pass

            return self.full_data

    def _search_dict(self, meta):
        self.meta = meta
        if self.ext == ".gguf":
            for key, value in self.meta.items():
                #print(f"{key} {value}")              ###################################### DEBUG
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
                                self.count_dict[model_type] = self.occurrence_counts.get(
                                    model_type, 0)  # pair match count to model type

        return self.meta

class ModelIndexer:
    def __init__(self):
        pass
