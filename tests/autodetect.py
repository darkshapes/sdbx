import os
import json
import struct

from math import isclose
from pathlib import Path
from collections import defaultdict

from sdbx import config, logger

peek = config.get_default("tuning", "peek")
known = config.get_default("tuning", "known")

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
    model_size_pct = 7e-2
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
    
    def __returner(self):
        logger.debug(f"{self.tag} {self.extract.get("filename", "")}")
    
    def data(self):
        if int(self.extract.get("unet", 0)) == 96:  # Check VAE
            self.vae_inside = True
            if int(self.extract.get("sdxl", 0)) == 12:
                self.process_vae_12()
            else:
                self.process_vae()     

        if int(self.extract.get("diffusers", 0)) > 256:  # Check LoRA
            self.process_lora()

        if int(self.extract.get("transformers", 0)) >= 2:  # Check CLIP
            self.clip_inside = True
            self.process_tf()

        if int(self.extract.get("size", 0)) > 1e9:  # Check model
            self.process_model()

        raise ValueError(f"Unknown model type '{self.extract}'.")
        
    def process_vae_12(self):
        tag = "VAE-"

        if self.extract.get("mmdit", 0) != 4:
            model = f"{tag}{vae_peek_12[167335343][248]}"
            return self.__returner(model)  #kolors hook  
        else:
            for size, attributes in vae_peek_12.items():
                if (isclose(int(self.extract.get("size", 0)), size, rel_tol=self.vae_pct)
                    or isclose(int(self.extract.get("size", 0)), size*2, rel_tol=self.vae_pct) 
                    or isclose(int(self.extract.get("size", 0)), size/2, rel_tol=self.vae_pct)
                    or int(self.extract.get("size", 0) == size)):
                        for tensor_params, model in attributes.items():
                            if isclose(int(extract.get("tensor_params", 0)), tensor_params, rel_tol=self.vae_pct): 
                                try:
                                    if 512 == extract.get("shape", 0)[0]:
                                        model = f"{tag}-{vae_peek_12[167666902][244]}" 
                                        return self.__returner(model) #flux hook
                                except:
                                    pass
                                model = f"{tag}{model}" 
                                return self.__returner(model) #found vae 12
    
    def process_vae(self):
        tag = "VAE-"

        if isclose(self.extract.get("size", 0), 284594513, rel_tol=self.vae_sd1_pct):
            if isclose(self.extract.get("tensor_params",0), 276, rel_tol=self.vae_sd1_tp_pct):
                try:
                    if self.extract.get("shape", 0)[0] == 32:
                        model = f"{tag}{model}" 
                        return self.__returner(model)  # sd1 hook
                except:
                    model = f"{tag}{model}" 
                    return self.__returner(model)  # sd1 hook
        else:
            for size, attributes in vae_peek.items():
                    if (
                        isclose(int(self.extract.get("size", 0)), 167333134, rel_tol=self.vae_xl_pct) or
                        isclose(int(self.extract.get("size", 0)), 334641162, rel_tol=self.vae_xl_pct)
                    ):
                        for tensor_params, model in attributes.items():
                            if isclose(int(self.extract.get("tensor_params", 0)),249, rel_tol=self.vae_pct):
                                model = f"{tag}-{vae_peek[167333134][248]}" 
                                return self.__returner(model) # flux hook     
                    
                    if (
                        isclose(int(self.extract.get("size", 0)), size,   rel_tol=self.vae_size_pct) or
                        isclose(int(self.extract.get("size", 0)), size*2, rel_tol=self.vae_size_pct) or
                        isclose(int(self.extract.get("size", 0)), size/2, rel_tol=self.vae_size_pct)
                    ):
                        for tensor_params, model in attributes.items():
                            if isclose(int(extract.get("tensor_params", 0)), tensor_params, rel_tol=self.vae_pct): 
                                try:
                                    if 512 == extract.get("shape", 0)[0]:
                                        if extract.get("transformers", 0) == 4:
                                            model = f"{tag}{vae_peek_12[167335343][248]}"
                                            return self.__returner(model)  #kolors hook 
                                        else:
                                            model = f"{tag}-{vae_peek[335304388][244]}" 
                                            return self.__returner(model) #flux hook
                                except:
                                    pass
                                else:
                                    model = f"{tag}{model}" 
                                    return self.__returner(model)#found vae

    def process_lora(self):
        tag = "LORA-"

        for size, attributes in lora_peek.items():
            if (
                isclose(int(self.extract.get("size", 0)), size,   rel_tol=self.lora_pct) or 
                isclose(int(self.extract.get("size", 0)), size*2, rel_tol=self.lora_pct) or 
                isclose(int(self.extract.get("size", 0)), size/2, rel_tol=self.lora_pct)
            ): 
                for tensor_params, desc in attributes.items():
                    if isclose(int(self.extract.get("tensor_params", 0)), tensor_params, rel_tol=self.lora_pct):
                            rep_count = 0
                            for each in next(iter([desc,0])):
                                if rep_count <= 1:
                                    if each.lower() not in str(self.extract.get('filename',0)).lower():
                                        rep_count += 1
                                    else:
                                        tag = f"{tag}{each}"
                                        model = desc[len(desc)-1]
                                        model = tag + "-" + model
                                        return self.__returner(model)  # found lora
    
    def process_tf(self):
        tag = "CLI-"

        for tensor_params, attributes in tf_peek.items():
            if isclose(int(self.extract.get("tensor_params", 0)), tensor_params, rel_tol=self.tf_leeway):
                for shape, model in attributes.items():
                    try:
                        if isclose(self.extract.get("shape", 0)[0], shape, rel_tol=self.tf_pct):
                            if isclose(int(self.extract.get("transformers", 0)), model[0], rel_tol=self.tf_pct):
                                model = model[1]
                                model = f"{model}"
                                return self.__returner(model)  # found transformer
                    except:
                        # if model[1]: model = model[1]
                        model = model[1] or model
                        model = f"{model} estimate"
                        return self.__returner(model)  # estimated transformer

    def process_model(self):
        for tensor_params, attributes, in model_peek.items():
            if isclose(int(self.extract.get("tensor_params", 0)), tensor_params, rel_tol=self.model_tensor_pct):
                for shape, model in attributes.items():
                    try:
                        if isclose(self.extract.get("shape", 0)[0], shape, rel_tol=self.model_shape_pct):
                            if (
                                isclose(int(self.extract.get("size", 0)), model[0],   rel_tol=self.model_size_pct) or 
                                isclose(int(self.extract.get("size", 0)), model[0]*2, rel_tol=self.model_size_pct) or
                                isclose(int(self.extract.get("size", 0)), model[0]/2, rel_tol=self.model_size_pct)
                            ):
                                model =  model[1]
                                print(f"{model}, VAE-{model}:{self.vae_inside}, CLI-{model}:{self.clip_inside}")
                                return self.__returner(model) #found model
                    except:
                            model =  model[1]
                            print(f"{model}, VAE-{model}:{self.vae_inside}, CLI-{model}:{self.clip_inside}")
                            return self.__returner(model) #found model

class ReadMeta:
    # level of certainty, counts tensor block type matches
    full_data, meta, count_dict = {}, {}, {}
    occurrence_counts = defaultdict(int)

    def __init__(self, path):
        self.model_tag = {  # measurements and metadata detected from the model
            "filename": "",
            "size": "",
            "dtype": "",
            "tensor_params": 0,
            "shape": "",
            "data_offsets": "",
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

        self.path = path  # the path of the file
        self.filename = os.path.basename(path)  # the title of the file only
        self.ext = Path(filename).suffix

        if not os.path.exists(self.path):  # be sure it exists, then proceed
            raise RuntimeError(f"Not found: {self.filename}")
        else:
            self.model_tag["filename"] = self.filename
            self.model_tag["size"] = os.path.getsize(self.path)

    def _parse_safetensors_metadata(self):
        with open(self.path, "rb") as json_file:
            header = struct.unpack("<Q", json_file.read(8))[0]
            try:
                return json.loads(json_file.read(header), object_hook=self._search_dict)
            except:
                return print(f"error loading {self.path}")

    def data(self):
        if self.ext in {".pt", ".pth", ".ckpt"}:  # process elsewhere
            return
        elif self.ext in {".safetensors" or ".sft"}:
            self.occurrence_counts.clear()
            self.full_data.clear()
            self.meta = self._parse_safetensors_metadata(self.path)
            self.full_data.update((k, v)
                                  for k, v in self.model_tag.items() if v != "")
            self.full_data.update((k, v)
                                  for k, v in self.count_dict.items() if v != 0)
            self.count_dict.clear()
            self.model_tag.clear()
        elif self.ext is ".gguf":
            # placeholder - parse gguf metadata(path) using llama lib
            meta = ""
        elif self.ext is ".bin":
            meta = ""  # placeholder - parse bin metadata(path) using ...???
        else:
            raise RuntimeError(f"Unrecognized file format: {self.filename}")

        return self.full_data

    def _search_dict(self, meta):
        self.meta = meta
        for num in list(self.meta):
            # handle inevitable exceptions invisibly
            if self.model_tag.get(num, "not_found") != "not_found":
                self.model_tag[num] = meta.get(num)  # drop it like its hot
            
            if "dtype" in num:
                self.model_tag["tensor_params"] += 1  # count tensors
            elif "shape" in num:
                # measure first shape size thats returned
                if meta.get(num) > self.model_tag["shape"]:
                    self.model_tag["shape"] = self.meta.get(num)
            
            if "data_offsets" not in num:
                if ("shapes" or "dtype") not in num:
                    for block, model_type in self.known.items():  # model type, dict data
                        if block in num:  # if value matches one of our key values
                            # count matches
                            self.occurrence_counts[model_type] += 1
                            self.count_dict[model_type] = self.occurrence_counts.get(
                                model_type, 0)  # pair match count with type of model

        return meta


def folder_run(path)
    for filename in os.listdir(path):
        fp = os.path.normpath(os.path.join(path, filename))  # full path
        if not os.path.isdir(filename):
            reader = ReadMeta(fp)  # object instantiation
            data = reader.data()  # method call
            return data
        else:
            print("no fies??")

def single_run(path, name)
    fp = os.path.join(path, name)  # full path
    if not os.path.isdir(fp):
        reader = ReadMeta(fp)
        data = reader.data()
        return data
    else:
        print("no fies??")
            
models_path = config.get_path("models")
search_path = "models"
metadata = folder_run(models_path)
#metareader = single_run(config_path, search_path, search_name)
#for k, v in metareader.items():            #uncomment to view model properties
#    print(k,v)
evalmeta = EvalMeta(metadata).data()  # class instantiation and method call in one-liner

# user select model
# match loader code to model type
# check lora availability
# if no lora, check if AYS or other optimization available
# MATCH LORA TO MODEL TYPE BY PRIORITY
#     PCM            
#     SPO
#     HYP
#     LCM
#     RT
#     TCD
#     FLA
# match loader code to model type
# ram calc+lora
# decide sequential offload on 75% util/ off 50% util
# match dtype to lora load
# MATCH VAE TO MODEL TYPE
# match dtype to vae load


#except:
#    pass
#else:
#    each = "klF8Anime2VAE_klF8Anime2VAE.safetensors" #ae.safetensors #noosphere_v42.safetensors #luminamodel .safetensors #mobius-fp16.safetensors

# def prioritize