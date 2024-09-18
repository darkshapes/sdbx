import os
import ctypes
import json
import struct
import multiprocessing as mp
from multiprocessing import Process
from time import process_time_ns
from math import isclose
from pathlib import Path
from collections import defaultdict
import llama_cpp
from llama_cpp import Llama
from sdbx import config, logger


print(f'begin: {process_time_ns()*1e-6} ms')

# def _path_dict():
#     root = {
#         n: os.path.join() for n, p in dict(self.location).items() # see self.location for details
#     }

#     for n, p in dict(self.location).items():
#         if ".." in p:
#             raise Exception("Cannot set location outside of config path.")

#     models = {f"models.{name}": os.path.join(root["models"], name) for name in self.get_default("directories", "models")}

#     return {**root, **models}

# def get_default(name, prop):
#     return self._defaults_dict[name][prop]


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

    model_peek, vae_peek, tf_peek, vae_peek_12, lora_peek, = defaultdict(dict), defaultdict(dict),  defaultdict(dict), defaultdict(dict), defaultdict(dict)

    # sd1.5  4265097012 3188362889 4244098984 4244099040
            # TP     Shape   MMD/DF
    model_peek[1131][1280] = [224, "STA-15"]
    # sdxl 6938040706+6938040714+6938041050+6938052474+6941318844+6938047840/6
    model_peek[2515][1280] = [980, "STA-XL"]
    # sd3, no text_encoder or vae
    model_peek[735][1536] = [286, "STA-3M"]
    model_peek[1448][768] = [550, "STA-3C"]  # sd3 no t5
    # sd3 with 1/4 precision(int8) t5
    model_peek[1668][32128] = [550, "STA-34"]
    # sd3 with 1/2 precision(float16) t5
    model_peek[1668][32128] = [550, "STA-32"]
    model_peek[1682][320] = [980, "KOL-01"]  # kolors
    model_peek[786][3072] = [2, "FLU-01"]  # flux
    model_peek[780][3072] = [204, "FLU-1D"]  # flux full size
    model_peek[776][3072] = [2, "FLU-1S"]  # flux full size
    model_peek[824][8] = [208, "AUR-03"]  # auraflow
    model_peek[612][6] = [560, "PIX-AL"]  # pixart alpha
    model_peek[437][1152] = [1262008531, "PIX-SI"]  # pixart sigma
    model_peek[2166][4309802] = [162, "HUN-12"]  # hunyuan dit
    model_peek[1220][81] = [640, "LUM-01"]  # lumina

    model_peek[1131][3188362889] = "STA-15"  # sd1.5
    model_peek[1065][3188362889] = "STA-15"
    model_peek[1131][4244099040] = "STA-15"

    # VAE-
    vae_peek_12[167666902][244] = "FLU-01"
    vae_peek_12[167335342][248] = "AUR-03-"
    vae_peek_12[167335343][248] = "KOL-01"
    vae_peek_12[334643238][248] = "PIX-SI"
    vae_peek_12[334643268][248] = "PIX-AL"
    vae_peek_12[167335344][248] = "SGM-VG"  # segmind vega

    vae_peek[335304388][244] = "FLU-01"
    vae_peek[167666902][244] = "KOL-01"
    vae_peek[404581567][304] = "STA-15"
    vae_peek[334641190][250] = "STA-15"
    vae_peek[114560782][248] = "STA-15"

    vae_peek[167333134][248] = "STA-XL"
    vae_peek[334641162][250] = "STA-XL"

    # CLIP-
    tf_peek[220][32128] = [49, "T5X-XL"]  # t5x mmdits
    tf_peek[258][32128] = [62, "T5V-11"]  # t5 base dit
    tf_peek[517][1280] = [2, "CLI-VG"]  # clip g xl 3
    tf_peek[1722][90] = [220, "CLI-OP"]  # open clip
    tf_peek[520][1024] = [4, "CLI-VH"]  # clip h
    tf_peek[242][32128] = [49, "T5X-PI"]  # umt5 pile-t5xl auraflow
    tf_peek[196][768] = [2, "CLI-VL"]  # clip l 15 xl 3
    tf_peek[312][4608] = [57, "CLI-G3"]  # chatglm3 auraflow

    # LORA-
    lora_peek[134621524][834] = ["PCM", "STA-15"]
    lora_peek[1][566] = ["PCM", "STA-15"]
    lora_peek[393854592][2364] = ["PCM", "DMD", "STA-XL"]
    lora_peek[103507000][534] = ["PCM", "STA-3"]
    lora_peek[21581612][1120] = ["SPO", "STA-XL"]
    lora_peek[371758976][1680] = ["SPO", "FLD", "STA-XL"]
    lora_peek[371841776][1680] = ["SPO", "STA-XL"]
    lora_peek[46687104][1680] = ["SPO", "STA-XL"]
    lora_peek[3227464][256] = ["SPO", "STA-15"]
    lora_peek[269127064][834] = ["HYP", "STA-15"]
    lora_peek[787359648][2166] = ["HYP", "STA-XL"]
    lora_peek[787359649][2364] = ["HYP", "STA-XL"]
    lora_peek[472049168][682] = ["HYP", "STA-3"]
    lora_peek[1388026440][1008] = ["HYP", "FLU-1D"]
    lora_peek[1][1562] = ["TCD", "STA-XL"]
    lora_peek[393854624][2364] = ["TCD", "STA-XL"]
    lora_peek[134621556][834] = ["LCM", "TCD", "STA-15"]
    lora_peek[393855224][2364] = ["LCM", "STA-XL"]
    lora_peek[393854624][2364] = ["LCM", 'TCD', "STA-XL"]
    lora_peek[509550144][3156] = ["FLA", "STA-XL"]
    lora_peek[239224692][699] = ["RT", "STA-VG"]

    def __init__(self, extract):
        self.extract = extract
        self.clip_inside = False
        self.vae_inside = False

    def __returner(self, tag):
        self.tag = tag
        print(f"{self.tag} {self.extract.get("extension", "")} {self.extract.get("filename", "")}") # logger.debug
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
            #logger.exception(error_log)
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
                                #logger.exception(error_log)
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
                        #logger.exception(error_log)
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
                        #logger.exception(error_log)
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
        self.known = {  # dict of tensor block keys with tag values
            "adaLN_modulation": "mmdit",
            "mlp.fc": "mmdit",
            "mlpX": "mmdit",
            "w1q": "mmdit",
            "self_attn.out_proj": "mmdit",
            "w1q": "mmdit",
            "w1o": "mmdit",
            "mlpX.c_proj": "mmdit",
            "mlpC.c_proj": "mmdit",
            "w2q.": "mmdit",
            "w2k.": "mmdit",
            "w2v.": "mmdit",
            "w2o.": "mmdit",
            "w1k.": "mmdit",
            "w1v.": "mmdit",
            "mlpX.c_fc": "mmdit",
            "mlpX.c_proj.": "mmdit",
            "mlpC.c_fc": "mmdit",
            "modC.": "mmdit",
            "modX.": "mmdit",
            "model.register_tokens": "auraflow",
            "model.positional_encoding": "auraflow",
            "model.init_x_linear": "auraflow",
            "model.t_embedder.mlp.": "auraflow",
            "model.t_embedder.mlp.": "auraflow",
            "modCX": "flux",
            "img_attn.proj": "flux",
            "time_in.in_layer": "flux",
            "time_in.out_layer": "flux",
            "vector_in.in_layer": "flux",
            "vector_in.out_layer": "flux",
            "guidance_in.in_layer": "flux",
            "guidance_in.in_layer": "flux",
            "txt_in": "flux",
            "img_in": "flux",
            "img_mod.lin": "flux",
            "txt_mod.lin": "flux",
            "img_attn.qkv": "flux",
            "txt_attn.qkv": "flux",
            "t_embedder.mlp": "pixart_s",
            "y_embedder.embedding_table": "pixart_s",
            "wkqv.": "hunyuan",
            "wqkv.": "hunyuan",
            "q_norm": "hunyuan",
            "k_norm": "hunyuan",
            "out_proj": "hunyuan",
            "kq_proj": "hunyuan",
            "default_modulation.": "hunyuan",
            "pooler": "hunyuan",
            "t_embedder": "hunyuan",
            "x_embedder": "hunyuan",
            "mlp_t5": "hunyuan",
            "time_extra_emb.extra_embedder": "hunyuan",
            "to_q": "diffusers",
            "to_k": "diffusers",
            "to_v": "diffusers",
            "norm_q": "diffusers",
            "norm_k": "diffusers",
            "to_out": "diffusers",
            "norm1.norm": "diffusers",
            "norm1.linear": "diffusers",
            "ff.net.0": "diffusers",
            "ff.net.2": "diffusers",
            "time_extra_emb": "diffusers",
            "time_embedding": "diffusers",
            "pos_embd": "diffusers",
            "text_embedder": "diffusers",
            "extra_embedder": "diffusers",
            "attn.norm_added_q": "diffusers",
            "skip.connection": "sdxl",
            "in.layers.2": "sdxl",
            "out.layers.3": "sdxl",
            "upsamplers": "sdxl",
            "downsamplers": "sdxl",
            "op": "sd",
            "in_layers": "sd",
            "out_layers": "sd",
            "emb_layers": "sd",
            "skip_connection": "sd",
            "text_model": "diffusers_lora",
            "self_attn": "diffusers_lora",
            "to_q_lora": "diffusers_lora",
            "to_k_lora": "diffusers_lora",
            "to_v_lora": "diffusers_lora",
            "to_out_lora": "diffusers_lora",
            "text_projection": "diffusers_lora",
            "to.q.lora": "unet_lora",
            "to.k.lora": "unet_lora",
            "to.v.lora": "unet_lora",
            "to.out.0.lora": "unet_lora",
            "proj.in": "unet_lora",
            "proj.out": "unet_lora",
            "emb.layers": "unet_lora",
            "proj_in.": "transformers",
            "proj_out.": "transformers",
            "norm.": "transformers",
            "norm1.": "unet",
            "norm2.": "unet",
            "norm3.": "unet",
            "attn1.to_q.": "unet",
            "attn1.to_k.": "unet",
            "attn1.to_v.": "unet",
            "attn1.to_out.0.": "unet",
            "attn2.to_q.": "unet",
            "attn2.to_k.": "unet",
            "attn2.to_v.": "unet",
            "attn2.to_out.0.": "unet",
            "ff.net.0.proj.": "unet",
            "ff.net.2.": "unet",
        }

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
                return print(f"Error loading {full_path}")
            
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
                for k, v in self.full_data.items():  # uncomment to view model properties
                    print(k, v)
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
                #    print(k, v)
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
                #print(f"{key} {value}")
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


config_path = os.path.join(os.environ.get('LOCALAPPDATA', os.path.join(os.path.expanduser('~'), 'AppData', 'Local')), 'Shadowbox')
search_path = "models"
path_name =  os.path.normpath(os.path.join(config_path, search_path, "download")) #multi read

each = "TCD_XL_pytorch_lora_weights.safetensors" ##SCAN SINGLE FILE
full_path = os.path.normpath(os.path.join(path_name, each)) #multi read
metareader = ReadMeta(full_path).data(full_path)
evaluate = EvalMeta(metareader).data


# for each in os.listdir(path_name): ###SCAN DIRECTORY
#     filename = each  # "PixArt-Sigma-XL-2-2K-MS.safetensors"
#     full_path = os.path.join(path_name, filename)
#     metareader = ReadMeta(full_path).data(full_path)
#     if metareader is not None:
#         evaluate = EvalMeta(metareader).data()
# path_name = os.path.join(os.environ.get('LOCALAPPDATA', os.path.join(os.path.expanduser('~'), 'AppData', 'Local')), 'Shadowbox',"models","download")

# for each in os.listdir(path_name): ###SCAN DIRECTORY
#       if not os.path.isdir(os.path.join(path_name, each)):
#         filename = each  # "PixArt-Sigma-XL-2-2K-MS.safetensors"
#         full_path = os.path.join(path_name, filename)
#         metareader = ReadMeta(full_path).data(full_path)
#         if metareader is not None:
#             evaluate = EvalMeta(metareader).data()



print(f'end: {process_time_ns()*1e-6} ms')
