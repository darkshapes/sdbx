import os
import json
import struct
import psutil
import platform

from math import isclose
from pathlib import Path
from collections import defaultdict

class EvalMeta:
    
    # CRITERIA THRESHOLDS
    model_pct = 0.01  # % of relative closeness to an actual checkpoint value
    vae_pct = 0.0001
    tf_pct = 0.01
    lora_pct = 0.05
    max_mem = .75  # 75% memory threshold
    safe_mem = .50  # 50% memory threshold
    model_peek, vae_peek, tf_peek, vae_peek_12, lora_peek, = defaultdict(dict), defaultdict(dict),  defaultdict(dict), defaultdict(dict), defaultdict(dict)

    model_peek[780][23802932552] = "FLU-1D"  # flux full size
    model_peek[780][23782506688] = "FLU-1S"  # flux full size
    model_peek[786][11901525888] = "FLU-01"  # flux
    model_peek[824][16463323066] = "AUR-03"  # auraflow
    model_peek[2166][8240228270] = "HUN-12"  # hunyuan dit
    model_peek[1682][5159140240] = "KOL-01"  # kolors
    model_peek[735][4337667306] = "STA-3M"  # sd3, no text_encoder or vae
    model_peek[1448][5973224240] = "STA-3C"  # sd3 no t5
    model_peek[1668][10867168284] = "STA-34" # sd3 with 1/4 precision(int8) t5
    model_peek[1668][15761074532] = "STA-32" # sd3 with 1/2 precision(fp16) t5
    model_peek[437][1262008531] = "PIX-SI"  # pixart sigma
    model_peek[2515][6938040706] = "STA-XL"  # sdxl 
    model_peek[612][2447431856] = "PIX-AL"  # pixart alpha
    model_peek[1131][3188362889] = "STA-15"  # sd1.5
    model_peek[1065][3188362889] = "STA-15"

    # VAE-
    vae_peek_12[167666902][244] = "FLU-01"
    vae_peek_12[167335342][248] = "AUR-03-"
    vae_peek_12[167335343][248] = "KOL-01"
    vae_peek_12[334643238][248] = "PIX-SI"
    vae_peek_12[334643268][248] = "PIX-AL"
    vae_peek_12[167335344][248] = "SGM-VG" # segmind vega

    vae_peek[334641162][250] = "STA-XL"
    vae_peek[334641164][248] = "STA-XL"
    vae_peek[404581567][304] = "STA-15"
    vae_peek[334641190][250] = "STA-15"
    vae_peek[114560782][248] = "STA-15"
    vae_peek[167333134][248] = "STA-XL"
    vae_peek[167333134][250] = "STA-XL"

    # CLIP-
    tf_peek[220][4893934904] = [49,"T5X-XL"] # t5x mmdits
    tf_peek[258][891646390] = [62,"T5V-11"] # t5 base dit
    tf_peek[517][1389382176] = [2,"CLI-VG"] # clip g xl 3
    tf_peek[1722][397696772] = [220,"CLI-OP"] # open clip
    tf_peek[520][1266183328] = [4,"CLI-VH"] # clip h
    tf_peek[242][2950448704] = [49,"T5V-UM"] # umt5 auraflow
    tf_peek[196][246144152] = [2,"CLI-VL"]  # clip l 15 xl 3

    # LORA-
    lora_peek[134621524][834] = ["PCM","STA-15"]
    lora_peek[1][566] = ["PCM","STA-15"]
    lora_peek[393854592][2364] = ["PCM","DMD", "STA-XL"]
    lora_peek[103507000][534] = ["PCM","STA-3"]
    lora_peek[21581612][1120] = ["SPO","STA-XL"]
    lora_peek[371758976][1680] = ["SPO","FLD","STA-XL"]
    lora_peek[371841776][1680] = ["SPO","STA-XL"]
    lora_peek[46687104][1680] = ["SPO","STA-XL"]
    lora_peek[3227464][256] = ["SPO","STA-15"]
    lora_peek[269127064][834] = ["HYP","STA-15"]
    lora_peek[787359648][2166] = ["HYP","STA-XL"]
    lora_peek[787359649][2364] = ["HYP","STA-XL"]
    lora_peek[472049168][682] = ["HYP","STA-3"]
    lora_peek[1388026440][1008] = ["HYP","FLU-1D"]
    lora_peek[1][1562] = ["TCD","STA-XL"]
    lora_peek[134621556][834] = ["LCM", "TCD", "STA-15"]
    lora_peek[393855224][2364] = ["LCM","STA-XL"]
    lora_peek[393854624][2364] = ["LCM","STA-XL"]
    lora_peek[509550144][3156] = ["FLA","STA-XL"]
    lora_peek[239224692][699] = ["RT","STA-VG"]


    def __init__(self):
        dram = psutil.virtual_memory().total
        try:
            memfit = metareader.get("size", 0)/dram
        except:
            print("Unknown file, skipping...")
        else:
            #print(metareader.get("filename", 0))
            # check vae
            clip_inside = False
            vae_inside = False
    
    @classmethod
    def data(self, extract)
        if int(extract.get("unet", 0)) == 96:
            vae_inside = True
            if int(extract.get("sdxl", 0)) == 12:
                if "512" in extract.get("shape", 0):
                    print(vae_peek_12[167666902][244]) #flux
                elif extract.get("mmdit", 0) != 4:
                    print(vae_peek_12[167335343][248])  #kolors   
                else:
                    for size, attributes in vae_peek_12.items():
                        if (isclose(int(extract.get("size", 0)), size, rel_tol=vae_pct)
                            or isclose(int(extract.get("size", 0)), size*2, rel_tol=vae_pct) 
                            or isclose(int(extract.get("size", 0)), size/2, rel_tol=vae_pct)
                            or int(extract.get("size", 0) == size)):
                                for tensor_params, model in attributes.items():
                                    if isclose(int(extract.get("tensor_params", 0)), tensor_params, rel_tol=vae_pct): 
                                        print(f"VAE-{model} found") #found hook
            else:
                for size, attributes in vae_peek.items():
                    if (isclose(int(extract.get("size", 0)), size, rel_tol=vae_pct)
                        or isclose(int(extract.get("size", 0)), size*2, rel_tol=vae_pct) 
                        or isclose(int(extract.get("size", 0)), size/2, rel_tol=vae_pct)
                        or int(extract.get("size", 0) == size)):
                            for tensor_params, model in attributes.items():
                                if isclose(int(extract.get("tensor_params", 0)), tensor_params, rel_tol=vae_pct): 
                                    print(f"VAE-{model} found") #found hook
            # check lora
        elif int(extract.get("diffusers",0)) > 256:
            for size, attributes in lora_peek.items():
                if (isclose(int(extract.get("size",0)),size, rel_tol=0.05)
                or isclose(int(extract.get("size", 0)), size*2, rel_tol=lora_pct) 
                or isclose(int(extract.get("size", 0)), size/2, rel_tol=lora_pct)): 
                    for tensor_params, desc in attributes.items():
                        if isclose(int(extract.get("tensor_params", 0)),tensor_params, rel_tol=lora_pct):
                                rep_count = 0
                                for each in next(iter([desc, 0])):
                                    if rep_count <= 1:
                                        if each.lower() not in str(extract.get('filename', 0)).lower():
                                            rep_count += 1
                                        else:          
                                            print(f"{each.lower()}-{desc[-1]}") #found hook
            # check clip
        elif int(extract.get("transformers", 0)) >= 2:
            clip_inside = True
            for tensor_params, attributes in tf_peek.items():
                if isclose(int(extract.get("tensor_params", 0)), tensor_params, rel_tol=.07):
                    for size, model in attributes.items():
                        if (isclose(int(extract.get("size", 0)), size, rel_tol=tf_pct) 
                        or isclose(int(extract.get("size", 0)), size*2, rel_tol=tf_pct)
                        or isclose(int(extract.get("size", 0)), size/2, rel_tol=tf_pct)):
                            if isclose(int(extract.get("transformers", 0)), model[0], rel_tol=tf_pct):
                                print(f"CLI-{model[1]}") #found hook
            # check model
        if int(extract.get("size",0)) > 120000000:
            for tensor_params, attributes, in model_peek.items():
                if isclose(int(extract.get("tensor_params", 0)), tensor_params, rel_tol=model_pct):
                    for size, model in attributes.items():
                        if (isclose(int(extract.get("size", 0)), size, rel_tol=model_pct)
                        or isclose(int(extract.get("size", 0)), size*2, rel_tol=model_pct)
                        or isclose(int(extract.get("size", 0)), size/2, rel_tol=model_pct)):
                                print(f"{model}, vae: {vae_inside}, text_encoder: {clip_inside}") #found hook

class ReadMeta:
    # level of certainty, counts tensor block type matches
    full_data, meta, count_dict = {}, {}, {}
    occurrence_counts = defaultdict(int)

    known = {
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

    def __init__(
        self, filename, full_path
    ):
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

        self.full_path = full_path  # the path of the file
        self.filename = filename  # the title of the file only
        if not os.path.exists(self.full_path):  # be sure it exists, then proceed
            raise RuntimeError(f"Not found: {self.filename}")
        else:
            self.model_tag["filename"] = self.filename
            self.model_tag["size"] = os.path.getsize(self.full_path)

    def _parse_safetensors_metadata(self, full_path):
        with open(full_path, "rb") as json_file:
            header = struct.unpack("<Q", json_file.read(8))[0]
            try:
                return json.loads(json_file.read(header), object_hook=self._search_dict)
            except:
                return print(f"error loading {full_path}")

    @classmethod
    def data(self, filename, full_path):
        filepath = Path(filename)
        ext = filepath.suffix

        if ext in {".pt", ".pth", ".ckpt"}:  # process elsewhere
            return
        elif ext in {".safetensors" or ".sft"}:
            self.occurrence_counts.clear()
            self.full_data.clear()
            self.meta = self._parse_safetensors_metadata(full_path)
            self.full_data.update((k, v)
                                  for k, v in self.model_tag.items() if v != "")
            self.full_data.update((k, v)
                                  for k, v in self.count_dict.items() if v != 0)
            self.count_dict.clear()
            self.model_tag.clear()
        elif ext is ".gguf":
            # placeholder - parse gguf metadata(path) using llama lib
            meta = ""
        elif ext is ".bin":
            meta = ""  # placeholder - parse bin metadata(path) using ...???
        else:
            raise RuntimeError(f"Unrecognized file format: {filename}")

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

def folder_run(config, search)
    path_name = os.path.normpath(os.path.join(config,search))  # multi read
    for filename in os.listdir(path_name):
        full_path = os.path.normpath(os.path.join(path_name,filename))  # multi read
        if not os.path.isdir(filename):
            metareader = ReadMeta.data(path_name, full_path)
            return metareader
        else:
            print("no fies??")
def single_run(config, path, search,)
    full_path = os.path.join(config, path, search)
    if not os.path.isdir(full_path):
        metareader = ReadMeta.data(search, full_path)
        return metareader
    else:
        print("no fies??")
            
config_path = config
search_path = "models"
search_name = 
metareader = folder_run(config_path, search_path)
#metareader = single_run(config_path, search_path, search_name)
#for k, v in metareader.items():            #uncomment to view model properties
#    print(k,v)
evalmeta = EvalMeta.data(metareader)

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

def prioritize