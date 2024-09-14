from time import process_time_ns
import psutil
import os
import json
import struct
from pathlib import Path
from collections import defaultdict
import platform
from math import isclose

print(f'begin: {process_time_ns()/1e6} ms')    
config = {
        'windows': os.path.join(os.environ.get('LOCALAPPDATA', os.path.join(os.path.expanduser('~'), 'AppData', 'Local')), 'Shadowbox'),
        'darwin': os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'Shadowbox'),
        'linux': os.path.join(os.path.expanduser('~'), '.config', 'shadowbox'),
    }[platform.system().lower()]

class ReadMeta:
    full_data, meta, count_dict = {}, {}, {} # level of certainty, counts tensor block type matches
    occurrence_counts = defaultdict(int)

    known = {
            "adaLN_modulation":"mmdit",
            "mlp.fc":"mmdit",
            "mlpX":"mmdit",
            "w1q":"mmdit",
            "self_attn.out_proj":"mmdit",
            "w1q":"mmdit",
            "w1o":"mmdit",
            "mlpX.c_proj":"mmdit",
            "mlpC.c_proj":"mmdit",
            "w2q.":"mmdit",
            "w2k.":"mmdit",
            "w2v.":"mmdit",
            "w2o.":"mmdit",
            "w1k.":"mmdit",
            "w1v.":"mmdit",
            "mlpX.c_fc":"mmdit",
            "mlpX.c_proj.":"mmdit",
            "mlpC.c_fc":"mmdit",
            "modC.":"mmdit",
            "modX.":"mmdit",
            "model.register_tokens":"auraflow",
            "model.positional_encoding":"auraflow",
            "model.init_x_linear":"auraflow",
            "model.t_embedder.mlp.":"auraflow",
            "model.t_embedder.mlp.":"auraflow",
            "modCX":"flux",
            "img_attn.proj":"flux",
            "time_in.in_layer":"flux",
            "time_in.out_layer":"flux",
            "vector_in.in_layer":"flux",
            "vector_in.out_layer":"flux",
            "guidance_in.in_layer":"flux",
            "guidance_in.in_layer":"flux",
            "txt_in":"flux",
            "img_in":"flux",
            "img_mod.lin":"flux",
            "txt_mod.lin":"flux",
            "img_attn.qkv":"flux",
            "txt_attn.qkv":"flux",
            "t_embedder.mlp":"pixart_s",
            "y_embedder.embedding_table":"pixart_s",
            "wkqv.":"hunyuan",
            "wqkv.":"hunyuan",
            "q_norm":"hunyuan",
            "k_norm":"hunyuan",
            "out_proj":"hunyuan",
            "kq_proj":"hunyuan",
            "default_modulation.":"hunyuan",
            "pooler":"hunyuan",
            "t_embedder":"hunyuan",
            "x_embedder":"hunyuan",
            "mlp_t5":"hunyuan",
            "time_extra_emb.extra_embedder":"hunyuan",
            "to_q":"diffusers",
            "to_k":"diffusers",
            "to_v":"diffusers",
            "norm_q":"diffusers",
            "norm_k":"diffusers",
            "to_out":"diffusers",
            "norm1.norm":"diffusers",
            "norm1.linear":"diffusers",
            "ff.net.0":"diffusers",
            "ff.net.2":"diffusers",
            "time_extra_emb":"diffusers",
            "time_embedding":"diffusers",
            "pos_embd":"diffusers",
            "text_embedder":"diffusers",
            "extra_embedder":"diffusers",
            "attn.norm_added_q":"diffusers",
            "skip.connection":"sdxl",
            "in.layers.2":"sdxl",
            "out.layers.3":"sdxl",
            "upsamplers":"sdxl",
            "downsamplers":"sdxl",
            "op":"sd",
            "in_layers":"sd",
            "out_layers":"sd",
            "emb_layers":"sd",
            "skip_connection":"sd",
            "text_model":"diffusers_lora",
            "self_attn":"diffusers_lora",
            "to_q_lora":"diffusers_lora",
            "to_k_lora":"diffusers_lora",
            "to_v_lora":"diffusers_lora",
            "to_out_lora":"diffusers_lora",
            "text_projection":"diffusers_lora",
            "to.q.lora":"unet_lora",
            "to.k.lora":"unet_lora",
            "to.v.lora":"unet_lora",
            "to.out.0.lora":"unet_lora",
            "proj.in":"unet_lora",
            "proj.out":"unet_lora",
            "emb.layers":"unet_lora",
            "proj_in.":"transformers",
            "proj_out.":"transformers",
            "norm.":"transformers",
            "norm1.":"unet",
            "norm2.":"unet",
            "norm3.":"unet",
            "attn1.to_q.":"unet",
            "attn1.to_k.":"unet",
            "attn1.to_v.":"unet",
            "attn1.to_out.0.":"unet",
            "attn2.to_q.":"unet",
            "attn2.to_k.":"unet",
            "attn2.to_v.":"unet",
            "attn2.to_out.0.":"unet",
            "ff.net.0.proj.":"unet",
            "ff.net.2.":"unet",
        }
    
    @classmethod
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


        self.full_path = full_path #the path of the file
        self.filename = filename #the title of the file only
        if not os.path.exists(self.full_path): #be sure it exists, then proceed
            raise RuntimeError(f"Not found: {self.filename}")
        else:
            self.model_tag["filename"] = self.filename
            self.model_tag["size"] = os.path.getsize(self.full_path)

    @classmethod
    def _parse_safetensors_metadata(self, full_path):
        with open(full_path, "rb") as json_file:
            header = struct.unpack("<Q", json_file.read(8))[0]
            try:
                return json.loads(json_file.read(header), object_hook=self._search_dict)
            except:
                return print(f"error loading {full_path}")

    @classmethod
    def data(self, filename, full_path):
        if Path(filename).suffix in {".pt", ".pth", ".ckpt"}: #process elsewhere
            return
        elif Path(filename).suffix in {".safetensors" or ".sft"}:
            self.occurrence_counts.clear()
            self.full_data.clear()
            self.meta = self._parse_safetensors_metadata(full_path)  
            self.full_data.update((k,v) for k,v in self.model_tag.items() if v != "")
            self.full_data.update((k,v) for k,v in self.count_dict.items() if v != 0)
            self.count_dict.clear()
            self.model_tag.clear()
            for  k, v in self.full_data.items():            #diagnostics
                print(k,v)

        elif Path(filename).suffix in {".gguf"}:
            meta = ""  # placeholder - parse gguf metadata(path) using llama lib
        elif Path(filename).suffix in {".bin"}:
            meta = ""  # placeholder - parse bin metadata(path) using ...???
        else:
            try:
                print(RuntimeError(f"Unrecognized file format: {filename}"))
            except:
                print("Path or File could not be read.")
                pass
        return self.full_data

    @classmethod
    def _search_dict(self, meta):
        self.meta = meta
        for num in list(self.meta):
            if self.model_tag.get(num, "not_found") != "not_found": #handle inevitable exceptions invisibly
                self.model_tag[num] = meta.get(num) #drop it like its hot 
            if "dtype" in num:
                 self.model_tag["tensor_params"] += 1 # count tensors
            elif "shape" in num:
                if meta.get(num) > self.model_tag["shape"]: #measure first shape size thats returned
                     self.model_tag["shape"] = self.meta.get(num)  # (room for improvement here, would prefer to be largest shape, tbh)
            if "data_offsets" in num:
                pass
            else:
                if ("shapes" or "dtype") not in num:
                    for block, model_type in self.known.items():  #model type, dict data
                        if block in num:     #if value matches one of our key values
                            self.occurrence_counts[model_type] += 1 #count matches
                            self.count_dict[model_type] = self.occurrence_counts.get(model_type, 0) #pair match count with type of model  
        return meta   

path_name = os.path.normpath(os.path.join(config, "models","lora","tcd")) #multi read
for each in os.listdir(path_name):
    filename = each # "PixArt-Sigma-XL-2-2K-MS.safetensors"
    full_path = os.path.join(path_name, filename)
    if not os.path.isdir(full_path):
        metareader = ReadMeta(each, full_path).data(each, full_path)
        max_mem = .75 #75% memory threshold
        safe_mem = .50 #50% memory threshold

        #print(metareader)    
        # print(metareader.get("size", 0), psutil.virtual_memory().total)
        try:
            memfit = metareader.get("size", 0)/dram
        except:
            print("Unknown file, skipping...")
        else:
            print(f'read: {process_time_ns()/1e6} ms')
            dram = psutil.virtual_memory().total
            print(f"max  memory allocation: {memfit*dram}")
            if memfit > max_mem:
                mem_75 = True
                print(f"model exceed 75%: {True}")
                #enable_sequential_cpu_offload()
            if  safe_mem > memfit:
                mem_50 = True
                print(f"model below 50%: {True}")

            pct = 0.05 # % of relative closeness to an actual value
            certainty = 0 #increase with each positive read
            max_cert = 8

            if metareader.get("sd", 0) >= 10:
                certainty +=1
                if metareader.get("shape", 0) == [1280, 1280]:
                    certainty +=1
                    if isclose(metareader.get("diffusers_lora", 0), 294, rel_tol=pct):
                            certainty +=1
                    if isclose(metareader.get("diffusers", 0), 980, rel_tol=pct):
                        certainty +=1
                        if isclose(metareader.get("tensor_params", 0), 2515, rel_tol=pct):
                            certainty +=1
                        if isclose(metareader.get("transformers", 0), 76, rel_tol=pct):
                            certainty +=1
                            print('includes clip')
                        if isclose(metareader.get("unet", 0), 1544, rel_tol=pct):
                            certainty +=1
                            print('includes vae')
                            if metareader.get("dtype") == "F16":
                                if isclose(metareader.get("size", 0),6938040706, rel_tol=pct):
                                    certainty +=1
                                print(f"model type sdxl, certainty {certainty}/{max_cert}")
                                    # precision = Precision.F16
                                    # scheduler_default = ays_xl
                                    # steps = 10
                    elif isclose(metareader.get("diffusers",0), 224, rel_tol=pct):
                        certainty +=1
                        if isclose(metareader.get("tensor_params", 0), 1131, rel_tol=pct):
                            certainty +=1
                        if isclose(metareader.get("transformers", 0), 106, rel_tol=pct):
                            certainty +=1
                            print('includes clip')
                        if isclose(metareader.get("unet", 0), 464, rel_tol=pct):
                            certainty +=1
                            print('includes vae')
                        if metareader.get("dtype") == "F16":
                            if isclose(metareader.get("size", 0), 3188362889, rel_tol=.5):
                                    certainty +=1
                            print(f"model type sd1.5, certainty {certainty}/{max_cert}")
                    # precision = Precision.F16
                    # scheduler_default = ays_sd
                        # steps = 10

                    # print("type:sd1.5")
                        # precision = Precision.F16
                        # scheduler_default = ays_xl

#elif metareader.get("sd", 0) >= 10:

# if metareader.get("size", 0)/psutil.virtual_memory().total > max_mem:
#     printf(f"mem exceed: {True}")
# else: print(f"mem exceed: {False}")
print(f'end: {process_time_ns()/1e6} ms')
