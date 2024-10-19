import os
from collections import defaultdict
from sdbx.config import config
from sdbx.nodes.helpers import soft_random
from sdbx.indexer import IndexManager
from sdbx.nodes.base import nodes


#print("\nInitializing model index, checking system specs.\n  Please wait...")
#create_index = IndexManager().write_index()      # (defaults to config/index.json)
#dif_index = config.get_default("index", "DIF")
#print(f"Ready.")
#name_path = input("""
#Please type the filename of an available checkpoint.
#Path will be detected.
#(default:diffusion_pytorch_model.fp16.safetensors):""" or "diffusion_pytorch_model.fp16.safetensors")

name_path = "sdxlbase.diffusion_pytorch_model.fp16.safetensors"
optimize         = config.node_tuner

name_path = os.path.basename(name_path)
spec = config.get_default("index","DIF")
name_path = name_path.strip()
name_path = os.path.basename(name_path)
if ".safetensors" not in name_path:
     name_path = name_path + ".safetensors"
for key,val in spec.items():
    if name_path in key:
        model = key
        pass

defaults = optimize.determine_tuning(model)

device = nodes.force_device(**defaults["force_device"])
if defaults["empty_cache"]["stage"].get("head", None) != None:
        nodes.empty_cache(**defaults["empty_cache"]["stage"]["head"])
queue = nodes.text_input(**defaults.get("text_input"))
transformer_models = nodes.load_transformer(**defaults["load_transformer"])

vae = nodes.load_vae_model(**defaults["load_vae_model"])
queue = nodes.encode_prompt(queue=queue, transformer_models=transformer_models, **defaults["encode_prompt"])
if defaults["empty_cache"]["stage"].get("encoder", None) != None:
    nodes.empty_cache(queue, defaults["empty_cache"]["stage"]["encoder"])
pipe = nodes.diffusion_pipe(vae, **defaults["diffusion_pipe"])
lora = nodes.load_lora(**defaults["load_lora"])
scheduler = nodes.noise_scheduler(**defaults["noise_scheduler"])
pipe = nodes.generate_image(pipe, queue, scheduler, **defaults["generate_image"])
if defaults["empty_cache"]["stage"].get("generate", None) != None:
    nodes.empty_cache(pipe, defaults["empty_cache"]["stage"]["generate"])
image = nodes.autodecode(pipe, **defaults["autodecode"])
if defaults["empty_cache"]["stage"].get("tail", None) != None:
        nodes.empty_cache(image, **defaults["empty_cache"]["stage"]["tail"])