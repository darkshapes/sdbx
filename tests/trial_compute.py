import os
from sdbx.config import config
from sdbx.nodes.base import nodes
from sdbx.config import config

# print("\nInitializing model index, checking system specs.\n  Please wait...")
# create_index = config.model_indexer.write_index()      # (defaults to config/index.json)
# dif_index = config.get_default("index", "DIF")
# print(f"Ready.")
name_path = "sdxlbase.diffusion_pytorch_model.fp16.safetensors" #input("""
# Please type the filename of an available checkpoint.
# Path will be detected.
# (default:diffusion_pytorch_model.fp16.safetensors):""" or "diffusion_pytorch_model.fp16.safetensors")

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

# device = nodes.force_device(**defaults.get("force_device"))
#pipe = nodes.empty_cache(transformer_models, lora_pipe, unet_pipe, vae_pipe)
defaults = optimize.determine_tuning(model)
queue = nodes.diffusion_prompt(**defaults.get("diffusion_prompt"))
tokenizers, text_encoders = nodes.load_transformer(**defaults.get("load_transformer"))
queue = nodes.encode_prompt(queue, tokenizers, text_encoders,**defaults.get("encode_prompt"))
vae = nodes.load_vae_model(**defaults.get("load_vae_model"))
pipe = nodes.diffusion_pipe(vae,**defaults.get("diffusion_pipe"))
pipe = nodes.load_lora(pipe, **defaults.get("load_lora"))
pipe = nodes.load_scheduler(pipe, **defaults.get("noise_scheduler"))
pipe, latent = nodes.generate_image(pipe, queue, **defaults.get("generate_image"))
image = nodes.autodecode(pipe, latent, **defaults.get("autodecode"))
