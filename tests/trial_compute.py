import os
from sdbx.config import config

index            = config.model_indexer
optimize         = config.node_tuner
sys_cap          = config.sys_cap

print("\nAnalyzing model & system capacity\n  Please wait...")
create_index = index.write_index()     # (defaults to config/index.json)
spec = sys_cap.write_capacity()
print(f"Ready.")
name_path = input("""
Please type the filename of an available checkpoint.
Path will be detected.
(default:diffusion_pytorch_model.fp16.safetensors):""" or "diffusion_pytorch_model.fp16.safetensors")
# name_path = "sdxlbase.diffusion_pytorch_model.fp16.safetensors"


name_path = os.path.basename(name_path)
diffusion_index = config.get_default("index","DIF")
name_path = name_path.strip()
name_path = os.path.basename(name_path)
if ".safetensors" not in name_path:
     name_path = name_path + ".safetensors"
for key,val in diffusion_index.items():
    if name_path in key:
        model = key
        pass

defaults = optimize.determine_tuning(model)

from sdbx.nodes.base import nodes
#pipe = nodes.empty_cache(transformer_models, lora_pipe, unet_pipe, vae_pipe)

device = nodes.force_device(**defaults.get("force_device"))
queue = nodes.diffusion_prompt(**defaults.get("diffusion_prompt"))
tokenizers, text_encoders = nodes.load_transformer(**defaults.get("load_transformer"))
queue = nodes.encode_prompt(**defaults.get("encode_prompt"), queue=queue, tokenizers_in=tokenizers, text_encoders_in=text_encoders,)
vae = nodes.load_vae_model(**defaults.get("load_vae_model"))
pipe = nodes.diffusion_pipe(**defaults.get("diffusion_pipe"), vae=vae)
pipe = nodes.load_lora(**defaults.get("load_lora"), pipe=pipe)
pipe = nodes.load_scheduler(**defaults.get("noise_scheduler"), pipe=pipe)
pipe, latent = nodes.generate_image(**defaults.get("generate_image"), pipe=pipe, queue=queue)
image = nodes.autodecode(**defaults.get("autodecode"), pipe=pipe, latent=latent)
