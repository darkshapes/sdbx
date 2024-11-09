import os
from sdbx.config import config
from sdbx.nodes.base import nodes

#print("\nInitializing model index, checking system specs.\n  Please wait...")
#create_index = IndexManager().write_index()      # (defaults to config/index.json)
#dif_index = config.get_default("index", "DIF")
#print(f"Ready.")
#name_path = input("""
#Please type the filename of an available checkpoint.
#Path will be detected.
#(default:vividpdxl_realVAE.safetensors):""" or "vividpdxl_realVAE.safetensors")
# "virtualDiffusionPony_25B3C4N3.safetensors"
name_path = "hellaineMixPDXL_v45.safetensors"
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
defaults["generate_image"]["width"] = 832
defaults["generate_image"]["height"] = 1152
defaults["diffusion_prompt"]["batch"] = 1
defaults["diffusion_prompt"]["prompt"] = "score_9, score_8_up, 2.5d, a confident adult woman making eye contact with a come-hither look, pov, straddling, blush, soft features, juicy pussy, deep penetration, thighhighs, long hair, garter belt, arched back, partially nude"
from sdbx.nodes.base import nodes

#pipe = nodes.empty_cache(transformer_models, lora_pipe, unet_pipe, vae_pipe)

device = nodes.force_device(**defaults.get("force_device"))
if defaults["empty_cache"]["stage"].get("head", None) != None:
        nodes.empty_cache(**defaults["empty_cache"]["stage"].get("head"))
queue = nodes.text_input(**defaults.get("text_input"))
transformer_models = nodes.load_transformer(**defaults.get("load_transformer"))

vae = nodes.load_vae_model(**defaults.get("load_vae_model"))
queue = nodes.encode_prompt(queue=queue, transformer_models=transformer_models, **defaults.get("encode_prompt"))
if defaults["empty_cache"]["stage"].get("encoder", None) != None:
    nodes.empty_cache(queue, defaults["empty_cache"]["stage"].get("encoder"))
pipe = nodes.diffusion_pipe(vae, **defaults.get("diffusion_pipe"))
lora = nodes.load_lora(**defaults.get("load_lora"))
scheduler = nodes.noise_scheduler(**defaults.get("noise_scheduler"))
pipe = nodes.generate_image(pipe, queue, scheduler, **defaults.get("generate_image"))
if defaults["empty_cache"]["stage"].get("generate", None) != None:
    nodes.empty_cache(pipe, defaults["empty_cache"]["stage"].get("generate"))
image = nodes.autodecode(pipe, **defaults.get("autodecode"))

#if defaults["empty_cache"]["stage"].get("tail", None) != None:
#       nodes.empty_cache(image, **defaults["empty_cache"]["stage"]["tail"])