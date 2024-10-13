import os
from sdbx.config import config
from sdbx.nodes.helpers import soft_random
from sdbx.indexer import IndexManager

print("\nInitializing model index, checking system specs.\n  Please wait...")
create_index = IndexManager().write_index()      # (defaults to config/index.json)
dif_index = config.get_default("index", "DIF")
print(f"Ready.")
name_path = input("""
Please type the filename of an available checkpoint.
Path will be detected.
(default:diffusion_pytorch_model.fp16.safetensors):""" or "diffusion_pytorch_model.fp16.safetensorss")

name_path = os.path.basename(name_path)
spec = config.get_default("index","DIF")
for key,val in spec.items():
    if name_path in key:
        model = key
        pass
prompt = "A slice of a rich and delicious chocolate cake presented on a table in a palace reminiscent of Versailles"
seed = int(soft_random())

optimize = config.node_tuner
insta = config.t2i_pipe
optimize.determine_tuning(model)
opt_exp = optimize.opt_exp() 
#below is fed from genesis

#prompt node
insta.queue_manager(prompt,seed)

#loader node transformer class
gen_exp = optimize.gen_exp(2)#clip skip
insta.cache_jettison()
insta.declare_encoders(gen_exp)

#enocder node
cond_exp = optimize.cond_exp()
insta.encode_prompt(cond_exp)
#insta.metrics()
#cache ctrl node
insta.cache_jettison(encoder=True)

#t2i
pipe_exp = optimize.pipe_exp()
vae_exp = optimize.vae_exp()
insta.construct_pipe(pipe_exp, vae_exp)
insta.diffuse_latent(gen_exp)
#insta.metrics()
#cache ctrl nocude
insta.cache_jettison(lora=True)

#vae nodesorta but

image = insta.decode_latent(vae_exp)

#cache ctrl node
insta.cache_jettison(vae=True)
#metrics node
#insta.metrics()