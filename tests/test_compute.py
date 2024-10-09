import os
import asyncio
from sdbx.config import config
from sdbx.nodes.helpers import soft_random

async def outer_coroutine():
    print("Initializing model index. \nPlease wait...\n")
    await write_index_now()
    return print("""Ready.
                 """)

async def write_index_now():
    create_index = config.model_indexer.write_index()       # (defaults to config/index.json)

asyncio.run(outer_coroutine())

name_path = input("Model Filename (default:ponyFaetality_v11.safetensors): ") or "ponyFaetality_v11.safetensors"
name_path = os.path.basename(name_path)

prompt = "A slice of a rich and delicious chocolate cake presented on a table in a palace reminiscent of Versailles"
seed = int(soft_random())

optimize = config.node_tuner
insta = config.t2i_pipe
optimize.determine_tuning(name_path)
opt_exp = optimize.opt_exp() 
#below is fed from genesis

#prompt node
insta.queue_manager(prompt,seed)

#loader node transformer class
gen_exp = optimize.gen_exp(2)#clip skip
insta.set_device()
insta.cache_jettison()
insta.declare_encoders(gen_exp)

#enocder node
cond_exp = optimize.cond_exp()
#insta.debugger(locals())
insta.encode_prompt(cond_exp)
#insta.metrics()
#cache ctrl node
insta.cache_jettison(encoder=True)

#t2i
pipe_exp = optimize.pipe_exp()
insta.construct_pipe(pipe_exp)
#insta.debugger(locals())
insta.diffuse_latent(gen_exp)
#insta.metrics()
#cache ctrl nocude
insta.cache_jettison(lora=True)

#vae node
vae_exp = optimize.vae_exp()
image = insta.decode_latent(vae_exp)

#cache ctrl node
insta.cache_jettison(vae=True)
#metrics node
#insta.metrics()