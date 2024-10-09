import os
import asyncio
from sdbx.nodes.compute import T2IPipe
from sdbx.nodes.tuner import NodeTuner
from sdbx.indexer import IndexManager
from sdbx.nodes.helpers import soft_random

async def outer_coroutine():
    print("Model index will now be written. \nPlease wait...\n")
    await write_index_now()
    return print("""Ready.
                 """)

async def write_index_now():
    create_index = IndexManager().write_index()       # (defaults to config/index.json)

asyncio.run(outer_coroutine())

name_path = input("Enter an existing file to tune for: ")
name_path = os.path.basename(name_path)

prompt = "A slice of a rich and delicious chocolate cake presented on a table in a palace reminiscent of Versailles"
seed = int(soft_random())

optimize = NodeTuner()
insta = T2IPipe()
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
insta.debugger(locals())
insta.encode_prompt(cond_exp)
insta.metrics()
#cache ctrl node
insta.cache_jettison(encoder=True)

#t2i
pipe_exp = optimize.pipe_exp()
insta.construct_pipe(pipe_exp)
insta.debugger(locals())
insta.diffuse_latent(gen_exp)
insta.metrics()
#cache ctrl node
insta.cache_jettison(lora=True)

#vae node
vae_exp = optimize.vae_exp()
image = insta.decode_latent(vae_exp)

#cache ctrl node
insta.cache_jettison(vae=True)
#metrics node
insta.metrics()