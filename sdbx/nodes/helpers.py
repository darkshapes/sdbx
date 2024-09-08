from sdbx.config import config

from functools import cache as function_cache, wraps
from numpy.random import SeedSequence, Generator, Philox
import secrets as secrets
import os
import gc
import re

import torch, torch.cuda, torch.cpu, torch.backends.cudnn, torch.backends.mps
from natsort import natsorted

def generator_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # If the cache exists and matches the arguments, return an iterator over the cached results
        if wrapper.cache and wrapper.cache_args == (args, kwargs):
            return iter(wrapper.cache)

        # If no cache exists, or arguments differ, create a new generator
        wrapper.cache = []
        wrapper.cache_args = (args, kwargs)
        
        def generator_with_cache():
            for result in func(*args, **kwargs):
                wrapper.cache.append(result)  # Cache the result as it's generated
                yield result  # Yield the result to the caller

        return generator_with_cache()

    # Initialize cache and arguments
    wrapper.cache = None
    wrapper.cache_args = None
    return wrapper

cache = lambda node: generator_cache(node) if node.generator else function_cache(node)

def rename_class(base, name):
    # Create a new class dynamically, inheriting from base_class
    new = type(name, (base,), {})
    
    # Set the __name__ and __qualname__ attributes to reflect the new name
    new.__name__ = name
    new.__qualname__ = name
    
    return new

def format_name(name):
    return ' '.join(word[0].upper() + word[1:] if word else '' for word in re.split(r'_', name))

#cpu random routines
def soft_random(size=0x2540BE3FF): # returns a deterministic random number using Philox
    entropy = f"0x{secrets.randbits(128):x}" # git gud entropy
    rndmc = Generator(Philox(SeedSequence(int(entropy,16))))
    return rndmc.integers(0, size) 

def hard_random(hardness=5): # returns a non-prng random number use secrets
    return int(secrets.token_hex(hardness),16) # make hex secret be int

def tensor_random(seed=None):
    return torch.random.seed() if seed is None else torch.random.manual_seed(seed)

def tensorify(hard, size=4): # creates an array of default size 4x1 using either softRandom or hardRandom
    num = []
    for s in range(size): # make array, convert float, negate it randomly
        if hard==False: # divide 10^10, truncate float
            conv = '{0:.6f}'.format((float(soft_random()))/0x2540BE400)
        else:  # divide 10^12, truncate float
            conv = '{0:.6f}'.format((float(hard_random()))/0xE8D4A51000)
        num.append(float(conv)) if secrets.choice([True, False]) else num.append(float(conv)*-1)
    return num

def seed_planter(seed, deterministic=True):
    torch.manual_seed(seed)
    if torch.cuda.is_available()==True:
        if deterministic == True:
            return {'torch.backends.cudnn.deterministic': 'True','torch.backends.cudnn.benchmark': 'False'}
        return torch.cuda.manual_seed(seed), torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available()==True:
        return torch.mps.manual_seed(seed)
    # elif torch.xpu.is_available():
    #    return torch.xpu.manual_seed(seed)

def get_gpus():
    gpus = ["cpu","cuda","mps","xpu"]
    gpus = natsorted([g for g in gpus])
    return gpus

def cache_bin():
    gc.collect()
    if torch.cuda.is_available(): return torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): return torch.mps.empty_cache()
    elif torch.xpu.is_available(): return torch.xpu.empty_cache()

def get_dir_files(folder, filtering=""):
    try: files = natsorted([f for f in os.listdir(config.get_path(folder)) if os.path.isfile(os.path.join(config.get_path(folder), f)) & (filtering.lower() in f.lower())])
    except: files = "No Files Found!"
    return files

def get_dir_files_count(folder, filtering=""):
    counts = [get_dir_files(folder.lower(), filtering.lower())]
    return format(len(counts))

model_list = {  #diffusion models
    "stabilityai/stable-diffusion-xl-base-1.0", 
}
pcm_list = { #normalcfg or smallcfg for 2step 4step 8step 16step safetensors)
    os.path.join(config.get_path("models.loras"),"pcm_sdxl_normalcfg_16step_converted_fp16.safetensors"),
}
ti_list = {}

def get_schedulers(filtering=""):

    schedulerdict = [
        "EulerDiscreteScheduler",
        "EulerAncestralDiscreteScheduler",
        "HeunDiscreteScheduler",
        "UniPCMultistepScheduler",
        "DDIMScheduler", # rescale_betas_zero_snr=True, timestep_spacing="trailing"
        "DPMSolverMultistepScheduler", # dpmpp2m, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", 
        "LMSDiscreteScheduler",  #use_karras_sigmas=True
        "DEISMultistepScheduler",
        "AysSchedules"
    ]

    schedulers = natsorted([s for s in schedulerdict if (filtering in s)])
    return schedulers

def get_solvers(filtering=""):

    solverdict = [
        "dpmsolver++",
        "sde-dpmsolver++",
        "sde-dpmsolver",
        "dpmsolver"
    ]
    solvers = natsorted([s for s in solverdict if (filtering in s)])
    return solvers
