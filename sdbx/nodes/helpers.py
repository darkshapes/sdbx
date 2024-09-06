from sdbx.config import config

from functools import cache as function_cache, wraps
from numpy.random import SeedSequence, Generator, Philox
import secrets as secrets
import os
import gc
import re

from torch import torch
from natsort import natsorted

def generator_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if the cached data exists and inputs are identical
        if wrapper.cache and wrapper.cache_args == (args, kwargs):
            return iter(wrapper.cache)
        
        # Otherwise, run the generator and cache the result
        gen = func(*args, **kwargs)
        wrapper.cache = list(gen)
        wrapper.cache_args = (args, kwargs)
        return iter(wrapper.cache)

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
def softRandom(size=0x2540BE3FF): # returns a deterministic random number using Philox
    entropy = f"0x{secrets.randbits(128):x}" # git gud entropy
    rndmc = Generator(Philox(SeedSequence(int(entropy,16))))
    return rndmc.integers(0, size) 

def hardRandom(hardness=5): # returns a non-prng random number use secrets
    return int(secrets.token_hex(hardness),16) # make hex secret be int

def tensorRandom(device,seed=None):
    return torch.random.seed() if seed is None else torch.random.manual_seed(seed)

def tensorify(hard, size=4): # creates an array of default size 4x1 using either softRandom or hardRandom
    num = []
    for s in range(size): # make array, convert float, negate it randomly
        if hard==False: # divide 10^10, truncate float
            conv = '{0:.6f}'.format((float(softRandom()))/0x2540BE400)
        else:  # divide 10^12, truncate float
            conv = '{0:.6f}'.format((float(hardRandom()))/0xE8D4A51000)
        num.append(float(conv)) if secrets.choice([True, False]) else num.append(float(conv)*-1)
    return num

def seedPlanter(seed, deterministic=True):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        if deterministic == True:
            return {'torch.backends.cudnn.deterministic': 'True','torch.backends.cudnn.benchmark': 'False'}
        return torch.cuda.manual_seed(seed), torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        return torch.mps.manual_seed(seed)
    elif torch.xpu.is_available():
        return torch.xpu.manual_seed(seed)

def getGPUs(filtering=""):
    if torch.cuda.is_available(): 
        for i in range(torch.cuda.device_count()):
            devices = [torch.cuda.get_device_properties(i).name]
    elif torch.backends.mps.is_available():
        for i in range(torch.mps.device_count()):
            devices = [torch.device(f"mps:{i}")]
    elif torch.xpu.is_available():
            devices = [torch.xpu.get_device_properties(i).name]
    else: devices = "cpu"
    return natsorted(filtering in [devices] if filtering in [devices] is not None else [devices])

def cacheBin():
    gc.collect()
    if torch.cuda.is_available(): return torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): return torch.mps.empty_cache()
    elif torch.xpu.is_available(): return torch.xpu.empty_cache()

def getDirFiles(folder, filtering=""):
   return natsorted([f for f in os.listdir(config.get_path(folder)) if os.path.isfile(os.path.join(config.get_path(folder), f)) & (filtering in f)])

def getDirFilesCount(folder, filtering=""):
    counts = [getDirFiles(folder, filtering)]
    return format(len(counts))

def switch_case(switch):
    def switcher(func):
        def case(case):
            return switch[case](case) if case in switch else None
        return case 
    return switcher

def getSchedulers(filtering=""):

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

def getSolvers(filtering=""):

    solverdict = [
        "dpmsolver++",
        "sde-dpmsolver++",
        "sde-dpmsolver",
        "dpmsolver"
    ]
    solvers = natsorted([s for s in solverdict if (filtering in s)])
    return solvers
