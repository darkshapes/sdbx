from random import random
from natsort import natsorted
import numpy as np
from numpy.random import SeedSequence, Generator, Philox, BitGenerator
import secrets as secrets
import os
from sdbx import config

def softRandom(size=0x2540BE3FF): # returns a deterministic random number using Philox
    entropy = f"0x{secrets.randbits(128):x}" # git gud entropy
    rndmc = Generator(Philox(SeedSequence(int(entropy,16))))
    return rndmc.integers(0, size) 

def hardRandom(hardness=5): # returns a non-prng random number use secrets
    return int(secrets.token_hex(hardness),16) # make hex secret be int

def tensorify(hard, size=4): # creates an array of default size 4x1 using either softRandom or hardRandom
    num = []
    for s in range(size): # make array, convert float, negate it randomly
        if hard==False: # divide 10^10, truncate float
            conv = '{0:.6f}'.format((float(softRandom()))/0x2540BE400)
        else:  # divide 10^12, truncate float
            conv = '{0:.6f}'.format((float(hardRandom()))/0xE8D4A51000)
        num.append(float(conv)) if secrets.choice([True, False]) else num.append(float(conv)*-1)
    return num

def getDirFiles(folder, filtering=""):
   return natsorted([f for f in os.listdir(config.get_path(folder)) if os.path.isfile(os.path.join(config.get_path(folder), f)) & (filtering in f)])

def getDirFilesCount(folder, filtering=""):
    counts = [getDirFiles(folder, filtering)]
    return format(len(counts))

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

import re

def rename_class(base, name):
    # Create a new class dynamically, inheriting from base_class
    new = type(name, (base,), {})
    
    # Set the __name__ and __qualname__ attributes to reflect the new name
    new.__name__ = name
    new.__qualname__ = name
    
    return new

def format_name(name):
    return ' '.join(word[0].upper() + word[1:] if word else '' for word in re.split(r'_', name))
