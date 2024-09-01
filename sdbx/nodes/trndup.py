#
# - softRandom returns a deterministic random number
# - hardRandom does not
# - tensorify creates an array of default size 4x1 using either softRandom or hardRandom
#   set the flag to True if you are using hard numbers (or want larger division)

from random import random
import numpy as np
from numpy.random import SeedSequence, Generator, Philox, BitGenerator
import secrets as secrets

class softRndmc:
    def softRandom(size=0x2540BE3FF):  #use Philox
        entropy = f"0x{secrets.randbits(128):x}" # git gud entropy
        rndmc = Generator(Philox(SeedSequence(int(entropy,16))))
        return rndmc.integers(0, size) 

class hardRndmc:
    def hardRandom(hardness=5): # use secrets
        return int(secrets.token_hex(hardness),16) # make hex secret be int

class tensorRandom:
    def tensorify(hard, size=4):
        num = []
        for s in range(size): # make array, convert float, negate it randomly
            if hard==False: # divide 10^10, truncate float
                conv = '{0:.6f}'.format((float(softRandom()))/0x2540BE400)
            else:  # divide 10^12, truncate float
                conv = '{0:.6f}'.format((float(hardRandom()))/0xE8D4A51000)
            num.append(float(conv)) if secrets.choice([True, False]) else num.append(float(conv)*-1)
        return num
