import os
import json
import struct
from time import process_time_ns
from math import isclose
from pathlib import Path
from collections import defaultdict
from llama_cpp import Llama
from sdbx import config, logger
from sdbx.config import config
from sdbx.indexer import ReadMeta, EvalMeta


print(f'begin: {process_time_ns()*1e-6} ms')

peek = config.get_default("tuning", "peek") # block and tensor values for identifying
known = config.get_default("tuning", "known") # raw block & tensor data

path_name = config.get_path("models.image")
print(path_name)

each = "stable_cascade_stage_b.safetensors" ##SCAN SINGLE FILE
full_path = os.path.normpath(os.path.join(path_name, each)) #multi read
metareader = ReadMeta(full_path, verbose=True).data()
print(metareader)
evaluate = EvalMeta(metareader, verbose=True).data()


# for each in os.listdir(path_name): ###SCAN DIRECTORY
#       if not os.path.isdir(os.path.join(path_name, each)):
#         filename = each  # "PixArt-Sigma-XL-2-2K-MS.safetensors"
#         full_path = os.path.join(path_name, filename)
#         metareader = ReadMeta(full_path).data()
#         if metareader is not None:
#             evaluate = EvalMeta(metareader).data()



print(f'end: {process_time_ns()*1e-6} ms')