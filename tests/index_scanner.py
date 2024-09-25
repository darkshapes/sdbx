import os
from time import process_time_ns
from sdbx.config import config
from sdbx.indexer import ReadMeta, EvalMeta


print(f'begin: {process_time_ns()*1e-6} ms')

path_name = config.get_path("models.download")
each = "diffusion_pytorch_model.safetensors" ##SCAN SINGLE FILE
full_path = os.path.normpath(os.path.join(path_name, each)) #multi read
metareader = ReadMeta(full_path, verbose=True).data()
print(metareader)
evaluate = EvalMeta(metareader, verbose=True).data()
print(evaluate)

# for each in os.listdir(path_name): ###SCAN DIRECTORY
#       if not os.path.isdir(os.path.join(path_name, each)):
#         filename = each  # "PixArt-Sigma-XL-2-2K-MS.safetensors"
#         full_path = os.path.join(path_name, filename)
#         metareader = ReadMeta(full_path).data()
#         if metareader is not None:
#             evaluate = EvalMeta(metareader).data()



print(f'end: {process_time_ns()*1e-6} ms')