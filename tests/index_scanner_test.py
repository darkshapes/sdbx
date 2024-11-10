# import os
# from time import process_time_ns
# from sdbx import logger
# from sdbx.config import config
# from sdbx.indexer import ReadMeta, EvalMeta
# import uvicorn
# from fastapi import FastAPI
# name = input("Please enter a filename or an indexed sub-directory name:")
# try:
#     name_check = config.get_path(f"models.{name}")
# except:
#     not_name = True
#     name_check = 0
#     """
#     not name
#     """
# try:
#     root_check = config.get_path(name)
# except:
#     not_root = True
#     root_check = 0
#     """
#     no root
#     """

# if not os.path.isdir(name_check) or os.path.isdir(root_check) :
#     name_path = input("Enter path to file (ie: 'folder_name', or just 'models'):")
#     try:
#         path_name = config.get_path(f"models.{name_path}")
#     except KeyError:
#         logger.debug(f"No path {name_path}.", exc_info=True)
#         print(f"No path {name_path}")
#     else:
#         full_path = os.path.normpath(os.path.join(path_name, name)) #multi read
#         import variable_monitor_test as varmont
#         metareader = ReadMeta(full_path).data()
#         if metareader is not None:
#             evaluate = EvalMeta(metareader).data()
#             varmont.s(locals())
# else:
#     if name == "models":
#         root = name
#         name= input("Enter sub directory:")
#     else:
#         root = "models"
#     try:
#         path_name = config.get_path(f"{root}.{name}")
#     except KeyError:
#         logger.debug(f"No path {name}.", exc_info=True)
#         print(f"No path {name}")
#     for each in os.listdir(path_name): ###SCAN DIRECTORY
#         if not os.path.isdir(os.path.join(path_name, each)):
#             filename = each  # "PixArt-Sigma-XL-2-2K-MS.safetensors"
#             full_path = os.path.join(path_name, filename)
#             import variable_monitor_test as varmont
#             metareader = ReadMeta(full_path).data()
#             if metareader is not None:
#                 evaluate = EvalMeta(metareader).data()
#                 varmont.s(locals())
                