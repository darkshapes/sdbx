# # import os
# # import logging
# # from sdbx import logger
# # from sdbx.capacity import SystemCapacity as sys_cap
# # from sdbx.config import config
# # from rich.logging import RichHandler
# # from rich.console import Console
# # from rich.logging import RichHandler

# # log_level = "INFO"
# # msg_init = None
# # handler = RichHandler(console=Console(stderr=True))

# # if handler is None:
# #     handler = logging.StreamHandler(sys.stdout)  # same as print
# #     handler.propagate = False

# # formatter = logging.Formatter(
# #     fmt="%(message)s",
# #     datefmt="%Y-%m-%d %H:%M:%S",
# # )
# # handler.setFormatter(formatter)
# # logging.root.setLevel(log_level)
# # logging.root.addHandler(handler)

# # if msg_init is not None:
# #     logger = logging.getLogger(__name__)
# #     logger.info(msg_init)

# # log_level = getattr(logging, log_level)
# # logger = logging.getLogger(__name__)

# # logger.info("\nAnalyzing model & system capacity\n  Please wait...")

# #create_capacity = sys_cap().write_capacity()
# index    = config.model_indexer
# optimize = config.node_tuner
# log_level = "INFO"
# msg_init = None

# #create_index = index.write_index()     # (defaults to config/index.json)

# logger.info(f"Ready.")
# #name_path = input("""
# #Please type the file of an available checkpoint.
# #Path will be detected.
# #(default:vividpdxl_realVAE.safetensors):""" or "vividpdxl_realVAE.safetensors")
# # "virtualDiffusionPony_25B3C4N3.safetensors"
# name_path = "hellaineMixPDXL_v45.safetensors"
# name_path = os.path.basename(name_path)
# diffusion_index = config.get_default("index","DIF")
# name_path = name_path.strip()
# name_path = os.path.basename(name_path)
# if ".safetensors" not in name_path:
#      name_path = name_path + ".safetensors"
# for key,val in diffusion_index.items():
#     if name_path in key:
#         model = key
#         pass

# defaults = optimize.determine_tuning(model)
# defaults["generate_image"]["width"] = 832
# defaults["generate_image"]["height"] = 1152
# defaults["diffusion_prompt"]["batch"] = 1
# from sdbx.nodes.base import nodes

# # #pipe = nodes.empty_cache(transformer_models, lora_pipe, unet_pipe, vae_pipe)

# # device = nodes.force_device(**defaults.get("force_device"))
# # queue = nodes.diffusion_prompt(**defaults.get("diffusion_prompt"))
# # if defaults.get("load_transformer",0) != 0: tokenizers, text_encoders = nodes.load_transformer(**defaults.get("load_transformer"))
# # if defaults.get("encode_prompt",0) != 0: queue = nodes.encode_prompt(**defaults.get("encode_prompt"), queue=queue, tokenizers_in=tokenizers, text_encoders_in=text_encoders,)
# # if defaults.get("load_vae_model",0) != 0: vae = nodes.load_vae_model(**defaults.get("load_vae_model"))
# # pipe = nodes.diffusion_pipe(**defaults.get("diffusion_pipe"), vae=vae)
# # if defaults.get("load_lora",0) != 0: pipe = nodes.load_lora(**defaults.get("load_lora"), pipe=pipe)
# # pipe = nodes.load_scheduler(**defaults.get("noise_scheduler"), pipe=pipe)
# # pipe, latent = nodes.generate_image(**defaults.get("generate_image"), pipe=pipe, queue=queue)
# # if defaults.get("autodecode",0) != 0: image = nodes.autodecode(**defaults.get("autodecode"), pipe=pipe, latent=latent)
