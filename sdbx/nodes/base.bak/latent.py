import os
import json

import torch

import safetensors.torch

from sdbx import config
from sdbx.nodes.types import *

@node
def save_latent(
    samples: Latent, 
    filename_prefix: str = "sdbx", 
    prompt: str = None,
    extra_pnginfo: Dict[str, Any] = None
):
    full_output_folder, filename, counter, subfolder, filename_prefix = config.get_image_save_path(filename_prefix, config.get_path("output"))

    # support save metadata for latent sharing
    prompt_info = ""
    if prompt is not None:
        prompt_info = json.dumps(prompt)

    metadata = {"prompt": prompt_info}
    if extra_pnginfo is not None:
        for x in extra_pnginfo:
            metadata[x] = json.dumps(extra_pnginfo[x])

    file = f"{filename}_{counter:05}_.latent"

    results = [{
        "filename": file,
        "subfolder": subfolder,
        "type": "output"
    }]

    file = os.path.join(full_output_folder, file)

    output = {
        "latent_tensor": samples["samples"],
        "latent_format_version_0": torch.tensor([])
    }

    utils.save_torch_file(output, file, metadata=metadata)
    # return {"ui": {"latents": results}}

@node
def load_latent(
    latent: Literal[*sorted(
        [f for f in os.listdir(config.get_path("input")) if os.path.isfile(os.path.join(config.get_path("input"), f)) and f.endswith(".latent")]
    )]
) -> Latent:
    latent_path = folder_paths.get_annotated_filepath(latent)
    latent = safetensors.torch.load_file(latent_path, device="cpu")
    multiplier = 1.0
    if "latent_format_version_0" not in latent:
        multiplier = 1.0 / 0.18215
    samples = {"samples": latent["latent_tensor"].float() * multiplier}
    return samples
