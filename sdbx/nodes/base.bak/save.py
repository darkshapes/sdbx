import io
import os
import json
import struct
import random

import torch

from PIL.PngImagePlugin import PngInfo
import numpy as np

from sdbx import config
from sdbx.nodes.types import *

# from sdbx.cmd import folder_paths

@node
def save_latent(
    samples: Latent, 
    filename_prefix: str = "sdbx", 
    prompt: str = None, 
    extra_pnginfo: Dict[str, Any] = None
) -> None:
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
    return {"ui": {"latents": results}}

@node
def save_image(
    pixels: Image,
    filename_prefix: Annotated[str, Text()] = "sdbx",
    prompt: str = None, 
    extra_pnginfo: Dict[str, Any] = None
) -> None:
    type = "output"
    prefix_append = ""
    compress_level = 4
    filename_prefix += self.prefix_append
    full_output_folder, filename, counter, subfolder, filename_prefix = config.get_image_save_path(filename_prefix, config.get_path("output"), images[0].shape[1], images[0].shape[0])
    results = list()
    for (batch_number, image) in enumerate(images):
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_with_batch_num}_{counter:05}_.png"
        abs_path = os.path.join(full_output_folder, file)
        img.save(abs_path, pnginfo=metadata, compress_level=self.compress_level)
        results.append({
            "abs_path": os.path.abspath(abs_path),
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        counter += 1

    return { "ui": { "images": results } }

@node
# exdysa - ??? dont know how to handle this whole thing
def preview_image( # previously class PreviewImage(SaveImage)
    pixels: Image,
    prompt: str = None, 
    extra_pnginfo: Dict[str, Any] = None
) -> None:
        type = "temp" # and this
        prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        compress_level = 1

@node
def save_audio(
    audio: Audio,
    filename_prefix: Annotated[str, Text()] = "audio/sdbx_", 
    prompt: str = None, 
    extra_pnginfo: Dict[str, Any] = None
) -> None:
    filename_prefix += "Shadowbox"
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
    results = list()
    metadata = {}

    if prompt is not None:
        metadata["prompt"] = json.dumps(prompt)
    if extra_pnginfo is not None:
        for x in extra_pnginfo:
            metadata[x] = json.dumps(extra_pnginfo[x])

    for (batch_number, waveform) in enumerate(audio["waveform"]):
        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_with_batch_num}_{counter:05}_.flac"

        buff = io.BytesIO()
        torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")

        buff = insert_or_replace_vorbis_comment(buff, metadata)

        with open(os.path.join(full_output_folder, file), 'wb') as f:
            f.write(buff.getbuffer())

        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        counter += 1

    return { "ui": { "audio": results } }


# @node
# def preview_audio(SaveAudio):
#     def __init__(self):
#         self.output_dir = folder_paths.get_temp_directory()
#         self.type = "temp"
#         self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required":
#                     {"audio": ("AUDIO", ), },
#                 "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
#                 }

def create_vorbis_comment_block(comment_dict, last_block):
    vendor_string = b'Shadowbox'
    vendor_length = len(vendor_string)

    comments = []
    for key, value in comment_dict.items():
        comment = f"{key}={value}".encode('utf-8')
        comments.append(struct.pack('<I', len(comment)) + comment)

    user_comment_list_length = len(comments)
    user_comments = b''.join(comments)

    comment_data = struct.pack('<I', vendor_length) + vendor_string + struct.pack('<I', user_comment_list_length) + user_comments
    if last_block:
        id = b'\x84'
    else:
        id = b'\x04'
    comment_block = id + struct.pack('>I', len(comment_data))[1:] + comment_data

    return comment_block

def insert_or_replace_vorbis_comment(flac_io, comment_dict):
    if len(comment_dict) == 0:
        return flac_io

    flac_io.seek(4)

    blocks = []
    last_block = False

    while not last_block:
        header = flac_io.read(4)
        last_block = (header[0] & 0x80) != 0
        block_type = header[0] & 0x7F
        block_length = struct.unpack('>I', b'\x00' + header[1:])[0]
        block_data = flac_io.read(block_length)

        if block_type == 4 or block_type == 1:
            pass
        else:
            header = bytes([(header[0] & (~0x80))]) + header[1:]
            blocks.append(header + block_data)

    blocks.append(create_vorbis_comment_block(comment_dict, last_block=True))

    new_flac_io = io.BytesIO()
    new_flac_io.write(b'fLaC')
    for block in blocks:
        new_flac_io.write(block)

    new_flac_io.write(flac_io.read())
    return new_flac_io

