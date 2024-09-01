import math

import torch

from sdbx import config
from sdbx.nodes.types import *

@node
def vae_decode(vae: VAE, samples: Latent) -> Image:
    return vae.decode(samples["samples"])

@node
def vae_decode_tiled(
    vae: VAE, 
    samples: Latent, 
    tile_size: A[int, Numerical(min=320, max=4096, step=64)] = 512
) -> Image:
    return vae.decode_tiled(samples["samples"], tile_x=tile_size // 8, tile_y=tile_size // 8)

@node
def vae_encode(vae: VAE, pixels: Image) -> Latent:
    t = vae.encode(pixels[:, :, :, :3])
    return {"samples": t}

@node
def vae_encode_tiled(
    vae: VAE, 
    pixels: Image, 
    tile_size: A[int, Numerical(min=320, max=4096, step=64)] = 512
) -> Latent:
    t = vae.encode_tiled(pixels[:, :, :, :3], tile_x=tile_size, tile_y=tile_size)
    return {"samples": t}

@node(path="inpaint/")
def vae_encode_for_inpaint(
    vae: VAE, 
    pixels: Image, 
    mask: Mask, 
    grow_mask_by: A[int, Numerical(min=0, max=64, step=1)] = 6
) -> Latent:
    x = (pixels.shape[1] // vae.downscale_ratio) * vae.downscale_ratio
    y = (pixels.shape[2] // vae.downscale_ratio) * vae.downscale_ratio
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

    pixels = pixels.clone()
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % vae.downscale_ratio) // 2
        y_offset = (pixels.shape[2] % vae.downscale_ratio) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

    # Grow mask by a few pixels to keep things seamless in latent space
    if grow_mask_by == 0:
        mask_erosion = mask
    else:
        kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
        padding = math.ceil((grow_mask_by - 1) / 2)
        mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

    m = (1.0 - mask.round()).squeeze(1)
    for i in range(3):
        pixels[:, :, :, i] -= 0.5
        pixels[:, :, :, i] *= m
        pixels[:, :, :, i] += 0.5
    t = vae.encode(pixels)

    return {"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())}