import math

import torch

from sdbx.nodes.types import *

from sdbx import samplers
from sdbx import sample
from sdbx import model_management

@node
def vae_decode(
    vae: VAE, 
    samples: Latent
) -> Image:
    return vae.decode(samples["samples"])

@node
def vae_decode_tiled(
    vae: VAE, 
    samples: Latent, 
    tile_size: A[int, Numerical(min=320, max=4096, step=64)] = 512
) -> Image:
    return vae.decode_tiled(samples["samples"], tile_x=tile_size // 8, tile_y=tile_size // 8)

@node
def vae_encode(
    vae: VAE, 
    pixels: Image
) -> Latent:
    t = vae.encode(pixels[:, :, :, :3])
    return { "samples": t }

@node
def vae_encode_tiled(
    vae: VAE, 
    pixels: Image, 
    tile_size: A[int, Numerical(min=320, max=4096, step=64)] = 512
) -> Latent:
    t = vae.encode_tiled(pixels[:, :, :, :3], tile_x=tile_size, tile_y=tile_size)
    return { "samples": t }

@node(path="inpaint/")
def vae_encode_for_inpaint(
    vae: VAE, 
    pixels: Image, 
    mask: Mask, 
    grow_mask_by: A[int, Slider(min=0, max=64, step=1)] = 6
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

    return { "samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round()) }

@node
def sampler(
    model: Model,
    positive: Conditioning,
    negative: Conditioning,
    latent: Latent,
    sampler_name: Literal[*samplers.KSampler.SAMPLERS] = None,
    scheduler: Literal[*samplers.KSampler.SCHEDULERS] = None,
    seed: A[int, Numerical(min=0, max=0xffffffffffffffff, step=1, randomizable=True)] = 0,
    steps: A[int, Numerical(min=1, max=10000, step=1)] = 20,
    cfg: A[float, Slider(min=0.0, max=100.0, step=0.01)] = 8.0,
    denoise: A[float, Slider(min=0.0, max=1.0, step=0.01)] = 1.0,
    advanced_options: bool = False,
       disable_noising: A[bool, Dependent(on="advanced_options", when=True)] = False,
       force_full_denoise: A[bool, Dependent(on="advanced_options", when=True)] = False,
       start_step: A[int, Numerical(min=0, max=10000, step=1), Dependent(on="advanced_options", when=True)] = None,
       last_step: A[int, Numerical(min=0, max=10000, step=1), Dependent(on="advanced_options", when=True)] = None,
       noise_types: A[Literal["cpu", "gpu"], Dependent(on="advanced_options", when=True)] = "cpu",
) -> Latent:
    # start_step=None
    # last_step=None
    latent_image = latent["samples"]
    latent_image = sample.fix_empty_latent_channels(model, latent_image)
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=noise_type)
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not current_execution_context().server.receive_all_progress_notifications # exdysa - progress bar
    samples = sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return out

@node
def llm_sampler(
    model: Model,
    prompt: str,
    system_msg: A[str, Text(multiline=True, dynamic_prompts=False)] = "The following instruction describes a task and is paired with an input that provides further context. Write a response that appropriately completes the request.",
    max_tokens: A[int, Numerical(min=1, max=2048, step=1)] = 512,
    temperature: A[float, Numerical(min=0.01, max=2.0, step=0.01)] = 0.20, 
    top_p: A[float, Slider(min=0.1, max=1.0, step=0.01)] = 0.95,
    top_k: A[int, Slider(min=0, max=100, step=1)] = 40,
    frequency_penalty: A[float, Numerical(min=-2.0, max=2.0, step=0.01)] = 0,
    presence_penalty: A[float, Numerical(min=-2.0, max=2.0, step=0.01)] = 0,
    repeat_penalty: A[float, Numerical(min=-2.0, max=2.0, step=0.01)] = 1.10,
    seed: A[int, Numerical(min=0, max=0xffffffffffffffff, step=1)] = 0
) -> str:
        llm = model
        response = llm.create_chat_completion(messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt + " Assistant:"},
        ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
	    seed=seed
            
        )
        return f"{response['choices'][0]['message']['content']}"

@node
def empty_latent_image(
    width: A[int, Numerical(min=16, max=MAX_RESOLUTION, step=8)] = 512,
    height: A[int, Numerical(min=16, max=MAX_RESOLUTION, step=8)] = 512,
    batch_size: A[int, Numerical(min=1, max=4096, step=1512)] = 1,
) -> Latent:
    device = model_management.intermediate_device(),
    latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
    return { "samples": latent }

@node
def empty_image(
    width: A[int, Numerical(min=1, max=MAX_RESOLUTION, step=1)] = 512,
    height: A[int, Numerical(min=1, max=MAX_RESOLUTION, step=1)] = 512,
    batch_size: A[int, Numerical(min=1, max=4096)] = 1,
    color: A[int, Numerical(min=0, max=0xFFFFFF, step=1)] = 0,
    # exdysa - original code used 'display: "color"' probably to trigger a picker
    # device: Literal["cpu", "gpu"] = "cpu"
) -> Image:
    r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
    g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
    b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
    return torch.cat((r, g, b), dim=-1)

@node
def empty_latent_audio(
    seconds: A[float, Numerical(min=1.0, max=1000.0, step=0.1)] = 47.6, # is this constant just random or what
    batch_size: A[int, Numerical(min=1, max=16, step=1)] = 1,
) -> Latent:
        device = model_management.intermediate_device(),
        length = round((seconds * 44100 / 2048) / 2) * 2
        latent = torch.zeros([batch_size, 64, length], device=self.device)
        return { "samples": latent, "type": "audio" }

@node
def vae_encode_audio(
    audio: Audio,
    vae: VAE,
) -> Latent:
        sample_rate = audio["sample_rate"]
        if 44100 != sample_rate:
            import torchaudio  # pylint: disable=import-error
            waveform = torchaudio.functional.resample(audio["waveform"], sample_rate, 44100)
        else:
            waveform = audio["waveform"]

        t = vae.encode(waveform.movedim(1, -1))
        return { "samples": t }

@node
def vae_decode_audio(
    samples: Latent,
    vae: VAE,
) -> Audio:
        audio = vae.decode(samples["samples"]).movedim(-1, 1)
        return { "waveform": audio, "sample_rate": 44100 }
