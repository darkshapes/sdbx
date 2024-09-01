import os
import hashlib
import logging

import torch
import torchaudio

from PIL import Image, ImageOps, ImageSequence, ImageFile
from huggingface_hub import snapshot_download
from natsort import natsorted
import numpy as np
import safetensors.torch

from sdbx import config
from sdbx.nodes.types import *

from sdbx import controlnet
from sdbx import clip_vision as clip_vision_module
from sdbx import diffusers_load
from sdbx import sd
from sdbx import utils
from sdbx.model_downloader import get_filename_list_with_downloadable, get_or_download, KNOWN_CHECKPOINTS, KNOWN_CLIP_VISION_MODELS, KNOWN_GLIGEN_MODELS, KNOWN_UNCLIP_CHECKPOINTS, KNOWN_LORAS, KNOWN_CONTROLNETS, KNOWN_DIFF_CONTROLNETS, KNOWN_VAES, KNOWN_APPROX_VAES, get_huggingface_repo_list, KNOWN_CLIP_MODELS, KNOWN_UNET_MODELS
from sdbx.open_exr import load_exr

@node
def checkpoint_loader(
    ckpt_name: Literal[*get_filename_list_with_downloadable("checkpoints", KNOWN_CHECKPOINTS)] = None
) -> Tuple[Model, CLIP, VAE]:
    ckpt_path = get_or_download("checkpoints", ckpt_name, KNOWN_CHECKPOINTS)
    out = sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=config.folder_paths_from_name("embeddings"))
    return out[:3]

@node
def diffusers_loader(
    model_path: Literal[
        *tuple(frozenset(
            [os.path.relpath(root, start=search_path) for search_path in config.folder_paths_from_name("diffusers") for root, _, files in os.walk(search_path, followlinks=True) if "model_index.json" in files] + get_huggingface_repo_list()
        ))
    ] = None
) -> Tuple[Model, CLIP, VAE]:
    for search_path in config.folder_paths_from_name("diffusers"):
        if os.path.exists(search_path):
            path = os.path.join(search_path, model_path)
            if os.path.exists(path):
                model_path = path
                break
    if not os.path.exists(model_path):
        with sdbx_tqdm():
            model_path = snapshot_download(model_path)
    return diffusers_load.load_diffusers(model_path, output_vae=True, output_clip=True, embedding_directory=config.folder_paths_from_name("embeddings"))

@node
def unclip_checkpoint_loader(
    ckpt_name: Literal[*get_filename_list_with_downloadable("checkpoints", KNOWN_UNCLIP_CHECKPOINTS)] = None
) -> Tuple[Model, CLIP, VAE, CLIPVision]:
    ckpt_path = get_or_download("checkpoints", ckpt_name, KNOWN_UNCLIP_CHECKPOINTS)
    out = sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=True, embedding_directory=config.folder_paths_from_name("embeddings"))
    return out

@node
def load_lora(
    self,
    model: Model,
    clip: CLIP,
    lora_name: Literal[*get_filename_list_with_downloadable("loras", KNOWN_LORAS)] = None,
    strength_model: A[float, Numerical(min=-100.0, max=100.0, step=0.01)] = 1.0,
    strength_clip: A[float, Numerical(min=-100.0, max=100.0, step=0.01)] = 1.0
) -> Tuple[Model, CLIP]:
    if strength_model == 0 and strength_clip == 0:
        return model, clip

    lora_path = get_or_download("loras", lora_name, KNOWN_LORAS)
    lora = None
    if self.loaded_lora is not None:
        if self.loaded_lora[0] == lora_path:
            lora = self.loaded_lora[1]
        else:
            temp = self.loaded_lora
            self.loaded_lora = None
            del temp

    if lora is None:
        lora = utils.load_torch_file(lora_path, safe_load=True)
        self.loaded_lora = (lora_path, lora)

    model_lora, clip_lora = sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
    return model_lora, clip_lora

@node
def lora_loader_model_only(
    model: Model,
    lora_name: Literal[*config.folder_paths_from_name("loras").filename_list] = None,
    strength_model: A[float, Numerical(min=-100.0, max=100.0, step=0.01)] = 1.0
) -> Model:
    loader = LoraLoader()
    return loader.load_lora(model, None, lora_name, strength_model, 0)[0]

class VAELoader:
    @staticmethod
    def vae_list():
        vaes = get_filename_list_with_downloadable("vae", KNOWN_VAES)
        approx_vaes = get_filename_list_with_downloadable("vae_approx", KNOWN_APPROX_VAES)
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd_ = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = utils.load_torch_file(folder_paths.get_full_path("vae_approx", encoder))
        for k in enc:
            sd_["taesd_encoder.{}".format(k)] = enc[k]

        dec = utils.load_torch_file(folder_paths.get_full_path("vae_approx", decoder))
        for k in dec:
            sd_["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd_["vae_scale"] = torch.tensor(0.18215)
            sd_["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd_["vae_scale"] = torch.tensor(0.13025)
            sd_["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd_["vae_scale"] = torch.tensor(1.5305)
            sd_["vae_shift"] = torch.tensor(0.0609)
        return sd_

@node
def vae_loader(
    vae_name: A[Literal[*VAELoader.vae_list()], Name("VAE Name")] = None
) -> VAE:
    if vae_name in ["taesd", "taesdxl", "taesd3"]:
        sd_ = VAELoader.load_taesd(vae_name)
    else:
        vae_path = get_or_download("vae", vae_name, KNOWN_VAES)
        sd_ = utils.load_torch_file(vae_path)
    vae = sd.VAE(sd=sd_)
    return vae

@node
def controlnet_loader(
    control_net_name: Literal[*get_filename_list_with_downloadable("controlnet", KNOWN_CONTROLNETS)] = None
) -> ControlNet:
    controlnet_path = get_or_download("controlnet", control_net_name, KNOWN_CONTROLNETS)
    return controlnet.load_controlnet(controlnet_path)

@node
def diff_controlnet_loader(
    model: Model,
    control_net_name: Literal[*get_filename_list_with_downloadable("controlnet", KNOWN_DIFF_CONTROLNETS)] = None
) -> ControlNet:
    controlnet_path = get_or_download("controlnet", control_net_name, KNOWN_DIFF_CONTROLNETS)
    return controlnet.load_controlnet(controlnet_path, model)

@node
def clip_loader(
    clip_name: Literal[*get_filename_list_with_downloadable("clip", KNOWN_CLIP_MODELS)] = None,
    type: Literal["stable_diffusion", "stable_cascade", "sd3", "stable_audio"] = "stable_diffusion",
) -> CLIP:
        clip_type = sd.CLIPType.STABLE_DIFFUSION
        if type == "stable_cascade":
            clip_type = sd.CLIPType.STABLE_CASCADE
        elif type == "sd3":
            clip_type = sd.CLIPType.SD3
        elif type == "stable_audio":
            clip_type = sd.CLIPType.STABLE_AUDIO
        else:
            logging.warning(f"Unknown clip type argument passed: {type} for model {clip_name}")

        clip_path = get_or_download("clip", clip_name, KNOWN_CLIP_MODELS)
        clip = sd.load_clip(ckpt_paths=[clip_path], embedding_directory=config.folder_paths_from_name("embeddings"), clip_type=clip_type)
        return clip

@node
def dual_clip_loader(
    clip_name_a: Literal[*config.folder_paths_from_name("clip").filename_list] = None,
    clip_name_b: Literal[*config.folder_paths_from_name("clip").filename_list] = None,
    type: Literal["sdxl", "sd3"] = "sdxl",
) -> CLIP:
    clip_path_a = config.folder_paths["clip"].get_path_from_filename(clip_name_a)
    clip_path_b = config.folder_paths["clip"].get_path_from_filename(clip_name_b)
    if type == "sdxl":
        clip_type = sd.CLIPType.STABLE_DIFFUSION
    elif type == "sd3":
        clip_type = sd.CLIPType.SD3
    else:
        raise ValueError(f"Unknown clip type argument passed: {type} for model {clip_name_a} and {clip_name_b}")

    clip = sd.load_clip(ckpt_paths=[clip_path_a, clip_path_b], embedding_directory=config.folder_paths_from_name("embeddings"), clip_type=clip_type)
    return clip

@node
def clip_vision_loader(
    clip_name: Literal[*get_filename_list_with_downloadable("clip_vision", KNOWN_CLIP_VISION_MODELS)] = None
) -> CLIPVision:
    clip_path = get_or_download("clip_vision", clip_name, KNOWN_CLIP_VISION_MODELS)
    clip_vision = clip_vision_module.load(clip_path)
    return clip_vision

@node
def gligen_loader(
    gligen_name: Literal[*get_filename_list_with_downloadable("gligen", KNOWN_GLIGEN_MODELS)] = None
) -> GLIGen:
        gligen_path = get_or_download("gligen", gligen_name, KNOWN_GLIGEN_MODELS)
        return sd.load_gligen(gligen_path)

@node
def style_model_loader(
    style_model_name: Literal[*config.folder_paths_from_name("style_models").filename_list] = None
) -> StyleModel:
        style_model_path = config.folder_paths_from_name("style_models").get_path_from_filename(style_model_name)
        return sd.load_style_model(style_model_path)
        
@node
def load_latent(
    latent: Literal[*sorted(
        [f for f in os.listdir(config.get_path("input")) if os.path.isfile(os.path.join(config.get_path("input"), f)) and f.endswith(".latent")]
    )] = None
) -> Latent:
    latent_path = folder_paths.get_annotated_filepath(latent)
    latent = safetensors.torch.load_file(latent_path, device="cpu")
    multiplier = 1.0
    if "latent_format_version_0" not in latent:
        multiplier = 1.0 / 0.18215
    samples = {"samples": latent["latent_tensor"].float() * multiplier}
    return samples

@node
def load_image(
    image: Literal[*natsorted([f for f in os.listdir(config.get_path("input")) if os.path.isfile(os.path.join(config.get_path("input"), f))])] = None,
    image_upload: bool = True,
) -> Tuple[Image, Mask]:
    image_path = folder_paths.get_annotated_filepath(image)

    img = node_helpers.pillow(Image.open, image_path)

    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']

    # maintain the legacy path
    # this will ultimately return a tensor, so we'd rather have the tensors directly
    # from cv2 rather than get them out of a PIL image
    _, ext = os.path.splitext(image)
    if ext == ".exr":
        return load_exr(image_path, srgb=False)
    with open_image(image_path) as img:
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)


@node
def load_image_mask(
    image: Literal[*sorted([f for f in os.listdir(config.get_path("input")) if os.path.isfile(os.path.join(config.get_path("input"), f))])] = None,
    image_upload: bool = True,
    channel: Literal["alpha", "red", "green", "blue"] = "alpha"
) -> Mask:
    image_path = folder_paths.get_annotated_filepath(image)
    i = node_helpers.pillow(Image.open, image_path)
    i = node_helpers.pillow(ImageOps.exif_transpose, i)
    if i.getbands() != ("R", "G", "B", "A"):
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        i = i.convert("RGBA")
    mask = None
    c = channel[0].upper()
    if c in i.getbands():
        mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)
        if c == 'A':
            mask = 1. - mask
    else:
        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
    return mask.unsqueeze(0)

@node
def llm_loader(
    ckpt_name: Literal[*config.folder_paths_from_name("llm").filename_list] = None,
    max_ctx: A[int, Numerical(min=128, max=128000, step=64)] = 2048,
    gpu_layers: A[int, Slider(min=0, max=100, step=1)] = 27,
    n_threads: A[int, Slider(min=1, max=100, step=1)] = 8,
) -> Model:
    ckpt_path = folder_paths.get_full_path("llm", ckpt_name)
    llm = Llama(model_path = ckpt_path, chat_format="chatml", offload_kqv=True, f16_kv=True, use_mlock=False, embedding=False, n_batch=1024, last_n_tokens_size=1024, verbose=True, seed=42, n_ctx = max_ctx, n_gpu_layers=gpu_layers, n_threads=n_threads,) 
    return llm 

SUPPORTED_AUDIO_FORMATS = ('.wav', '.mp3', '.ogg', '.flac', '.aiff', '.aif') # TODO: use mimetype?

@node
def load_audio(
    audio: Literal[*sorted(
        [f for f in os.listdir(config.get_path("input")) if (os.path.isfile(os.path.join(config.get_path("input"), f))) and f.endswith(SUPPORTED_AUDIO_FORMATS)]
    )] = None,
    audio_upload: bool = True,
) -> Audio:
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = torchaudio.load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return audio
