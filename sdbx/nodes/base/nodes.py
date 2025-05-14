from sdbx.nodes.types import *

from sdbx import config
from sdbx.compute import Inference
from sdbx.indexer import ModelType

from sdbx.nodes.helpers import soft_random
# from sdbx.nodes.tuner import NodeTuner

import os

from typing import Any, Union

import PIL
from PIL import Image
from llama_cpp import Llama
from transformers import AutoModel, TensorType as Tensor  # DataType, Tensor (?)

# DataType = Any
# Model = Union[Tensor, Llama]

# spec = config.spec["data"] # needs to be set by system @ launch
# devices = spec["devices"]
# flash_attn = spec["flash-attention"]
# algorithms = config.get_default("algorithms", "schedulers")
# algorithms = config.get_default("algorithms", "solvers")

# llms                = config.model_indexer.index[ModelType.LANGUAGE.value]
# diffusion_models    = config.model_indexer.index[ModelType.DIFFUSION.value]
# lora_models         = config.model_indexer.index[ModelType.LORA.value]
# transformers        = config.model_indexer.index[ModelType.TRANSFORMER.value]
# vae_models          = config.model_indexer.index[ModelType.VAE.value]

# primary_models = {**llms, **diffusion_models}

# @node(name="Genesis Node", display=True)
# def genesis_node(
#     user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
#     model: Literal[*primary_models.keys()] = next(iter([*primary_models.keys()]),""),
# # ) -> DataType:
# ) -> None:
#     strand = NodeTuner().determine_tuning(model)
#     return strand

# @node(name="Load Diffusion", display=True)
# def load_diffusion(
#     model: Literal[*diffusion_models.keys()] = next(iter([*diffusion_models.keys()]), ""),
#     # safety: A[bool, Dependent(on="model_type", when="diffusion")] = False,
#     device: Literal[*spec] = next(iter(spec), "cpu"),
#     precision: A[int, Dependent(on=next(iter(devices), ""), when=(not "cpu")), Slider(min=16, max=64, step=16)] = 16,
#     bfloat: A[bool, Dependent(on="precision", when="16")] = False,
#     verbose: bool = False,
# ) -> AutoModel:
#     #do model stuff
#     return model

# @node(name="Load LoRA", display=True)
# def load_lora(
#     lora: Literal[*lora_models.keys()] = next(iter([*lora_models.keys()]), ""),
#     device: Literal[*devices] = next(iter(devices), "cpu"),
# ) -> Llama:
#     # Inference do_lora_stuff
#     return lora

# @node(name="Load Vae", display=True)
# def load_vae(
#     vae: Literal[*vae_models.keys()] = next(iter([*vae_models.keys()]), ""),
#     device: Literal[*devices] = next(iter(devices), "cpu"),
#     vae_slice: bool = False,
#     vae_tile: bool = True,
#     upcast: bool = False
# ) -> Tensor:
#     #do pipe ops
#     return vae

# @node(name="Load Vision Model", display=True)
# def load_text_model(
#     transformer: Literal[*llms.keys()] = next(iter([*llms.keys()]),""),
#     batch: A[int, Numerical(min=0, max=512, step=1)] = 1,
#     device: Literal[*devices] = next(iter(devices), "cpu"),
#     cpu_only: bool = (True if next(iter(spec), "cpu") == "cpu" else False),
#     gpu_layers: A[int, Dependent(on="cpu_only", when=False), Slider(min=-1, max=35, step=1)] = -1,
#     flash_attention: bool = flash_attn, #autodetect
#     threads: A[int, Slider(min=0, max=64, step=1)] = 8,
#     max_context: A[int, Slider(min=0, max=32767, step=64),] = None, #let ollama do its thing
#     verbose: bool = False,
#     one_time_seed: bool = False,
# ) -> Model:
#     llama = Inference.gguf_load(transformer, threads, max_context, verbose)
#     return llama

# @node(name="Text Prompt", display=True)
# def text_prompt(
#     model: Model,
#     external_user_prompt: str = None,
#     prompt: A[str, Dependent(on="external_user_prompt", when=None), Text(multiline=True, dynamic_prompts=True)] = "", #prompt default
#     negative_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = None,
#     spec_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "", #spec prompt default
#     top_k: A[int, Slider(min=0, max=100)] = None,
#     top_p: A[float, Slider(min=0, max=1, step=0.01, round=0.01)] = None,
#     repeat_penalty: A[float, Numerical(min=0.0, max=2.0, step=0.01, round=0.01),] = None,
#     temperature: A[float, Numerical(min=0.0, max=2.0, step=0.01, round=0.01),] = None,
#     max_tokens: A[int, Numerical(min=0, max=2)] = None,
#     streaming: bool = True,
# ) -> Tensor:
#     if external_user_prompt:
#         user_prompt = external_user_prompt
#     # if none, don't pass values
#     request = Inference.llm_request(spec_prompt, user_prompt, streaming=True)
#     return request

# @node(name="Text Input", display=True)
# def text_input(
#     prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
# ) -> str:
#     return prompt

# @node(name="Image Prompt", display=True)
# def image_prompt(
#     text_encoder: Model,
#     text_encoder_2: Model = None,
#     text_encoder_3: Model = None,
#     lora: Tensor = None,
#     external_prompt: DataType = None,
#     external_negative_prompt: DataType = None,
#     prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "spec prompt",
#     seed: A[int, Numerical(min=0, max=0xFFFFFFFFFFFFFF, step=1, randomizable=True)] = int(soft_random()),
#     # type: ignore
#     gpu_id: A[int, Dependent(on="device", when=(not "cpu")), Slider(min=0, max=100)] = 0,
# ) -> Llama:
#     queue = Inference.push_prompt(prompt, seed)
#     embeddings = Inference.start_encoding(queue, text_encoder, text_encoder_2)
#     return embeddings

# # @node(name="Image Input", display=True)
# # def image_input(
# #  image: format(len(os.listdir(config.get_path("input")))),
# # ) -> Image
# # return image

# # @node(name="Autoencode", display=True)
# # def autoencode(
# #     image: Image,
# #     vae: Tensor,
# #     file_prefix: A[str, Text(multiline=False, dynamic_prompts=True)] = "Shadowbox-",
# # ) -> Image:
# #     batch = Inference.autoencode(vae, latent, file_prefix)
# #     for image in range(batch):
# #         return image

# @node(name="Lora Fuse", path=None, display=True)
# def lora_fuse(
#     model: Model,
#     lora: Tensor = None,
# ) -> Model:
#     #do pipe fuse operation
#     return model

# @node(name="Generate Image", path=None, display=True)
# def generate_image(
#     model: Model,
#     encodings: Tensor,
#     lora_1: Tensor = None,
#     lora_2: Tensor = None,
#     scheduler: Literal[*algorithms] = next(iter(algorithms), ""),
#     inference_steps: A[int, Numerical(min=0, max=250, step=1)] = 8,
#     guidance_scale: A[float, Numerical(min=0.00, max=50.00, step=0.01, round=".01")] = 5,
#     dynamic_guidance: bool = False,
#     compile_unet: bool = False,
#     device: Literal[*spec] = next(iter(spec), "cpu"),
# ) -> Tensor:
#     latent = Inference.run_inference(
#         """queue""",
#         inference_steps,
#         guidance_scale,
#         dynamic_guidance,
#         scheduler,
#         device
#     )
#     Inference.clear_memory_cache()
#     return latent

# @node(name="Autodecode", display=True)
# def autodecode(
#     latent: Tensor,
#     vae: Tensor,
#     file_prefix: A[str, Text(multiline=False, dynamic_prompts=True)] = "Shadowbox-",
# ) -> Image:
#     batch = Inference.autodecode(vae, latent, file_prefix)
#     for image in range(batch):
#         return image

# @node(name="Save / Preview Image", display=True)
# def save_preview_img(
#     image: Image,
#     file_prefix: A[str, Text(multiline=False)]= "Shadowbox-",
#     # format: A[Literal, Dependent(on:"temp", when="False"), "png","jpg","optimize"]] = "optimize",
#     # compress_level: A[int, Slider(min=1, max=9, step=1),  Dependent(on:"format", when=(not "optimize"))] = 7,
#     compress_level: A[int, Slider(min=0, max=4, step=1)]= 4,
#     temp: bool = False,
# ) -> I[Any]:
#         # tempformat="optimize", compress_level="7"
#         image = """Inference.postprocess""" #pipe.image_processor.postprocess(image, output_type='pil')[0]
#         counter = format(len(os.listdir(config.get_path("output")))) #file count
#         file_prefix = os.path.join(config.get_path("output"), file_prefix)
#         image.save(f'{file_prefix + counter}.png')
#         print("Complete.")
#         yield image


@node(name="LLM Print", display=True)
def llm_print(response: str) -> I[str]:
    print("Calculating Resposnse")
    for chunk in range(response):
        delta = chunk["choices"][0]["delta"]
        # if 'role' in delta:               # this prints assistant: user: etc
        # print(delta['role'], end=': ')
        # yield (delta['role'], ': ')
        if "content" in delta:  # the response itself
            print(delta["content"], end="")
            yield delta["content"], ""


@node(name="Outputs String")
def outputs_string(string: str) -> str:
    return string


# @node(name="Outputs String")
# def outputs_string(string: str = None) -> str:
#     return string


# @node(name="Displays String", display=True)
# def displays_string(string: str):
#     print("prints_string prints:", string)


# @node(name="Text Inversion")
# def text_inversion(
#    embeddings: next(other.keys(),"None")
# )
