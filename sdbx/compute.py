"""
Credits:
Felixsans
"""

import gc
import os
from time import perf_counter
from collections import defaultdict
from enum import Enum

from sdbx import config
from sdbx.config import cache
from sdbx.nodes.helpers import seed_planter, soft_random, get_gpus

import torch
from transformers import AutoTokenizer, AutoModel
from diffusers import AutoPipelineForText2Image, AutoencoderKL
from diffusers.schedulers import AysSchedules, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler, LCMScheduler, TCDScheduler, HeunDiscreteScheduler, UniPCMultistepScheduler, LMSDiscreteScheduler, DEISMultistepScheduler

from pydantic import BaseModel, ValidationError


# Define Enums for data types and schedulers
class DataType(Enum):
    FP64 = ("fp64", torch.float64)
    FP32 = ("fp32", torch.float32)
    FP16 = ("fp16", torch.float16)
    BF16 = ("BF16", torch.bfloat16)
    # Add other data types as needed


class SchedulerType(Enum):
    EulerDiscreteScheduler = ("EulerDiscreteScheduler", EulerDiscreteScheduler)
    EulerAncestralDiscreteScheduler = ("EulerAncestralDiscreteScheduler", EulerAncestralDiscreteScheduler)
    DPMSolverMultistepScheduler = ("DPMSolverMultistepScheduler", DPMSolverMultistepScheduler)
    DDIMScheduler = ("DDIMScheduler", DDIMScheduler)
    LCMScheduler = ("LCMScheduler", LCMScheduler)
    TCDScheduler = ("TCDScheduler", TCDScheduler)
    HeunDiscreteScheduler = ("HeunDiscreteScheduler", HeunDiscreteScheduler)
    UniPCMultistepScheduler = ("UniPCMultistepScheduler", UniPCMultistepScheduler)
    LMSDiscreteScheduler = ("LMSDiscreteScheduler", LMSDiscreteScheduler)
    DEISMultistepScheduler = ("DEISMultistepScheduler", DEISMultistepScheduler)
    # Add other schedulers as needed


class Inference:
    def __init__(self):
        self.config_path = config.get_path("models.metadata")
        self.queue = []

    def float_converter(self, data_type_key):
        try:
            data_type_enum = DataType[data_type_key]
            return data_type_enum.value  # Returns tuple (string, torch data type)
        except KeyError:
            raise ValueError(f"Invalid data type: {data_type_key}")

    def algorithm_converter(self, algorithm_name):
        for scheduler in SchedulerType:
            if scheduler.value[0] == algorithm_name:
                return scheduler.value[1]
        raise ValueError(f"Invalid scheduler name: {algorithm_name}")

    def declare_encoders(self, device, transformers):
        self.device = device
        self.tokenizers = {}
        self.text_encoders = {}
        self.transformers = transformers  # Should be a dict with relevant data

        for class_name, transformer_data in self.transformers.items():
            weights = transformer_data.get('path')  # Assuming transformer_data contains 'path' to weights
            dtype_key = transformer_data.get('dtype_or_context_length', 'FP16')
            variant, tensor_dtype = self.float_converter(dtype_key)

            # Now, load the tokenizer and text encoder
            self.tokenizers[class_name] = AutoTokenizer.from_pretrained(weights)
            self.text_encoders[class_name] = AutoModel.from_pretrained(
                weights,
                torch_dtype=tensor_dtype,
                variant=variant,
            ).to(self.device)

        return self.tokenizers, self.text_encoders

    def _generate_embeddings(self, prompts, tokenizers, text_encoders):
        embeddings_list = []
        self.prompts = prompts
        self.tokenizers = tokenizers
        self.text_encoders = text_encoders

        # Create instances of the models
        for prompt, tokenizer, text_encoder in zip(self.prompts, self.tokenizers.values(), self.text_encoders.values()):
            cond_input = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            with torch.no_grad():
                prompt_embed = text_encoder(cond_input.input_ids.to(self.device), output_hidden_states=True)
                embeddings_list.append(prompt_embed.hidden_states[-2])

        prompt_embeds = torch.cat(embeddings_list, dim=-1)
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        pooled_prompt_embeds = torch.zeros((prompt_embeds.shape[0], prompt_embeds.shape[-1]), device=prompt_embeds.device)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def encode_prompt(self, queue, tokenizers, text_encoders):
        self.queue = queue
        self.tokenizers = tokenizers
        self.text_encoders = text_encoders
        for generation in self.queue:
            generation['embeddings'] = self._generate_embeddings(
                [generation['prompt']],
                self.tokenizers,
                self.text_encoders
            )
        return self.queue

    def cache_jettison(self, device, pipe=None, lora=False, discard=True):
        self.device = device
        if pipe and lora:
            pipe.unload_lora_weights()
        if discard:
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()

    def latent_diffusion(self, device, parameters):
        self.device = device
        self.parameters = parameters

        self.model = self.parameters.model["file"]
        self.lora_dict = self.parameters.lora_dict
        self.lora_name = os.path.basename(self.lora_dict.get('file', ''))
        self.lora_path = os.path.dirname(self.lora_dict.get('file', ''))
        variant, tensor_dtype = self.float_converter(self.parameters.model.get('dtype', 'FP16'))
        self.use_low_cpu = self.parameters.optimized.get('cpu', False)
        scheduler_name = self.parameters.optimized.get('scheduler', 'EulerDiscreteScheduler')
        self.scheduler = self.algorithm_converter(scheduler_name)

        self.pipe_args = {
            "torch_dtype": tensor_dtype,
            "low_cpu_mem_usage": self.use_low_cpu,
            "variant": variant,
            "tokenizer": None,
            "text_encoder": None,
            "tokenizer_2": None,
            "text_encoder_2": None,
        }

        self.pipe = AutoPipelineForText2Image.from_pretrained(self.model, **self.pipe_args).to(device)
        self.scheduler_args = self.parameters.schedule
        self.pipe.scheduler = self.scheduler.from_config(self.pipe.scheduler.config, **self.scheduler_args)

        if self.parameters.optimized.get('sequential_offload', False):
            self.pipe.enable_sequential_cpu_offload()
        if self.parameters.optimized.get('cpu_offload', False):
            self.pipe.enable_model_cpu_offload()

        if self.lora_dict.get('unet_only', False):
            self.pipe.unet.load_attn_procs(self.lora_path, weight_name=self.lora_name)
        else:
            self.pipe.load_lora_weights(self.lora_path, weight_name=self.lora_name)

        if self.parameters.optimized.get('fuse', False):
            self.pipe.fuse_lora(lora_scale=self.lora_dict.get('scale', 1.0))

        self.generator = torch.Generator(device=self.device)
        self.gen_args = {}
        self.gen_args["output_type"] = 'latent'

        if self.parameters.optimized.get('dynamic_cfg', False):
            def dynamic_cfg(pipe, step_index, timestep, callback_val):
                if step_index == int(pipe.num_timesteps * 0.5):
                    callback_val['prompt_embeds'] = callback_val['prompt_embeds'].chunk(2)[-1]
                    callback_val['add_text_embeds'] = callback_val['add_text_embeds'].chunk(2)[-1]
                    callback_val['add_time_ids'] = callback_val['add_time_ids'].chunk(2)[-1]
                    pipe._guidance_scale = 0.0
                return callback_val

            self.gen_args["callback_on_step_end"] = dynamic_cfg
            self.gen_args["callback_on_step_end_tensor_inputs"] = ['prompt_embeds', 'add_text_embeds', 'add_time_ids']

        if self.parameters.optimized.get('compile_unet', False):
            self.pipe.unet = torch.compile(
                self.pipe.unet,
                mode=self.parameters.optimized['compile'].get('mode', 'default'),
                fullgraph=self.parameters.optimized['compile'].get('fullgraph', False)
            )

        for generation in self.queue:
            seed_planter(self.parameters.optimized.get('seed', 42))
            self.generator.manual_seed(generation['seed'])

            generation['latents'] = self.pipe(
                prompt_embeds=generation['embeddings'][0],
                negative_prompt_embeds=generation['embeddings'][1],
                pooled_prompt_embeds=generation['embeddings'][2],
                negative_pooled_prompt_embeds=generation['embeddings'][3],
                num_inference_steps=self.parameters.gen_dict.get('num_inference_steps', 20),
                generator=self.generator,
                **self.gen_args,
            ).images

        return self.queue

    def decode_latent(self, parameters):
        self.parameters = parameters
        if self.parameters.optimized.get('upcast_vae', False):
            self.pipe.upcast_vae()
        if self.parameters.optimized.get('slice', False):
            self.pipe.upcast_vae()

        self.autoencoder_path = self.parameters.optimized.get('vae', None)
        if not self.autoencoder_path:
            raise ValueError("VAE path not provided")

        dtype = self.float_converter(self.parameters.vae_dict.get('dtype', 'FP16'))[1]
        self.vae = AutoencoderKL.from_single_file(
            self.autoencoder_path,
            torch_dtype=dtype,
            cache_dir="vae_"
        ).to(self.device)

        self.pipe.vae = self.vae
        self.pipe.upcast_vae()

        output_path = config.get_path("output")
        counter = sum(1 for s in os.listdir(output_path) if s.endswith('png'))  # get existing images

        for i, generation in enumerate(self.queue, start=1):
            generation['latents'] = generation['latents'].to(next(self.pipe.vae.post_quant_conv.parameters()).dtype)

            with torch.no_grad():
                image = self.pipe.vae.decode(
                    generation['latents'] / self.pipe.vae.config.scaling_factor,
                    return_dict=False,
                )[0]

            image = self.pipe.image_processor.postprocess(image, output_type='pil')[0]
            counter += 1
            filename = f"{self.parameters.optimized.get('file_prefix', 'output')}-{counter}-batch-{i}.png"
            image.save(os.path.join(output_path, filename))

        return self.queue


class QueueManager:
    def __init__(self):
        self.queue = []

    def prompt(self, prompts_with_seeds):
        self.queue = []
        for prompt_text, seed in prompts_with_seeds:
            self.queue.append({
                'prompt': prompt_text,
                'seed': seed,
            })
        self.embed_start = perf_counter()
        return self.queue

    def metrics(self):
        total_time = perf_counter() - self.embed_start
        num_items = len(self.queue)
        avg_time = total_time / num_items if num_items > 0 else 0
        print(f'Total time: {total_time:.1f}s, Average time per item: {avg_time:.1f}s')
        return self.queue


def test_process():
    filename = "tPonynai3_v55.safetensors"
    path = config.get_path("models.image")
    device = "cuda"
    print("Beginning inference")
    print(os.path.join(path, filename))
    node_tuner = NodeTuner()
    node_tuner.determine_tuning(filename)
    if node_tuner.fetch is not None:
        seed = soft_random()
        prompt_text = node_tuner.pipe_dict.get("prompt", "A default prompt")
        prompts_with_seeds = [(prompt_text, seed)]
        queue_manager = QueueManager()
        active_queue = queue_manager.prompt(prompts_with_seeds)
        inference_instance = Inference()
        tokenizer, text_encoder = inference_instance.declare_encoders(device, node_tuner.transformers)
        embeddings_queue = inference_instance.encode_prompt(active_queue, tokenizer, text_encoder)
        inference_instance.cache_jettison(device, discard=node_tuner.optimized.get("cache_jettison", False))
        inference_instance.latent_diffusion(device, node_tuner)
        inference_instance.cache_jettison(device, discard=node_tuner.optimized.get("cache_jettison", False))
        inference_instance.decode_latent(node_tuner)
        queue_manager.metrics()