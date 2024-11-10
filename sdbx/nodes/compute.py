# """
# Credits:
# Felixsans
# """
# import re
# import gc
# import os
# import torch
# from PIL import Image
# from collections import defaultdict

# from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
# from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
# from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
# from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import EDMDPMSolverMultistepScheduler
# from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from diffusers.schedulers.scheduling_lcm import LCMScheduler
# from diffusers.schedulers.scheduling_tcd import TCDScheduler
# from diffusers.schedulers.scheduling_utils import AysSchedules
# from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
# from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
# from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
# from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler

# #from hidiffusion import apply_hidiffusion, remove_hidiffusion
# from diffusers.utils import logging as df_log
# from transformers import logging as tf_log
# from diffusers import AutoencoderKL, StableDiffusionXLPipeline, AutoPipelineForText2Image, AutoencoderTiny
# from transformers import CLIPTokenizerFast, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, T5Tokenizer, T5EncoderModel
# import accelerate

# from sdbx.config import logging
# from sdbx.config import config
# from sdbx.nodes.helpers import seed_planter

# class T2IPipe:
#     # __call__? NO __init__! ONLY __call__. https://huggingface.co/docs/diffusers/main/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForText2Image

# ############## MUTE LOGGING TEMPORARILY
#     def hf_log(self, on=False, fatal=False):
#         if on is True:
#             tf_log.enable_default_handler()
#             df_log.enable_default_handler()
#             tf_log.set_verbosity_warning()
#             df_log.set_verbosity_warning()
#         if fatal is True:
#             tf_log.disable_default_handler()
#             df_log.disable_default_handler()
#             tf_log.set_verbosity(tf_log.FATAL)
#             tf_log.set_verbosity(df_log.FATAL)
#         else:
#             tf_log.set_verbosity_error()
#             df_log.set_verbosity_error()

# ############## TORCH DATATYPE
#     def float_converter(self, old_index):
#         float_chart = {
#                 "F64": ["fp64", torch.float64],
#                 "F32": ["fp32", torch.float32],
#                 "F16": ["fp16", torch.float16],
#                 "BF16": ["bf16", torch.bfloat16],
#                 "F8_E4M3": ["fp8e4m3fn", torch.float8_e4m3fn],
#                 "F8_E5M2": ["fp8e5m2", torch.float8_e5m2],
#                 "I64": ["i64", torch.int64],
#                 "I32": ["i32", torch.int32],
#                 "I16": ["i16", torch.int16],
#                 "I8": ["i8", torch.int8],
#                 "U8": ["u8", torch.uint8],
#                 "NF4": ["nf4", "nf4"],
#                 "AUTO": ["auto", "auto"]
#         }
#         for key, val in float_chart.items():
#             if old_index is key:
#                 return val[0], val[1]

# ############## SET DEVICE
#     def set_device(self, device=None):
#         self.capacity         = config.get_default("spec","data")
#         self.device = device
#         tf32 = self.capacity.get("allow_tf32",False)
#         fasdp = self.capacity.get("flash_attention",False)
#         mps_as = self.capacity.get("attention_slicing",False)
#         if self.device == "cuda":
#             torch.backends.cudnn.allow_tf32 = tf32
#             torch.backends.cuda.enable_flash_sdp = fasdp
#         elif self.device == "mps":
#                 torch.backends.mps.enable_attention_slicing = mps_as
#         return self.device

# ############## QUEUE
#     def queue_manager(self, prompt, seed, negative_terms=None):
#         self.prompt = prompt
#         self.negative_prompt = negative_terms
#         self.seed = seed
#         return

# ############## MEM USE
#     def metrics(self):
#         if "cuda" in self.device:
#             memory = round(torch.cuda.max_memory_allocated(self.device) * 1e-9, 2)
#             logging.debug(f"vram use: {memory}.", exc_info=True)
#             print(f"RAM use: {memory}.")

# ############## ENCODERS
#     def declare_encoders(self, model_symlinks, tk_expressions, te_expressions):
#         model_class = os.path.basename(model_symlinks)

#         if te_expressions.get("variant",0) != 0:
#             var, dtype = self.float_converter(te_expressions["variant"])
#             te_expressions["variant"] = var
#             te_expressions.setdefault("torch_dtype", dtype)


#         self.hf_log(fatal=True) #suppress layer skip messages

#         if model_class == "CLI-VL":
#             tokenizer = CLIPTokenizer.from_pretrained(
#                 model_symlinks,
#                 **tk_expressions
#                 )

#             text_encoder = CLIPTextModel.from_pretrained(
#                 model_symlinks,
#                 **te_expressions
#             ).to(self.device)

#         elif model_class == "CLI-VG":
#             tokenizer = CLIPTokenizerFast.from_pretrained(
#                 model_symlinks,
#                 **tk_expressions
#                 )

#             text_encoder = CLIPTextModelWithProjection.from_pretrained(
#                 model_symlinks,
#                 **te_expressions
#             ).to(self.device)

#         elif "T5" in model_class:
#             tokenizer = T5Tokenizer.from_pretrained(
#                 model_symlinks,
#                 **tk_expressions
#                 )

#             text_encoder = T5EncoderModel.from_pretrained(
#                 model_symlinks,
#                 **te_expressions
#             ).to(self.device)

#         self.hf_log(on=True) #return to normal

#         return tokenizer, text_encoder

# ############## EMBEDDINGS
#     def generate_embeddings(self, prompts, tokenizers_in, text_encoders_in, conditioning):
#         embeddings_list = []

#         for prompt, tokenizer, text_encoder in zip(prompts, tokenizers_in, text_encoders_in):
#             cond_input = tokenizer(
#             prompt,
#             max_length=tokenizer.model_max_length,
#             **conditioning
#         )
#             prompt_embeds = text_encoder(cond_input.input_ids.to(self.device), output_hidden_states=True)

#             pooled_prompt_embeds = prompt_embeds[0]
#             embeddings_list.append(prompt_embeds.hidden_states[-2])

#             prompt_embeds = torch.concat(embeddings_list, dim=-1)

#         negative_prompt_embeds = torch.zeros_like(prompt_embeds)
#         negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

#         bs_embed, seq_len, _ = prompt_embeds.shape
#         prompt_embeds = prompt_embeds.repeat(1, 1, 1)
#         prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

#         seq_len = negative_prompt_embeds.shape[1]
#         negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
#         negative_prompt_embeds = negative_prompt_embeds.view(1 * 1, seq_len, -1)

#         pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
#         negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)

#         return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

#  ############## VAE PT1
#     def add_vae(self, model, vae_in):
#         if vae_in.get("variant",0) != 0:
#             var, dtype = self.float_converter(vae_in["variant"])
#             vae_in["variant"] = var
#             vae_in.setdefault("torch_dtype", dtype)
#         autoencoder = AutoencoderKL.from_single_file(model,**vae_in).to(self.device)
#         return autoencoder

# ############## PIPE
#     def construct_pipe(self, model, pipe_data):
#         if pipe_data.get("variant",0) != 0:
#             var, dtype = self.float_converter(pipe_data["variant"])
#             pipe_data["variant"] = var
#             pipe_data.setdefault("torch_dtype", dtype)
#         original_config = "/home/maxtretikov/.config/shadowbox/models/metadata/STA-XL/sdxl_base.yaml"
#         pipe = StableDiffusionXLPipeline.from_single_file(model, original_config=original_config, **pipe_data).to(self.device)

#         if self.device == "mps":
#             if self.capacity.get("attention_slicing",False) == True:
#                 pipe.enable_attention_slicing(True)
#         # elif self.capacity.get("xformers",False) == True:
#         #         pipe.set_use_memory_efficient_attention_xformers(True)
#         return pipe

# ############## LORA
#     def add_lora(self, pipe, lora, weight_name, lora_class, unet_only, fuse, scale):
#         self.lora_class = lora_class #add to metadata
#         if unet_only:
#             pipe.unet.load_attn_procs(lora, weight_name=weight_name)
#         else:
#             pipe.load_lora_weights(lora, weight_name=weight_name)
#         if fuse:
#             pipe.fuse_lora(scale=scale)
#         return pipe

# ############## ENCODE
#     def encode_prompt(self, queue, tokenizers, text_encoders, conditioning):
#         with torch.no_grad():
#             for generation in queue:
#                 generation['embeddings'] = self.generate_embeddings(
#                     [generation['prompt'], generation['prompt']],
#                     tokenizers, text_encoders, conditioning,
#                     )
#         self.metrics()
#         for i in tokenizers:
#             del i
#         del tokenizers
#         for i in text_encoders:
#             del i
#         del text_encoders
#         self.cache_jettison()
#         return queue

# ############## COMPILE
#     def compile_model(self, pipe, compile_data):
#         if pipe.transformer is not None: pipe.transformer = torch.compile(pipe.transformer, **compile_data)
#         if pipe.unet is not None: pipe.unet = torch.compile(pipe.unet, **compile_data)
#         if pipe.vae is not None: pipe.vae = torch.compile(pipe.vae, **compile_data)
#         return pipe

# ############## CACHE MANAGEMENT
#     def cache_jettison(self, pipe=None):
#         gc.collect()
#         if self.device == "cuda": torch.cuda.empty_cache()
#         if self.device == "mps": torch.mps.empty_cache()
#         if self.device == "xpu": torch.xpu.empty_cache()
#         if pipe: return pipe

# ############## SCHEDULER
#     def add_scheduler(self, pipe, scheduler_in, scheduler_data):
#         self.scheduler_in = scheduler_in # add to metadata
#         scheduler_data = defaultdict(dict)
#         schedule_chart = {
#             "EulerDiscreteScheduler" : EulerDiscreteScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#             "EulerAncestralDiscreteScheduler" : EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#             "FlowMatchEulerDiscreteScheduler" : FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#             "EDMDPMSolverMultistepScheduler" : EDMDPMSolverMultistepScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#             "DPMSolverMultistepScheduler" : DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#             "DDIMScheduler" : DDIMScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#             "LCMScheduler" : LCMScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#             "TCDScheduler" : TCDScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#             "HeunDiscreteScheduler" : HeunDiscreteScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#             "UniPCMultistepScheduler" : UniPCMultistepScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#             "LMSDiscreteScheduler" : LMSDiscreteScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#             "DEISMultistepScheduler" : DEISMultistepScheduler.from_config(pipe.scheduler.config,**scheduler_data),
#         }

#         if scheduler_in in schedule_chart:
#             pipe.scheduler = schedule_chart[scheduler_in]
#             return pipe
#         else:
#             try:
#                 raise ValueError(f"Scheduler '{scheduler_in}' not supported")
#             except ValueError as error_log:
#                  logging.debug(f"Scheduler error {error_log}.", exc_info=True)


# ############## MEMORY OFFLOADING
#     def offload_to(self, pipe, offload_method):
#         if not "cpu" in self.device:
#             if offload_method == "sequential": pipe.enable_sequential_cpu_offload()
#             elif offload_method == "cpu": pipe.enable_model_cpu_offload()
#         elif offload_method == "disk": accelerate.disk_offload()
#         return pipe

# ############## CFG CUTOFF
#     def _dynamic_guidance(self, pipe, step_index, timestep, callback_key):
#         if step_index is int(pipe.num_timesteps * 0.5):
#             callback_key['prompt_embeds'] = callback_key['prompt_embeds'].chunk(2)[-1]
#             callback_key['add_text_embeds'] = callback_key['add_text_embeds'].chunk(2)[-1]
#             callback_key['add_time_ids'] = callback_key['add_time_ids'].chunk(2)[-1]
#             pipe._guidance_scale = 0.0
#         return callback_key

# ############## INFERENCE
#     def diffuse_latent(self, pipe, queue, gen_data):
#         self.metrics()
#         #apply_hidiffusion(pipe)
#         generator = torch.Generator(device=self.device)
#         for i, generation in enumerate(queue, start=1):
#             seed_planter(generation['seed'])
#             generator.manual_seed(generation['seed'])
#             if generation.get("embeddings",False) is not False:
#                 gen_data["prompt_embeds"] = generation["embeddings"][0]
#                 gen_data["negative_prompt_embeds"] = generation["embeddings"][1]
#                 gen_data["pooled_prompt_embeds"] = generation["embeddings"][2]
#                 gen_data["negative_pooled_prompt_embeds"] = generation["embeddings"][3]
#             if self.capacity.get("dynamo",False) == True:
#                 generation['latents'] = pipe(generator=generator,**gen_data)[0] # return individual for compiled
#             else:
#                 generation['latents'] = pipe(generator=generator,**gen_data).images # return entire batch at once
#                 #  pipe ends with image, but really its a latent...
#         if queue[0].get("embeddings", False) is not False:
#             pipe.unload_lora_weights()
#             del pipe.unet
#             del generator
#             gen_data["prompt_embeds"] = None
#             gen_data["negative_prompt_embeds"] = None
#             gen_data["pooled_prompt_embeds"] = None
#             gen_data["negative_pooled_prompt_embds"] = None
#             pipe = self.cache_jettison(pipe)
#         self.metrics()
#         return pipe, queue

# ############## AUTODECODE
#     def decode_latent(self, pipe, queue, file_prefix, upcast, tile, slicing):
#         if upcast == True: pipe.upcast_vae()
#         if tile == True: pipe.enable_vae_tiling()
#         if slicing == True: pipe.enable_vae_slicing()
#         output_dir = config.get_path("output")
#         counter = [s.endswith('png') for s in output_dir].count(True) # get existing images
#         filename = []
#         #pipe.vae.set_use_memory_efficient_attention_xformers(True)
#         with torch.no_grad():
#             for i, generation in enumerate(queue, start=1):
#                 counter += i
#                 generation['latents'] = generation['latents'].to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

#                 image = pipe.vae.decode(          #latent gets processed here
#                     generation['latents'] / pipe.vae.config.scaling_factor,
#                     return_dict=False,
#                 )[0]

#                 image = pipe.image_processor.postprocess(image, output_type='pil')[0]
#                 filename.append(f"{file_prefix}-{generation['seed']}-{counter}-batch-{i}.png")

#                 image.save(os.path.join(config.get_path("output"), filename[i-1])) # optimize=True,

#         return image
