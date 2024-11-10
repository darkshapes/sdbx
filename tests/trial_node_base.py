# import os
# from sdbx.nodes.helpers import soft_random, seed_planter
# from sdbx.config import config

# import torch
# from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, AutoencoderKL, UNet2DConditionModel #EulerAncestralDiscreteScheduler StableDiffusionXLPipeline
# from diffusers.schedulers import AysSchedules
# from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# from sdbx.config import config
# from sdbx.nodes.base import nodes
# from sdbx.indexer import IndexManager

# optimize         = config.node_tuner
# defaults = optimize.determine_tuning("sdxlbase.diffusion_pytorch_model.fp16.safetensors")
# queue = nodes.text_input(**defaults.get("text_input"))
# device = "cuda"

# import os

# # # print("\nInitializing model index, checking system specs.\n  Please wait...")
# # # create_index = IndexManager().write_index()      # (defaults to config/index.json)
# # # dif_index = config.get_default("index", "DIF")
# # # print(f"Ready.")
# # name_path = "sdxlbase.diffusion_pytorch_model.fp16.safetensors" #input("""
# # # Please type the filename of an available checkpoint.
# # # Path will be detected.
# # # (default:diffusion_pytorch_model.fp16.safetensors):""" or "diffusion_pytorch_model.fp16.safetensors")


# # name_path = os.path.basename(name_path)
# # spec = config.get_default("index","DIF")
# # name_path = name_path.strip()
# # name_path = os.path.basename(name_path)
# # if ".safetensors" not in name_path:
# #      name_path = name_path + ".safetensors"
# # for key,val in spec.items():
# #     if name_path in key:
# #         model = key
# #         pass


# # device = nodes.force_device(**defaults.get("force_device"))
# # if defaults["empty_cache"]["stage"].get("head", None) != None:
# #     nodes.empty_cache(**defaults["empty_cache"]["stage"].get("head"))


# # queue = nodes.encode_prompt(queue=queue, transformer_models=transformer_models, **defaults.get("encode_prompt"))
# # if defaults["empty_cache"]["stage"].get("encoder", None) != None:
# #     nodes.empty_cache(queue, defaults["empty_cache"]["stage"].get("encoder"))
# # pipe = nodes.diffusion_pipe(vae, **defaults.get("diffusion_pipe"))

# # pipe = nodes.generate_image(pipe, queue, scheduler, **defaults.get("generate_image"))
# # if defaults["empty_cache"]["stage"].get("generate", None) != None:
# #     nodes.empty_cache(pipe, defaults["empty_cache"]["stage"].get("generate"))
# # image = nodes.autodecode(pipe, **defaults.get("autodecode"))

# #if defaults["empty_cache"]["stage"].get("tail", None) != None:
# #       nodes.empty_cache(image, **defaults["empty_cache"]["stage"]["tail"])
# def encode_prompt(prompts, transformers_models, text_encoders):
#     tokenizers, text_encoders = transformers_models
#     embeddings_list = []
#     for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
#         cond_input = tokenizer(
#         prompt,
#         max_length=tokenizer.model_max_length,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt',
#     )

#         prompt_embeds = text_encoder(cond_input.input_ids.to("cuda"), output_hidden_states=True)

#         pooled_prompt_embeds = prompt_embeds[0]
#         embeddings_list.append(prompt_embeds.hidden_states[-2])

#         prompt_embeds = torch.concat(embeddings_list, dim=-1)

#     negative_prompt_embeds = torch.zeros_like(prompt_embeds)
#     negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

#     bs_embed, seq_len, _ = prompt_embeds.shape
#     prompt_embeds = prompt_embeds.repeat(1, 1, 1)
#     prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

#     seq_len = negative_prompt_embeds.shape[1]
#     negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
#     negative_prompt_embeds = negative_prompt_embeds.view(1 * 1, seq_len, -1)

#     pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
#     negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)

#     return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
# model = "C:\\Users\\Public\\models\\metadata\\STA-XL"
# clip = "C:\\Users\\Public\\models\\metadata\\CLI-VL"
# clip2 = "C:\\Users\\Public\\models\\metadata\\CLI-VG"

# transformers_data = nodes.load_transformer(**defaults.get("load_transformer"))


# # tokenizer = CLIPTokenizer.from_pretrained(
# #     clip,
# #     local_files_only=True,
# # )

# # text_encoder = CLIPTextModel.from_pretrained(
# #     clip,
# #     use_safetensors=True,
# #     torch_dtype=torch.float16,
# #     variant='fp16',
# #     local_files_only=True,
# # ).to(device)

# # tokenizer_2 = CLIPTokenizer.from_pretrained(
# #     clip2,
# #     local_files_only=True,
# # )

# # text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
# #     clip2,
# #     use_safetensors=True,
# #     torch_dtype=torch.float16,
# #     variant='fp16',
# #     local_files_only=True,
# # ).to(device)

# conditioning = {
#     "padding"   : "max_length",
#     "truncation": True,
#     "return_tensors": 'pt'
#                 }

# with torch.no_grad():
#     for generation in queue:
#         generation['embeddings'] = encode_prompt(
#             [generation['prompt'], generation['prompt']],
#             transformers_data, conditioning
#             )

# del transformers_data

# torch.cuda.empty_cache()
# max_memory = round(torch.cuda.max_memory_allocated(device=device) / 1e9, 2)
# print('Max. memory used:', max_memory, 'GB')

# vae_file = "C:\\Users\\Public\\models\\image\\flatpiecexlVAE_baseonA1579.safetensors"
# config_file = "C:\\Users\\Public\\models\\metadata\\STA-XL\\vae\\config.json"
# # vae = AutoencoderKL.from_single_file(vae_file, config=config_file, local_files_only=True,  torch_dtype=torch.float16, variant="fp16").to("cuda")
# vae   = nodes.load_vae_model(**defaults.get("load_vae_model"))
# pipe = nodes.diffusion_pipe(vae, **defaults.get("diffusion_pipe"))
# #ays = AysSchedules["StableDiffusionXLTimesteps"]

# lora = nodes.load_lora(pipe, **defaults.get("load_lora"))
# scheduler = nodes.load_scheduler(pipe, **defaults.get("noise_scheduler"))

# # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")
# pipe.scheduler = scheduler
# pipe.enable_sequential_cpu_offload()

# generator = torch.Generator(device=device)

# for i, generation in enumerate(queue, start=1):
#     seed_planter(generation['seed'])
#     generator.manual_seed(generation['seed'])

#     generation['latents'] = pipe(
#         prompt_embeds=generation['embeddings'][0],
#         negative_prompt_embeds =generation['embeddings'][1],
#         pooled_prompt_embeds=generation['embeddings'][2],
#         negative_pooled_prompt_embeds=generation['embeddings'][3],
#         num_inference_steps=2,
#         #timesteps=ays,
#         guidance_scale=0,
#         generator=generator,
#         output_type='latent',
#     ).images

# torch.cuda.empty_cache()

# pipe.upcast_vae()
# output_dir = config.get_path("output")
# with torch.no_grad():
#     counter = [s.endswith('png') for s in output_dir].count(True) # get existing images
#     for i, generation in enumerate(queue, start=1):
#         generation['latents'] = generation['latents'].to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

#         image = pipe.vae.decode(
#             generation['latents'] / pipe.vae.config.scaling_factor,
#             return_dict=False,
#         )[0]

#         image = pipe.image_processor.postprocess(image, output_type='pil')[0]

#         counter += 1
#         filename = f"Shadowbox-{counter}-batch-{i}.png"

#         image.save(os.path.join(config.get_path("output"), filename)) # optimize=True,


# torch.cuda.empty_cache()
# max_memory = round(torch.cuda.max_memory_allocated(device='cuda') / 1e9, 2)
# print('Max. memory used:', max_memory, 'GB')
