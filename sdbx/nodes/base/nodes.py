# import os
# from PIL import Image
# from collections import defaultdict
# from sdbx.config import config
# from sdbx.nodes.types import *
# from sdbx.nodes.helpers import soft_random, hard_random
# import datetime
# from time import perf_counter_ns, sleep
# from transformers import AutoConfig
# from sdbx.config import config
# from diffusers.schedulers import AysSchedules

# index            = config.model_indexer
# optimize         = config.node_tuner
# insta            = config.t2i_pipe
# capacity         = config.get_default("spec","data")
# algorithms       = config.get_default("algorithms","schedulers")
# solvers          = config.get_default("algorithms","solvers")
# metadata         = config.get_path("models.metadata")
# model_symlinks   = defaultdict(dict)
# queue_data       = []
# attn_list        = ["none","sdpa","flash_attention_2","flash_attention","xformers"]
# compile_list     = ["max-autotune","reduce-overhead"]
# timestep_list    = ["trailing", "linear"]
# offload_list     = ["none","sequential", "cpu", "disk"]
# extensions_list  = [".fp16.safetensors",".safetensors"]
# variant_list     = ["F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I64", "I32", "I16", "I8", "U8", "nf4", "BOOL"]


# # class SD_ASPECT(str, Enum):
# #         "1:1___ 512x512"= (512, 512)
# #         "4:3___ 682x512"= (682, 512)
# #         "3:2___ 768x512"= (768, 512)
# #         "16:9__ 910x512"= (910, 512)
# #         "1:85:1 952x512"= (952, 512)
# #         "2:1__ 1024x512"= (1024, 512)

# # class SD21_ASPECT(str,Enum):
# #         "1:1_ 768x768"= (768, 768)

# # class SVD_ASPECT(str, Enum):
# #         "1:1__ 576x576" = (576, 576)
# #         "16:9 1024X576"= (1024, 576)

# # class XL_ASPECT(str, Enum):
# #         "1:1_ 1024x1024"= (1024, 1024)
# #         "16:15 1024x960"= (1024, 960)
# #         "17:15 1088x960"= (1088, 960)
# #         "17:14 1088x896"= (1088, 896)
# #         "4:3__ 1152x896"= (1152, 896)
# #         "18:13 1152x832"= (1152, 832)
# #         "3:2__ 1216x832"= (1216, 832)
# #         "5:3__ 1280x768"= (1280, 768)
# #         "7:4__ 1344x768"= (1344, 768)
# #         "21:11 1344x704"= (1344, 704)
# #         "2:1__ 1408x704"= (1408, 704)
# #         "23:11 1472x704"= (1472, 704)
# #         "21:9_ 1536x640"= (1536, 640)
# #         "5:2__ 1600x640"= (1600, 640)
# #         "26:9_ 1664x576"= (1664, 576)
# #         "3:1__ 1728x576"= (1728, 576)
# #         "28:9_ 1792x576"= (1792, 576)
# #         "29:8_ 1856x512"= (1856, 512)
# #         "15:4_ 1920x512"= (1920, 512)
# #         "31:8_ 1984x512"= (1984, 512)
# #         "4:1__ 2048x512"= (2048, 512)

# def model_store(model_1, model_2=None):
#     if model_2 == None:
#         data = config.get_default("index",model_1)
#         return data.keys()
#     else:
#         primary = config.get_default("index",model_1) | config.get_default("index",model_2)
#         return primary.keys()


# def symlink_prepare(model, folder=None, full_path=False):
#     """Accept model filename only, find class and type, create symlink in metadata subfolder, return correct path to model metadata"""
#     model_category  = index.fetch_id(model)[0]
#     model_class = index.fetch_id(model)[1]
#     model_path = index.fetch_id(model)[2][1] # node wants to know your model's location
#     model_prefix = "model" if model_category == "TRA" else "diffusion_pytorch_model" #
#     for model_extension in extensions_list:
#         file_extension =  model_prefix + model_extension
#         if folder is not None: file_extension = os.path.join(folder, file_extension)
#         symlink_location = optimize.symlinker(model_path, model_class, file_extension, full_path=full_path)
#     return symlink_location

# import os
# from llama_cpp import Llama

# @node(name="Genesis Node", display=True)
# def genesis_node(self,
#     user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
#     model: Literal[*model_store("DIF","LLM")] = None, # type: ignore
# ) -> AutoConfig:
#     defaults = optimize.determine_tuning(model)
#     return defaults

# @node(path="load/", name="GGUF Loader")
# def gguf_loader(self,
#     llm     : Literal[*model_store("LLM")]             = None, # type: ignore # type: ignore # type: ignore
#     gpu_layers: A[int, Slider(min=-1, max=35, step=1)] = -1,
#     advanced_options: bool = False,
#         threads        : A[int, Dependent(on="advanced_options", when=True), Slider(min=0, max=64, step=1)]       = None,
#         max_context    : A[int,Dependent(on="advanced_options", when=True), Slider(min=0, max=32767, step=64)]    = None,
#         one_time_seed  : A[bool, Dependent(on="advanced_options", when=True)]                                     = False,
#         flash_attention: A[bool, Dependent(on="advanced_options", when=True)]                                     = False,
#         batch          : A[int, Dependent(on="advanced_options", when=True), Numerical(min=0, max=512, step=1), ] = 1,
#         device         : Literal[*capacity.get("devices","cpu")]                                                   = None, # type: ignore # type: ignore # type: ignore
# ) -> Llama:
#     model_path = index.fetch_id(llm)[2][1]
#     llama_expression = {
#         "model_path": model_path,
#         'seed': soft_random() if one_time_seed is False else hard_random(),
#         "n_gpu_layers": gpu_layers if device != "cpu" else 0,
#         }
#     if threads         is not None: llama_expression.setdefault("n_threads", threads)
#     if max_context     is not None: llama_expression.setdefault("n_ctx", max_context)
#     if batch           is not None:  llama_expression.setdefault("n_batch",batch)
#     if flash_attention is not None:  llama_expression.setdefault("flash_attn",flash_attention)
#     return Llama(**llama_expression)


# @node(path="prompt/", name="LLM Prompt")
# def llm_prompt(
#     llama: Llama,
#     system_prompt   : A[str, Text(multiline=True, dynamic_prompts=True)] = "You're a guru for what you know, yet wiser for knowing what you don't.", # my poor attempt at a profound sentiment
#     user_prompt     : A[str, Text(multiline=True, dynamic_prompts=True)] = "",
#     streaming       : bool                                               = True,
#     advanced_options: bool                                               = False,
#         top_k         : A[int, Dependent(on="advanced_options", when=True), Slider(min=0, max=100)]                               = 40,
#         top_p         : A[float, Dependent(on="advanced_options", when=True), Slider(min=0, max=1, step=0.01, round=0.01)]        = 0.95,
#         repeat_penalty: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 1,
#         temperature   : A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 0.2,
#         max_tokens    :  A[int, Dependent(on="advanced_options", when=True),  Numerical(min=0, max=1024)]                            = 256,
# ) -> str:
#     llama_expression = {
#         "messages": [
#             {
#                 "role": "system",
#                 "content": system_prompt
#             },
#             {
#                 "role": "user",
#                 "content": user_prompt
#             }
#         ]
#     }
#     if streaming     : llama_expression.setdefault("stream",streaming)
#     if repeat_penalty: llama_expression.setdefault("repeat_penalty",repeat_penalty)
#     if temperature   : llama_expression.setdefault("temperature",temperature)
#     if top_k         : llama_expression.setdefault("top_k",top_k)
#     if top_p         : llama_expression.setdefault("top_p",top_p)
#     if max_tokens    : llama_expression.setdefault("max_tokens",max_tokens)
#     return llama.create_chat_completion(**llama_expression)

# @node(path="save/", name="LLM Print")
# def llm_print(
#     response: A[str, Text()]
# ) -> I[str]:
#     print("Calculating Resposnse")
#     for chunk in range(response):
#         delta = chunk['choices'][0]['delta']
#         # if 'role' in delta:               # this prints assistant: user: etc
#             # print(delta['role'], end=': ')
#             #yield (delta['role'], ': ')
#         if 'content' in delta:              # the response itself
#             #print(delta['content'], end='')
#             yield delta['content'], ''

# from transformers import data, TensorType, Cache, AutoModel
# from transformers import models as ModelType

# @node(path="prompt/", name="Diffusion Prompt", display=True)
# def diffusion_prompt(
#     prompt         : A[str, Text(multiline=True, dynamic_prompts=True)]                                                = "A slice of a rich and delicious chocolate cake presented on a table in a palace reminiscent of Versailles",
#     negative_terms: A[str, Text(multiline=True, dynamic_prompts=True)]                                                 = None,
#     seed           : A[int, Dependent(on="prompt"), Numerical(min=0, max=0xFFFFFFFFFFFFFF, step=1, randomizable=True)] = None,                                                                                                        # cross compatible with ComfyUI and A1111 seeds
#     batch          : A[int, Numerical(min=0, max=512, step=1)]                                                         = 1,
# ) -> A[str, Name("Queue")]:
#     full_prompt = []
#     #if pony == True # todo: active pony detection
#     prompt = f"score_9, score_8_up, {prompt}"
#     full_prompt.append(prompt)
#     for i in range(batch):
#         seed = soft_random()
#         queue_data.extend([{
#             "prompt": full_prompt,
#             "seed": seed,
#             }])
#     if negative_terms is not None:
#         queue_data.extend([{"negative_prompt": negative_terms,}])
#     insta.queue_manager(prompt, negative_terms, seed)
#     return queue_data

# @node(path="load/", name="Load Transformers", display=True)
# def load_transformer(
#     transformer_models : List[Literal[*model_store("TRA")]]        = None,  # type: ignore
#     precisions          : List[Literal[*variant_list]]             = "F16", # type: ignore
#     #clip_skip          : A[int, Numerical(min=0, max=12, step=1)] = 2,
#     device              : Literal[*capacity.get("devices","cpu")]  = None,  # type: ignore
#     attn_implementation : List[Literal[*attn_list]]                = False,
#     low_cpu_mem_usage   : bool                                     = False,
#     safety_checker      : bool                                     = False,
#     add_watermarker     : bool                                     = False,
# ) -> (A[ModelType,Name("Tokenizer")], A[ModelType,Name("Text_Encoder")]):
#     tk_expressions   = defaultdict(dict)
#     te_expressions   = defaultdict(dict)
#     if device is not None:
#         insta.set_device(device)
#     #num_hidden_layers = 12 - (clip_skip - 1)
#     token_data = []
#     text_encoder_data = []

#     for i in range(len(transformer_models)):
#         te_expressions[i].setdefault("variant", "F16") #precisions[i]
#         #te_expressions[i]["num_hidden_layers"] = num_hidden_layers
#         if low_cpu_mem_usage == True:
#             te_expressions[i]["low_cpu_mem_usage"] = low_cpu_mem_usage
#         te_expressions[i]["use_safetensors"] = True
#         te_expressions[i]["local_files_only"] = True
#         tk_expressions[i]["local_files_only"] = True
#         if safety_checker == False: tk_expressions[i]["safety_checker"] = None
#         tk_expressions[i]["add_watermarker"] = add_watermarker
#         #expressions[i]["subfolder"] = f"text_encoder_{i + 1}" if i > 0 else "text_encoder"  may need for later cases of model symlinking
#         #tokenizers[i]["subfolder"]  = f"tokenizer_{i + 1}" if i > 0 else "tokenizer"
#         if attn_implementation != False: te_expressions[i]["attn_implementation"] = attn_implementation
#         model_symlinks["transformer"][i] = symlink_prepare(transformer_models[i])# expressions[i]["subfolder"])
#         tk, te = insta.declare_encoders(model_symlinks["transformer"][i], tk_expressions[i], te_expressions[i])
#         token_data.insert(i,tk)
#         text_encoder_data.insert(i,te)

#     return token_data, text_encoder_data

# @node(path="transform/", name="Force Device", display=True)
# def force_device(
#     device_name: Literal[*capacity.get("devices","cpu")] = None, # type: ignore
# ) ->  A[iter, Name("Device")]:
#     device_name = next(iter(capacity.get("devices","cpu")),"cuda")
#     insta.set_device(device_name)
#     return device_name

# @node(path="load/", name="Load Vae Model", display=True)
# def load_vae_model(
#     vae              : Literal[*model_store("VAE")],                    # type: ignore
#     precision        : Literal[*variant_list] = "F16",                  # type: ignore
#     low_cpu_mem_usage: bool = False,
#     device           : Literal[*capacity.get("devices","cpu")] = None,  # type: ignore
#     config           : Literal[*os.listdir(metadata)]          = None,

# ) -> A[TensorType, Name("VAE")]:
#     vae_input        = defaultdict(dict)
#     if device is not None:
#         insta.set_device(device)
#     vae_input["vae"] = vae
#     vae_class = index.fetch_id(vae)[1]
#     vae_input["config"] = os.path.join(metadata, "STA-XL", "vae","config.json") #config
#     vae_input["variant"] = precision
#     if low_cpu_mem_usage == True:
#         vae_input["low_cpu_mem_usage"] = low_cpu_mem_usage
#     vae_input["local_files_only"]  = True
#     model_symlinks["vae"] = symlink_prepare(vae, "vae", full_path=True)
#     vae_tensor = insta.add_vae(model_symlinks["vae"], vae_input)
#     return vae_tensor

# @node(path="load/", name="Load Diffusion Pipe", display=True)
# def diffusion_pipe(
#     vae                 : ModelType                                                                           = None,
#     model               : Literal[*model_store("DIF")]                                                        = None,  # type: ignore
#     use_model_to_encode : bool                                                                                = False,
#     precision           : Literal[*variant_list]                                                              = "F16", # type: ignore
#     tokenizers          : ModelType                                                                           = None,
#     text_encoders       : ModelType                                                                           = None,
#     device              : Literal[*capacity.get("devices","cpu")]                                             = None,  # type: ignore
#     low_cpu_mem_usage   : bool                                                                                = False,
#     fuse_unet           : bool                                                                                = False,
#     padding             : A[Literal['max_length'], Dependent(on="use_model_to_encode", when=Condition(True))] = None,
#     truncation          : A[bool, Dependent(on="use_model_to_encode", when=True)]                             = None,
#     return_tensors      : A[Literal["pt"],Dependent(on="use_model_to_encode", when=True)]                     = None,
#     safety_checker      : bool                                                                                = False,
#     add_watermarker     : bool                                                                                = False,
#     original_config     : Literal[*os.listdir(metadata)]                                                      = None,
# ) -> A[TensorType, Name("Model")]:
#     pipe_input       = defaultdict(dict)
#     if device is not None:
#         insta.set_device(device)
#     pipe_input["vae"] = vae
#     pipe_input["variant"] = precision

#     if low_cpu_mem_usage == True: pipe_input["low_cpu_mem_usage"] = low_cpu_mem_usage
#     if safety_checker == False: pipe_input["safety_checker"] = None
#     pipe_input["add_watermarker"] = add_watermarker
#     pipe_input["local_files_only"]  = True
#     pipe_input["original_config"] = original_config

#     model_class = index.fetch_id(model)[1]
#     model_symlinks["model"] = symlink_prepare(model, "unet", full_path=True)
#     model_symlinks["unet"] = os.path.join(model_symlinks["model"], "unet")

#     transformer_classes = index.fetch_compatible(model_class)

#     for i in range(len(transformer_classes[1])):
#         if tokenizers is not None:
#             if i > 0: pipe_input[f"tokenizer_{i+1}"]     = model_symlinks["transformer"][i]
#             else: pipe_input["tokenizer"]     = model_symlinks["transformer"][i]
#         if text_encoders is not None:
#                 if i > 0: pipe_input[f"text_encoder_{i+1}"] = model_symlinks["transformer"][i]
#                 else: pipe_input["text_encoder"] = model_symlinks["transformer"][i]
#         elif use_model_to_encode is True:
#             if i > 0:
#                 pipe_input[f"text_encoder_{i+1}"] = model_symlinks["model"][i]
#                 pipe_input[f"tokenizer_{i+1}"]    = model_symlinks["model"][i]
#             else:
#                 pipe_input["text_encoder"] = model_symlinks["model"][i]
#                 pipe_input["tokenizer"]    = model_symlinks["model"][i]
#         else:
#             if i > 0:
#                 pipe_input[f"text_encoder_{i+1}"] = None
#                 pipe_input[f"tokenizer_{i+1}"]    = None
#             else:
#                 pipe_input["text_encoder"] = None
#                 pipe_input["tokenizer"]    = None
#     model_path = index.fetch_id(model)[2][1]
#     pipe = insta.construct_pipe(model_path, pipe_input) #model_symlinks["model"]
#     return pipe

# @node(path="load/", name="Load LoRA", display=True)
# def load_lora(
#     pipe  : ModelType                                              = None,
#     lora  : List[Literal[*model_store("LOR")]]                     = None,
#     fuse  : A[List[bool], EqualLength(to="lora")]                  = False,
#     scale : A[float, Numerical(min=0.0, max=1.0, step=0.01)]       = 1.0,
#     unet_only : A[List[bool], EqualLength(to="lora")]              = False, #make unet only dependent
# ) ->  A[ModelType, Name("LoRA")]:
#     lora_class = []
#     for i in range(len(lora)):
#         weight_name = os.path.basename(lora[i])
#         lora_class.append(index.fetch_id(weight_name)[1]) # metadata
#         new_lora        = lora[i].replace(weight_name,"")
#         values = {"class": lora_class}
#         try: values.setdefault("unet_only", unet_only[i])
#         except: pass
#         try: values.setdefault("fuse", fuse[i])
#         except: pass
#         try:values.setdefault("scale", scale[i])
#         except: pass
#         pipe = insta.add_lora(pipe, new_lora, weight_name, lora_class, unet_only, fuse[i], scale)
#     return pipe

# @node(path="transform/", name="Compile Pipe", display=True)
# def compile_pipe(
#     pipe      : ModelType             = None,
#     fullgraph : bool                  = True,
#     mode      :Literal[*compile_list] = "reduce-overhead", # type: ignore
# ) -> A[TensorType, Name(f"Compiler")]:
#     compile_input    = defaultdict(dict)
#     compile_input["mode"]      = mode
#     compile_input["fullgraph"] = fullgraph
#     if capacity.get("dynamo",False) == True:
#         pipe = insta.compile_model(compile_input)
#     return pipe

# @node(path="prompt/", name="Encode Vision Prompt", display=True)
# def encode_prompt(
#     queue            : str                                     = None,
#     tokenizers_in    : ModelType                               = None,
#     text_encoders_in: ModelType                                = None,
#     padding          : Literal['max_length']                   = "max_length",
#     truncation       : bool                                    = True,
#     return_tensors   : Literal["pt"]                           = 'pt',
#     device           : Literal[*capacity.get("devices","cpu")] = None,         # type: ignore
# ) ->  A[TensorType, Name("Embeddings")]:
#     if device is not None:
#         insta.set_device(device)
#     conditioning = {
#         "padding"   : padding,
#         "truncation": truncation,
#         "return_tensors": return_tensors
#                     }
#     queue = insta.encode_prompt(queue, tokenizers_in, text_encoders_in, conditioning)
#     return queue

# @node(path="transform/",name="Empty Cache", display=True)
# def empty_device_cache(
#     pipe   : ModelType                               = None,
#     device : Literal[*capacity.get("devices","cpu")] = None, # type: ignore
# ) -> TensorType:
#     if device is not None:
#         insta.set_device(device)
#     pipe = insta.cache_jettison()
#     return pipe

# @node(path="load/", name="Load Noise Scheduler", display=True)
# def load_scheduler(
#         pipe                   : ModelType                   = None,
#         scheduler              : Literal[*algorithms]        = None, # type: ignore
#         algorithm_type         : Literal[*solvers]           = None, # type: ignore
#         timesteps              : A[str, Text()]              = None,
#         sigmas                 : A[str, Text()]              = None,
#         interpolation_type     : Literal["linear"]           = None, #todo: turn into constant
#         timestep_spacing       : Literal[*timestep_list]     = None, # type: ignore #todo: turn into constant
#         use_beta_sigmas        : bool                        = False,
#         use_karras_sigmas      : bool                        = False,
#         ays_schedules          : bool                        = False,
#         set_alpha_to_one       : bool                        = False,
#         rescale_betas_zero_snr : bool                        = False,
#         clip_sample            : bool                        = False,
#         use_exponential_sigmas : bool                        = False,
#         euler_at_final         : bool                        = False,
#         lu_lambdas             : bool                        = False,
#         beta_schedule          : Literal["scaled_linear"]    = None,  #todo: turn into constant
# ) -> A[ModelType, Name("Scheduler")]:
#     scheduler_data = defaultdict(dict)
#     #scheduler = "DPMSolverMultistepScheduler" #overriding,

#     data = {
#         "algorithm_type"        : algorithm_type, #algorithm_type,
#         "timesteps"             : timesteps, #timesteps,
#         "interpolation_type"    : interpolation_type,
#         "timestep_spacing"      : timestep_spacing,
#         "use_beta_sigmas"       : use_beta_sigmas,
#         "use_karras_sigmas"     : use_karras_sigmas, #use_karras_sigmas,
#         "ays_schedules"         : ays_schedules,
#         "set_alpha_to_one"      : set_alpha_to_one,
#         "rescale_betas_zero_snr": rescale_betas_zero_snr,
#         "clip_sample"           : clip_sample,
#         "use_exponential_sigmas": use_exponential_sigmas,
#         "euler_at_final"        : euler_at_final,
#         "lu_lambdas"            : lu_lambdas,
#         "sigmas"                : sigmas,
#         }

#     for name, value in data.items():
#         if value != None:
#             if value != False:
#                 scheduler_data.setdefault(name,value)
#     pipe = insta.add_scheduler(pipe, scheduler, scheduler_data)

#     return pipe

# @node(path="generate/", name="Generate Image", display=True)
# def generate_image(
#     pipe                : ModelType                                           = None,
#     queue               : TensorType                                          = None,
#     num_inference_steps : A[int, Numerical(min=0, max=250, step=1)]           = None,
#     guidance_scale      : A[float, Numerical(min=0.00, max=50.00, step=0.01)] = 5.0,
#     width               : A[int, Numerical(min=0, max=16384, step=1)]         = 832,
#     height              : A[int, Numerical(min=0, max=16384, step=1)]         = 1216,
#     eta                 : A[float, Numerical(min=0.00, max=1.00, step=0.01)]  = 0.3,
#     dynamic_guidance    : bool                                                = False,
#     precision           : Literal[*variant_list]                              = "F16",  # type: ignore
#     device              : Literal[*capacity.get("devices","cpu")]             = None,   # type: ignore
#     offload_method      : Literal[*offload_list]                              = "none", # type: ignore
#     output_type         : Literal["latent"]                                   = "latent",
#     safety_checker      : bool                                                = False,
#     add_watermarker     : bool                                                = False,
# ) -> (A[TensorType, Name("Pipe")], A[TensorType, Name("Queue")]):
#     if device is not None:
#         insta.set_device(device)
#     gen_input                        = defaultdict(dict)
#     if safety_checker == False: gen_input["safety_checker"] = None
#     gen_input["add_watermarker"] = add_watermarker
#     gen_input["width"] = width
#     gen_input["height"] = height
#     gen_input["num_inference_steps"] = num_inference_steps
#     gen_input["guidance_scale"]      = guidance_scale
#     if eta is not None: gen_input["eta"] = eta
#     gen_input["variant"]     = precision
#     gen_input["output_type"] = output_type
#     if offload_method != "none":
#         pipe = insta.offload_to(pipe, offload_method)
#     if capacity.get("dynamo",False) != False:
#         gen_input["return_dict"]   = False
#     if dynamic_guidance is True:
#         gen_input["callback_on_step_end"]               = insta._dynamic_guidance
#         gen_input["callback_on_step_end_tensor_inputs"] = ['prompt_embeds', 'add_text_embeds','add_time_ids']
#     pipe, latent = insta.diffuse_latent(pipe, queue, gen_input)
#     return pipe, latent

# @node(path="save/", name="Autodecode/Save/Preview", display=True)
# def autodecode(
#     pipe            : ModelType                                                                           = None,
#     latent          : TensorType                                                                          = None,
#     file_prefix     : A[str, Text(multiline=False, dynamic_prompts=True)]                                 = "Shadowbox",
#     device          : Literal[*capacity.get("devices","cpu")]                                             = None,        # type: ignore
#     upcast          : bool                                                                                = True,
#     slicing         : bool                                                                                = False,
#     tiling          : bool                                                                                = False,
#     file_format     : A[Literal["png", "jpg", "optimize"], Dependent(on="temp", when=False)]              = "optimize",
#     compress_level  : A[int, Slider(min=1, max=9, step=1), Dependent(on="format", when=(not "optimize"))] = 7,
#     temporary_only  : bool                                                                                = False,
# ) -> Image:
#     print("Decoding...")
#     decode_args = { "upcast": upcast, "slicing": slicing, "tile": tiling}
#     if device is not None:
#         insta.set_device(device)
#     image = insta.decode_latent(pipe, latent, file_prefix, **decode_args)
#     return image
#     #I[Any]:re
#     # for img in range(image):
#     #     yield img

# @node(path="generate/", name="Timecode", display=True)
# def tc() -> I[Any]:
#     if locals(clock) is not None: clock = perf_counter_ns() # 00:00:00
#     tc = f"[ {str(datetime.timedelta(milliseconds=(((perf_counter_ns()-clock)*1e-6))))[:-2]} ]"
#     print(tc)
#     sleep(0.6)
#     #yield tc, sleep(0.6)

# @node(path="load/", name="Load Refiner", display=True)
# def load_refiner(
#     pipe                : TensorType                                            = None,
#     high_aesthetic_score: A[int, Numerical(min=0.00, max=10, step=0.01)]        = 7,
#     low_aesthetic_score  : A[int, Numerical(min=0.00, max=10, step=0.01)]       = 5,
#     padding              : A[float, Numerical(min=0.00, max=511.00, step=0.01)] = 0.0,
#     device: Literal[*capacity.get("devices","cpu")] = None,         # type: ignore
# )-> A[ModelType, Name("Refiner")]:
#     if device is not None:
#         insta.set_device(device)
#     # todo: do refiner stuff refiner
#    # return refiner