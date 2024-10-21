import os
from PIL import Image
from torch import imag as img
from collections import defaultdict
from sdbx.config import config
from sdbx.nodes.types import *
from sdbx.nodes.helpers import soft_random, hard_random
import datetime
from time import perf_counter_ns, sleep
from transformers import AutoConfig
from sdbx.config import config

index            = config.model_indexer
optimize         = config.node_tuner
insta            = config.t2i_pipe
capacity         = config.sys_cap
algorithms       = config.get_default("algorithms","schedulers")
solvers          = config.get_default("algorithms","solvers")
metadata         = config.get_path("models.metadata")
model_symlinks   = defaultdict(dict)
queue_data       = []
compile_list     = ["max-autotune","reduce-overhead"]
timestep_list    = ["trailing", "linear"]
offload_list     = ["none","sequential", "cpu", "disk"]
extensions_list  = [".fp16.safetensors",".safetensors"]
variant_list     = ["F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I64", "I32", "I16", "I8", "U8", "nf4", "BOOL"]


# class SD_ASPECT(str, Enum):
#         "1:1___ 512x512"= (512, 512)
#         "4:3___ 682x512"= (682, 512)
#         "3:2___ 768x512"= (768, 512)
#         "16:9__ 910x512"= (910, 512)
#         "1:85:1 952x512"= (952, 512)
#         "2:1__ 1024x512"= (1024, 512)

# class SD21_ASPECT(str,Enum):
#         "1:1_ 768x768"= (768, 768)

# class SVD_ASPECT(str, Enum):
#         "1:1__ 576x576" = (576, 576)
#         "16:9 1024X576"= (1024, 576)

# class XL_ASPECT(str, Enum):
#         "1:1_ 1024x1024"= (1024, 1024)
#         "16:15 1024x960"= (1024, 960)
#         "17:15 1088x960"= (1088, 960)
#         "17:14 1088x896"= (1088, 896)
#         "4:3__ 1152x896"= (1152, 896)
#         "18:13 1152x832"= (1152, 832)
#         "3:2__ 1216x832"= (1216, 832)
#         "5:3__ 1280x768"= (1280, 768)
#         "7:4__ 1344x768"= (1344, 768)
#         "21:11 1344x704"= (1344, 704)
#         "2:1__ 1408x704"= (1408, 704)
#         "23:11 1472x704"= (1472, 704)
#         "21:9_ 1536x640"= (1536, 640)
#         "5:2__ 1600x640"= (1600, 640)
#         "26:9_ 1664x576"= (1664, 576)
#         "3:1__ 1728x576"= (1728, 576)
#         "28:9_ 1792x576"= (1792, 576)
#         "29:8_ 1856x512"= (1856, 512)
#         "15:4_ 1920x512"= (1920, 512)
#         "31:8_ 1984x512"= (1984, 512)
#         "4:1__ 2048x512"= (2048, 512)

def model_store(self, llm=False, diffusion=False, vae=False, text_encoder=False, lora=False):
    if llm         : return config.get_default("index","LLM")
    if diffusion   : return config.get_default("index","DIF")
    if vae         : return config.get_default("index","VAE")
    if text_encoder: return config.get_default("index","TRA")
    if lora        : return config.get_default("index","LOR")

def symlink_prepare(model, folder=None, full_path=False): 
    """Accept model filename only, find class and type, create symlink in metadata subfolder, return correct path to model metadata"""
    model_class = index.fetch_id(model)[1]
    model_type  = index.fetch_id(model)[0]
    text_models = model_store("text_encoder")
    vae_models = model_store("vae")
    diffusion_models = model_store("diffusion")
    if model_type == "TRA":
        path        = text_models[model][model_class][1]
        model_prefix = "model" # node wants to know your model's location
    elif model_type == "VAE":
        path        = vae_models[model][model_class][1]
        model_prefix = "diffusion_pytorch_model"
        full_path=True
    elif model_type == "DIF":
        path        = diffusion_models[model][model_class][1]
        model_prefix = "diffusion_pytorch_model"
    for model_extension in extensions_list:
        file_extension =  model_prefix + model_extension
        if folder is not None: file_extension = os.path.join(folder, file_extension)
        symlink_location = optimize.symlinker(path, model_class, file_extension, full_path=full_path)
    return symlink_location

import os
from llama_cpp import Llama

@node(name="Genesis Node", display=True)
def genesis_node(self,
    user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    model: Literal[*primary_models.keys()] = None, # type: ignore
) -> AutoConfig:
    optimize.determine_tuning(model)
    optimize_expressions = optimize.opt_exp() 
    generator_expressions = optimize.gen_exp(2)#clip skip
    conditional_expressions = optimize.cond_exp()
    pipe_expressions = optimize.pipe_exp()
    vae_expressions = optimize.vae_exp()   
    return (model, user_prompt, optimize_expressions, generator_expressions, conditional_expressions, pipe_expressions, vae_expressions)

@node(path="load/", name="GGUF Loader")
def gguf_loader(self,
    llm     : Literal[*index.model_store("llm").keys()]                  = None, # type: ignore # type: ignore # type: ignore
    gpu_layers: A[int, Slider(min=-1, max=35, step=1)] = -1,
    advanced_options: bool = False,
        threads        : A[int, Dependent(on="advanced_options", when=True), Slider(min=0, max=64, step=1)]       = None,
        max_context    : A[int,Dependent(on="advanced_options", when=True), Slider(min=0, max=32767, step=64)]    = None,
        one_time_seed  : A[bool, Dependent(on="advanced_options", when=True)]                                     = False,
        flash_attention: A[bool, Dependent(on="advanced_options", when=True)]                                     = False,
        device         : A[Literal[*capacity.get("device","cpu")], Dependent(on="advanced_options", when=True)]                      = None, # type: ignore # type: ignore # type: ignore
        batch          : A[int, Dependent(on="advanced_options", when=True), Numerical(min=0, max=512, step=1), ] = 1,
) -> Llama:
    model_class = next(iter(llms[llm])),
    llama_expression = {
        "model_path":  llms[llm][model_class][1],
        'seed': soft_random() if one_time_seed is False else hard_random(),
        "n_gpu_layers": gpu_layers if device != "cpu" else 0,
        }
    if threads         is not None: llama_expression.setdefault("n_threads", threads)
    if max_context     is not None: llama_expression.setdefault("n_ctx", max_context)
    if batch           is not None:  llama_expression.setdefault("n_batch",batch)
    if flash_attention is not None:  llama_expression.setdefault("flash_attn",flash_attention)
    return Llama(**llama_expression)


@node(path="prompt/", name="LLM Prompt")
def llm_prompt(
    llama: Llama,
    system_prompt   : A[str, Text(multiline=True, dynamic_prompts=True)] = "You're a guru for what you know, yet wiser for knowing what you don't.", # my poor attempt at a profound sentiment
    user_prompt     : A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    streaming       : bool                                               = True,
    advanced_options: bool                                               = False,
        top_k         : A[int, Dependent(on="advanced_options", when=True), Slider(min=0, max=100)]                               = 40,
        top_p         : A[float, Dependent(on="advanced_options", when=True), Slider(min=0, max=1, step=0.01, round=0.01)]        = 0.95,
        repeat_penalty: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 1,
        temperature   : A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 0.2,
        max_tokens    :  A[int, Dependent(on="advanced_options", when=True),  Numerical(min=0, max=2)]                            = 256,
) -> str:
    llama_expression = {
        "messages": [
            { 
                "role": "system", 
                "content": system_prompt 
            },
            { 
                "role": "user", 
                "content": user_prompt 
            }
        ]
    }
    if streaming     : llama_expression.setdefault("stream",streaming)
    if repeat_penalty: llama_expression.setdefault("repeat_penalty",repeat_penalty)
    if temperature   : llama_expression.setdefault("temperature",temperature)
    if top_k         : llama_expression.setdefault("top_k",top_k)
    if top_p         : llama_expression.setdefault("top_p",top_p)
    if max_tokens    : llama_expression.setdefault("max_tokens",max_tokens)
    return llama.create_chat_completion(**llama_expression)

@node(path="save/", name="LLM Print")
def llm_print(
    response: A[str, Text()]
) -> I[str]:
    print("Calculating Resposnse")
    for chunk in range(response):
        delta = chunk['choices'][0]['delta']
        # if 'role' in delta:               # this prints assistant: user: etc
            # print(delta['role'], end=': ')
            #yield (delta['role'], ': ')
        if 'content' in delta:              # the response itself
            #print(delta['content'], end='')
            yield delta['content'], ''

from transformers import data, TensorType, Cache, AutoModel
from transformers import models as ModelType

@node(path="prompt/", name="Diffusion Prompt", display=True)
def diffusion_prompt(
    prompt        : A[str,
        Text(multiline=True, dynamic_prompts=True)] = "A slice of a rich and delicious chocolate cake presented on a table in a palace reminiscent of Versailles",
    negative_terms: A[str, Text(multiline=True, dynamic_prompts=True)] = None,
    seed          : A[int, Dependent(on="prompt"),
        Numerical(min=0, max=0xFFFFFFFFFFFFFF, step=1, randomizable=True)] = soft_random(),  # cross compatible with ComfyUI and A1111 seeds
    batch         : A[int, Numerical(min=0, max=512, step=1)]          = 1,
) -> A[list, Name("Queue")]:
    full_prompt = []
    full_prompt.append(prompt)
    if negative_terms is not None:
        full_prompt.append(negative_terms)
    for i in range(batch):
        queue_data.extend([{
            "prompt": full_prompt,
            "seed": seed,
            }])
    return queue_data

@node(path="load/", name="Load Transformers", display=True)
def load_transformer(
    transformer_0     : Literal[*text_models.keys()]                                                    = None, # type: ignore
        transformer_1 : A[Literal[*text_models.keys()], Dependent(on="transformer", when=(not None))]   = None, # type: ignore
        transformer_2 : A[Literal[*text_models.keys()], Dependent(on="transformer_2", when=(not None))] = None, # type: ignore
    precision_0       : Literal[*variant_list]                                                          = "F16", # type: ignore
        precision_1   : A[Literal[*variant_list], Dependent(on="transformer_2", when=(not None))]       = None, # type: ignore
        precision_2   : A[Literal[*variant_list], Dependent(on="transformer_3", when=(not None))]       = None, # type: ignore
    clip_skip         : A[int, Numerical(min=0, max=12, step=1)]                                        = 2,
    device            : A[Literal[*capacity.get("device","cpu")]]                                                          = None, # type: ignore
    flash_attention   : bool                                                                            = False,
    low_cpu_mem_usage : bool                                                                            = False,
) -> A[TensorType, Name("Encoders")]:
    tk_expressions   = defaultdict(dict)
    te_expressions   = defaultdict(dict)
    if device is not None:
        insta.set_device(device)
    num_hidden_layers = 12 - clip_skip
    transformer_list = []
    if transformer_0:
        transformer_list.append(transformer_0)
    if transformer_1:
        transformer_list.append(transformer_1)
    if transformer_2:
        transformer_list.append(transformer_2)

    token_data = []
    text_encoder_data = []

    for i in range(len(transformer_list)):
        te_expressions[i].setdefault("variant", "F16")
        te_expressions[i]["num_hidden_layers"] = num_hidden_layers
        if low_cpu_mem_usage == True:
            te_expressions[i]["low_cpu_mem_usage"] = low_cpu_mem_usage
        te_expressions[i]["use_safetensors"] = True
        te_expressions[i]["local_files_only"] = True
        tk_expressions[i]["local_files_only"] = True
        #expressions[i]["subfolder"] = f"text_encoder_{i + 1}" if i > 0 else "text_encoder"
        #tokenizers[i]["subfolder"]  = f"tokenizer_{i + 1}" if i > 0 else "tokenizer"
        if flash_attention == True: te_expressions[i]["attn_implementation"] = "flash_attention_2"
        model_symlinks["transformer"][i] = symlink_prepare(transformer_list[i])# expressions[i]["subfolder"])
        tk, te = insta.declare_encoders(model_symlinks["transformer"][i], tk_expressions[i], te_expressions[i])
        token_data.insert(i,tk)
        text_encoder_data.insert(i,te)

    return token_data, text_encoder_data

@node(path="transform/", name="Force Device", display=True)
def force_device(
    device_name: Literal[*capacity.get("device","cpu")] = None, # type: ignore
    gpu_id     : A[int, Dependent(on="device"), Slider(min=0, max=100)] = 0,
) ->  A[iter, Name("Device")]:
    device_name = next(iter(capacity.get("device","cpu")),"cuda")
    if gpu_id != 0:
        device_name = f"{device_name}:{gpu_id}"
    insta.set_device(device_name)
    return device_name

@node(path="load/", name="Load Vae Model", display=True)
def load_vae_model(
    vae              : Literal[*vae_models.keys()],                  # type: ignore
    precision        : Literal[*variant_list] = "F16",               # type: ignore
    low_cpu_mem_usage: bool = False,
    slicing          : bool = False,
    tiling           : bool = False,
    device           : A[Literal[*capacity.get("device","cpu")]] = None         # type: ignore
) -> A[TensorType, Name("VAE")]:
    vae_input        = defaultdict(dict)
    if device is not None:
        insta.set_device(device)
    de = ["disable","enable"]
    vae_input[f"{de[slicing]}_slicing"]
    vae_input[f"{de[tiling]}_tiling"]
    vae_input["vae"] = vae
    model_class = index.fetch_id(vae_input["vae"])[1]
    vae_input["config"]           = os.path.join(metadata, model_class, "vae","config.json")
    vae_input["variant"]          = precision
    if low_cpu_mem_usage == True:
        vae_input["low_cpu_mem_usage"] = low_cpu_mem_usage
    vae_input["local_files_only"]  = True
    model_symlinks["vae"] = symlink_prepare(vae, "vae", full_path=True)
    vae_tensor = insta.add_vae(model_symlinks["vae"], vae_input)
    return vae_tensor

@node(path="load/", name="Load Diffusion Pipe", display=True)
def diffusion_pipe(
    vae                 : ModelType                                                                = None,
    model               : Literal[*diffusion_models.keys()]                                        = None,  # type: ignore
    use_model_to_encode : bool                                                                     = False,
    tokenizers          : ModelType                                                                = None,
    text_encoders       : ModelType                                                                = None,
    precision           : Literal[*variant_list]                                                   = "F16", # type: ignore
    device              : Literal[*capacity.get("device","cpu")]                                   = None,  # type: ignore
    low_cpu_mem_usage   : bool                                                                     = False,
    safety_checker      : bool                                                                     = False,
    fuse                : bool                                                                     = False,
    padding             : A[Literal['max_length'], Dependent(on="use_model_to_encode", when=True)] = None,
    truncation          : A[bool, Dependent(on="use_model_to_encode", when=True)]                  = None,
    return_tensors      : A[Literal["pt"],Dependent(on="use_model_to_encode", when=True)]          = None,
    add_watermarker     : bool                                                                     = False,
) -> A[TensorType, Name("Model")]:
    pipe_input       = defaultdict(dict)
    if device is not None:
        insta.set_device(device)

    pipe_input["vae"] = vae
    pipe_input["variant"] = precision

    if low_cpu_mem_usage == True:
        pipe_input["low_cpu_mem_usage"] = low_cpu_mem_usage

    pipe_input["add_watermarker"] = False
    pipe_input["local_files_only"]  = True

    if safety_checker is False:
        pipe_input["safety_checker"] = None

    model_class = index.fetch_id(model)[1]
    model_symlinks["model"] = symlink_prepare(model, "unet")
    model_symlinks["unet"] = os.path.join(model_symlinks["model"], "unet")

    transformer_classes = index.fetch_compatible(model_class)

    for i in range(len(transformer_classes[1])):
        if tokenizers is not None:
            if i > 0: pipe_input[f"tokenizer_{i+1}"]     = model_symlinks["transformer"][i]
            else: pipe_input["tokenizer"]     = model_symlinks["transformer"][i]
        if text_encoders is not None:
                if i > 0: pipe_input[f"text_encoder_{i+1}"] = model_symlinks["transformer"][i]
                else: pipe_input["text_encoder"] = model_symlinks["transformer"][i]
        elif use_model_to_encode is True:
            if i > 0:
                pipe_input[f"text_encoder_{i+1}"] = model_symlinks["model"][i]
                pipe_input[f"tokenizer_{i+1}"]     = model_symlinks["model"][i]
            else:
                pipe_input["text_encoder"] = model_symlinks["model"][i]
                pipe_input["tokenizer"]     = model_symlinks["model"][i]
        else:
            if i > 0:
                pipe_input[f"text_encoder_{i+1}"] = None
                pipe_input[f"tokenizer_{i+1}"]     = None
            else:
                pipe_input["text_encoder"] = None
                pipe_input["tokenizer"]     = None

    pipe = insta.construct_pipe(model_symlinks["model"], pipe_input)
    return pipe

@node(path="load/", name="Load LoRA", display=True)
def load_lora(
    pipe  : ModelType                           = None,
    lora: List[Literal[*lora_models.keys()]]    = None, # type: ignore
    fuse: A[List[bool], EqualLength(to="lora")] = False,
    scale: A[List[A[float, Numerical(min=0.0, max=1.0, step=0.01)]], EqualLength(to="lora")] = 1.0,
) ->  A[ModelType, Name("LoRA")]:
    for i in range(3):
        lora_data = globals().get(f"lora_{i}", None)
        if lora_data is not None:
            weight_name = os.path.basename(lora_data)
            model_class = next(iter(lora_models[weight_name]))
            lora        = lora_data.replace(weight_name,"")
            fuse_data   = locals().get(f"fuse_{i}", False)
            scale_data  = locals().get(f"scale_{i}", None)
            if scale_data is not None:
                lora_scale  = {"lora_scale": scale_data }
            pipe = insta.add_lora(pipe, lora, weight_name, fuse_data, lora_scale)

    return pipe

@node(path="transform/", name="Compile Pipe", display=True)
def compile_pipe(
    pipe      : ModelType             = None,
    fullgraph : bool                  = True,
    mode      :Literal[*compile_list] = "reduce-overhead", # type: ignore
) -> A[TensorType, Name(f"Compiler")]:
    compile_input    = defaultdict(dict)
    compile_input["mode"]      = mode
    compile_input["fullgraph"] = fullgraph
    if capacity.get("dynamo",False) == True:
        pipe = insta.compile_model(compile_input)
    return pipe

@node(path="prompt/", name="Encode Vision Prompt", display=True)
def encode_prompt(
    queue            : tuple                   = None,
    tokenizers_in   : ModelType                = None,
    text_encoders_in: ModelType                = None,
    padding          : Literal['max_length']   = "max_length",
    truncation       : bool                    = True,
    return_tensors   : Literal["pt"]           = 'pt',
    device           : Literal[*capacity.get("device","cpu")]     = None, # type: ignore
) ->  A[TensorType, Name("Embeddings")]:
    if device is not None:
        insta.set_device(device)
    conditioning = {
        "padding"   : padding,
        "truncation": truncation,
        "return_tensors": return_tensors
                    }
    queue = insta.encode_prompt(queue, tokenizers_in, text_encoders_in, conditioning)
    return queue

@node(path="transform/",name="Empty Cache", display=True)
def empty_device_cache(
    pipe              : ModelType = None,
    device            : A[iter,Literal[*capacity.get("device","cpu")]]                          = None,  # type: ignore
) -> TensorType:
    if device is not None:
        insta.set_device(device)
    pipe = insta.cache_jettison()
    return pipe

@node(path="load/", name="Load Noise Scheduler", display=True)
def load_scheduler(
        pipe                   : ModelType                   = None,
        scheduler              : Literal[*algorithms]        = None, # type: ignore
        algorithm_type         : Literal[*solvers]           = None, # type: ignore
        timesteps              : A[str, Text()]              = None,
        sigmas                 : A[str, Text()]              = None,
        interpolation_type     : Literal["linear"]           = None,
        timestep_spacing       : Literal[*timestep_list]= None, # type: ignore
        use_beta_sigmas        : bool                        = True,
        use_karras_sigmas      : bool                        = False,
        ays_schedules          : bool                        = False,
        set_alpha_to_one       : bool                        = False,
        rescale_betas_zero_snr : bool                        = False,
        clip_sample            : bool                        = False,
        use_exponential_sigmas : bool                        = False,
        euler_at_final         : bool                        = False,
        lu_lambdas             : bool                        = False,

) -> A[ModelType, Name("Scheduler")]:
    scheduler_input = defaultdict(dict)
    scheduler_input = {
        "scheduler"             : scheduler,
    }
    data = {
        "algorithm_type"        : algorithm_type,
        "timesteps"             : timesteps,
        "interpolation_type"    : interpolation_type,
        "timestep_spacing"      : timestep_spacing,
        "use_beta_sigmas"       : use_beta_sigmas,
        "use_karras_sigmas"     : use_karras_sigmas,
        "ays_schedules"         : ays_schedules,
        "set_alpha_to_one"      : set_alpha_to_one,
        "rescale_betas_zero_snr": rescale_betas_zero_snr,
        "clip_sample"           : clip_sample,
        "use_exponential_sigmas": use_exponential_sigmas,
        "euler_at_final"        : euler_at_final,
        "lu_lambdas"            : lu_lambdas,
        "sigmas"                : sigmas,
        }

    for name, value in data.items():
        if value != None:
            if value != False: scheduler_input.setdefault(name, value)
    pipe = insta.add_scheduler(pipe, scheduler, scheduler_input)

    return pipe

@node(path="generate/", name="Generate Image", display=True)
def generate_image(
    pipe                : ModelType                                           = None,
    queue               : TensorType                                          = None,
    num_inference_steps : A[int, Numerical(min=0, max=250, step=1)]           = None,
    guidance_scale      : A[float, Numerical(min=0.00, max=50.00, step=0.01)] = 5.0,
    eta                 : A[float, Numerical(min=0.00, max=1.00, step=0.01)]  = None,
    dynamic_guidance    : bool                                                = False,
    precision           : Literal[*variant_list]                              = "F16",  # type: ignore
    device              : Literal[*capacity.get("device","cpu")]              = None,   # type: ignore
    offload_method      : Literal[*offload_list]                              = "none", # type: ignore
    output_type         : Literal["latent"]                                   = "latent",
) -> Tuple[A[TensorType, Name("Pipe")], A[TensorType, Name("Queue")]]:
    if device is not None:
        insta.set_device(device)
    gen_input        = defaultdict(dict)
    gen_input["num_inference_steps"] = num_inference_steps
    gen_input["guidance_scale"] = guidance_scale
    if eta is not None: gen_input["eta"] = eta
    gen_input["variant"] = precision
    gen_input["output_type"] = output_type
    if offload_method != "none":
        pipe = insta.offload_to(pipe, offload_method)
    if capacity.get("dynamo",False) != False:
        gen_input["return_dict"]   = False
    if dynamic_guidance is True:
        gen_input["callback_on_step_end"] = insta._dynamic_guidance
        gen_input["callback_on_step_end_tensor_inputs"] = ['prompt_embeds', 'add_text_embeds','add_time_ids']
    pipe, latent = insta.diffuse_latent(pipe, queue, gen_input)
    return pipe, latent

@node(path="save/", name="Autodecode/Save/Preview", display=True)
def autodecode(
    pipe           : ModelType                                                                                                 = None,
    queue          : TensorType                                                                                                = None,
    upcast         : bool                                                                                                      = True,
    file_prefix    : A[str, Text(multiline=False, dynamic_prompts=True)]                                                       = "Shadowbox",
    device         : Literal[*capacity.get("device","cpu")]                                                                    = None,        # type: ignore
    file_format    : A[Literal["png", "jpg", "optimize"], Dependent(on="temp", when=False)]                                    = "optimize",
    compress_level: A[int, Slider(min=1, max=9, step=1), Dependent(on="format", when=(not "optimize"))]                        = 7,
    temp          : bool                                                                                                       = False,
) -> I[Any]:
    print("decoding")
    if device is not None:
        insta.set_device(device)
    image = insta.decode_latent(pipe, queue, upcast, file_prefix)
    for img in range(image):
        yield img

@node(path="generate/", name="Timecode", display=True)
def tc() -> I[Any]:
    if locals(clock) is not None: clock = perf_counter_ns() # 00:00:00
    tc = f"[ {str(datetime.timedelta(milliseconds=(((perf_counter_ns()-clock)*1e-6))))[:-2]} ]"
    yield tc, sleep(0.6)

@node(path="load/", name="Load Refiner", display=True)
def load_refiner(
    pipe                : TensorType                                            = None,
    high_aesthetic_score: A[int, Numerical(min=0.00, max=10, step=0.01)]        = 7,
    low_aesthetic_score  : A[int, Numerical(min=0.00, max=10, step=0.01)]       = 5,
    padding              : A[float, Numerical(min=0.00, max=511.00, step=0.01)] = 0.0,
    device: Literal[*capacity.get("device","cpu")] = None,         # type: ignore
)-> A[ModelType, Name("Refiner")]:
    if device is not None:
        insta.set_device(device)
    # todo: do refiner stuff refiner
   # return refiner