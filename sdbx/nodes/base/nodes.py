import os
import PIL
from PIL import Image
from collections import defaultdict
from sdbx.config import config, TensorDataType
from sdbx.nodes.types import *
from sdbx.nodes.helpers import soft_random, hard_random
import datetime
from time import perf_counter_ns, sleep
from transformers import AutoConfig



llms             = config.get_default("index","LLM")
diffusion_models = config.get_default("index","DIF")

primary_models   = llms | diffusion_models

index            = config.model_indexer
optimize         = config.node_tuner
insta            = config.t2i_pipe

system           = config.get_default("spec","data") #needs to be set by system @ launch
spec             = system.get("devices","cpu")
flash_attn       = system.get("flash_attention",False)

expressions      = defaultdict(dict)
model_symlinks   = defaultdict(dict)
tokenizers       = defaultdict(dict)
transform_models = config.get_default("index","TRA")
text_models      = llms | transform_models
extensions_list  = [".fp16.safetensors",".safetensors"]
variant_list     = ["F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I64", "I32", "I16", "I8", "U8", "nf4", "BOOL"]#
lora_models      = config.get_default("index","LOR")#
vae_data         = defaultdict(dict)
vae_models       = config.get_default("index","VAE")
metadata         = config.get_path("models.metadata")#
pipe_data        = defaultdict(dict)#
dynamo           = system.get("dynamo",0)
compile_data     = defaultdict(dict)
compile_list     = ["max-autotune","reduce-overhead"]#
algorithms       = config.get_default("algorithms","schedulers")
solvers          = config.get_default("algorithms","solvers")
timestep_list    = ["trailing", "linear"]
scheduler_data   = defaultdict(dict)

gen_data         = defaultdict(dict)
offload_list     = ["none","sequential", "cpu", "disk"]




def symlink_prepare(model, folder=None, full_path=False): 
    """Accept model filename only, find class and type, create symlink in metadata subfolder, return correct path to model metadata"""
    model_class = index.fetch_id(model)[1]
    model_type  = index.fetch_id(model)[0]
    path        = text_models[model][model_class][1]  # node wants to know your model's location
    for model_extension in extensions_list:
        if model_type == "TRA":
            file_extension = os.path.join("model",model_extension)
        else: 
            file_extension = os.path.join("diffusion_pytorch_model",model_extension)
        if folder is not None: file_extension = os.path.join(folder, file_extension)
        symlink_location = optimize.symlinker(path, model_class, file_extension, full_path=full_path)
    return symlink_location

#             # model_class = index.fetch_id(transformers[i])
#             # path        = text_models[transformers[i]][model_class][1] # node wants to know your model's location
#             # for model_extension in extensions_list:
#             #     tra_extension            = os.path.join("model",model_extension)
#             #     model_symlinks["tra"][i] = optimize.symlinker(path, model_class, tra_extension)

import os
from llama_cpp import Llama

@node(name="Genesis Node", display=True)
def genesis_node(
     user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
     model: Literal[*primary_models.keys()] = next(iter(primary_models.keys()),""), # type: ignore
 ) -> AutoConfig:
    optimize.determine_tuning(model)
    optimize_expressions = optimize.opt_exp() 
    generator_expressions = optimize.gen_exp(2)#clip skip
    conditional_expressions = optimize.cond_exp()
    pipe_expressions = optimize.pipe_exp()
    vae_expressions = optimize.vae_exp()   
    return (optimize_expressions, generator_expressions, conditional_expressions, pipe_expressions)

@node(path="load/", name="GGUF Loader")
def gguf_loader(
    llm     : Literal[*llms.keys()]                  = next(iter(llms.keys()),""), # type: ignore # type: ignore # type: ignore
    gpu_layers: A[int, Slider(min=-1, max=35, step=1)] = -1,
    advanced_options: bool = False,
        threads        : A[int, Dependent(on="advanced_options", when=True), Slider(min=0, max=64, step=1)]       = None,
        max_context    : A[int,Dependent(on="advanced_options", when=True), Slider(min=0, max=32767, step=64)]    = None,
        one_time_seed  : A[bool, Dependent(on="advanced_options", when=True)]                                     = False,
        flash_attention: A[bool, Dependent(on="advanced_options", when=True)]                                     = False,
        device         : A[iter,Literal[*spec], Dependent(on="advanced_options", when=True)]                      = next(iter(spec), "cpu"), # type: ignore # type: ignore # type: ignore
        batch          : A[int, Dependent(on="advanced_options", when=True), Numerical(min=0, max=512, step=1), ] = 1,
) -> Llama: 
    llama_expression = {
        "model_path": os.path.join(config.get_path("models.llms"), llm),
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
    system_prompt   : A[str, Text(multiline=True, dynamic_prompts=True)] = "You're a guru for revealing what you know, yet wiser for revealing what you do not.", # my poor attempt at a profound sentiment
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

@node(path="prompt/", name="Text Input", display=True)
def text_input(
    prompt        : A[str, Text(multiline=True, dynamic_prompts=True)] = "A slice of a rich and delicious chocolate cake presented on a table in a palace reminiscent of Versailles",
    negative_terms: A[str, Text(multiline=True, dynamic_prompts=True)] = None,
    seed          : A[int, Dependent(on="prompt", when=(not None)), Numerical(min=0, max=0xFFFFFFFFFFFFFF, step=1, randomizable=True)] = soft_random(),  # cross compatible with ComfyUI and A1111 seeds
    batch         : A[int, Numerical(min=0, max=512, step=1)]          = 1,
) -> A[tuple, Name("Queue")]:
    prompt = [prompt, negative_terms]
    for i in range(batch):
        queue = insta.queue_manager(prompt, seed)
    return queue

@node(path="load/", name="Load Transformers", display=True)
def load_transformer(
    transformer_0     : Literal[*text_models.keys()]                                                    = next(iter(text_models.keys()),""), # type: ignore
        transformer_1 : A[Literal[*text_models.keys()], Dependent(on="transformer", when=(not None))]   = None, # type: ignore
        transformer_2 : A[Literal[*text_models.keys()], Dependent(on="transformer_2", when=(not None))] = None, # type: ignore
    precision_0       : Literal[*variant_list]                                                          = "F16", # type: ignore
        precision_1   : A[Literal[*variant_list], Dependent(on="transformer_2", when=(not None))]       = None, # type: ignore
        precision_2   : A[Literal[*variant_list], Dependent(on="transformer_3", when=(not None))]       = None, # type: ignore
    clip_skip         : A[int, Numerical(min=0, max=12, step=1)]                                        = 2,
    device            : A[iter,Literal[*spec]]                                                          = next(iter(spec), "cpu"), # type: ignore
    flash_attention   : bool                                                                            = flash_attn,
    low_cpu_mem_usage : bool                                                                            = True,
) -> A[ModelType, Name("Encoders")]:
    insta.set_device(device)
    num_hidden_layers = 12 - clip_skip
    transformer_list = []
    for i in range(3):
        if globals().get(f"transformer_{i}",None) is not None:
            transformer_list.append(globals().get(f"transformer_{i}")) 
    
    for i in range(len(transformer_list)):
        expressions[i]["variant"].append(globals().get(f"precision_{i}",None))
        expressions[i]["subfolder"] = f"text_encoder_{i + 1}" if i > 0 else "text_encoder"
        expressions[i]["num_hidden_layers"] = num_hidden_layers
        expressions[i]["low_cpu_mem_usage"] = low_cpu_mem_usage
        tokenizers[i]["subfolder"]  = f"tokenizer_{i + 1}" if i > 0 else "tokenizer"
        if flash_attention is True: expressions[i]["attn_implementation"] = "flash_attention_2"
        model_symlinks["transformer"][i] = symlink_prepare(transformer_list[i], expressions[i]["subfolder"])

    tokenizer, text_encoder = insta.declare_encoders(model_symlinks["transformer"], tokenizers, expressions)
    return (tokenizer, text_encoder)

@node(path="load/", name="Load LoRA", display=True)
def load_lora(
    pipe  : ModelType                     = None,
    lora_0 : Literal[*lora_models.keys()] = next(iter(lora_models.keys()),None), # type: ignore
        fuse_0  : A[bool,  Dependent(on="lora_0", when=(not None))]                                   = False,
        scale_0 : A[float, Numerical(min=0.0, max=1.0, step=0.01), Dependent(on="fuse_0", when=True)] = None,
        lora_1  : A[Literal[*lora_models.keys()],  Dependent(on="lora_0", when=(not None))]           = None,  # type: ignore
        fuse_1  : A[bool,  Dependent(on="lora_1", when=(not None))]                                   = False,
        scale_1 : A[float, Numerical(min=0.0, max=1.0, step=0.01), Dependent(on="fuse_1", when=True)] = None,
        lora_2  : A[Literal[*lora_models.keys()],  Dependent(on="lora_1", when=(not None))]           = None,  # type: ignore
        fuse_2  : A[bool,  Dependent(on="lora_2", when=(not None))]                                   = False,
        scale_2 : A[float, Numerical(min=0.0, max=1.0, step=0.01), Dependent(on="fuse_2", when=True)] = None,
) ->  A[ModelType, Name("LoRA")]:
    for i in range(3):
        lora_data = globals().get(f"lora_{i}", None)
        if lora_data is not None:
            weight_name = lora_data
            model_class = next(iter(lora_models[lora_data]))
            lora_path   = lora_models[lora_data][model_class][1]
            lora        = lora_path.replace(lora_data,"")
            fuse_data   = globals().get(f"fuse_{i}", False)
            scale_data  = globals().get(f"scale_{i}", None)
            if scale_data is not None: 
                lora_scale  = {"lora_scale": scale_data }
            lora_tensor = insta.add_lora(lora, weight_name, fuse_data, lora_scale)
    return lora_tensor

@node(path="transform/", name="Force Device", display=True)
def force_device(
    device_name: Literal[*spec] = next(iter(spec), "cpu"), # type: ignore
    gpu_id     : A[int, Dependent(on="device", when=not None), Slider(min=0, max=100)] = 0,
) ->  A[iter, Name("Device")]:
    if device_name != "cpu": device = f"{device_name}:{gpu_id}"
    return device

@node(path="load/", name="Load Vae Model", display=True)
def load_vae_model(
    vae       : Literal[*vae_models.keys()] = next(iter(vae_models.keys()),""), # type: ignore
    device    : A[iter,Literal[*spec]]      = next(iter(spec), "cpu"),           # type: ignore
    precision : Literal[*variant_list]      = "F16",                             # type: ignore
    low_cpu_mem_usage : bool = True,
    slicing   : bool = False,
    tiling    : bool = False,
) -> A[ModelType, Name("VAE")]:
    insta.set_device(device)
    de = ["disable","enable"]
    vae_data[f"{de[slicing]}_slicing"]
    vae_data[f"{de[tiling]}_tiling"]
    model_class = index.fetch_id(vae)[1]
    vae_data["config"]           = os.path.join(metadata,  model_class, "vae","config.json")
    vae_data["local_files_only"] = True
    vae_data["variant"]          = precision
    vae_data["low_cpu_mem_usage"] = low_cpu_mem_usage
    model_symlinks["vae"] = symlink_prepare(vae, "vae")
    vae_tensor = insta.add_vae(vae, data)
    return vae_tensor

@node(path="load/", name="Load Diffusion Pipe", display=True)
def diffusion_pipe(
    model                 : Literal[*diffusion_models.keys()]                             = next(iter(diffusion_models.keys()),""), # type: ignore
    use_model_to_encode   : bool                                                          = False,
    transformers          : A[ModelType, Dependent(on="use_model_to_encode", when=False)] = None,
    vae                   : ModelType                                                     = None,
    precision             : Literal[*variant_list]                                        = "F16",                                  # type: ignore
    device                : A[iter,Literal[*spec]]                                                = next(iter(spec), "cpu"),                # type: ignore
    low_cpu_mem_usage     : bool                                                          = True,
    safety_checker        : bool                                                          = False,
    mps_attention_slicing : bool                                                          = False,
) -> A[ModelType, Name("Model")]:
    insta.set_device(device)

    pipe_data["vae"] = vae
    pipe_data["variant"] = precision
    pipe_data["low_cpu_mem_usage"] = low_cpu_mem_usage
    if safety_checker is False:
        pipe_data["safety_checker"] = None
    model_class = index.fetch_id(model)[1]
    model_symlinks["model"] = symlink_prepare(model, "unet")
    model_symlinks["unet"] = os.path.join(model_symlinks["model"], "unet")

    transformer_classes = index.fetch_compatible(model_class)

    for i in range(len(transformer_classes.keys())):
        if use_model_to_encode is True:
            if i > 0:
                pipe_data[f"token_encoder_{i}"] = model_symlinks["model"][i]
                pipe_data[f"tokenizer_{i}"]     = model_symlinks["model"][i]
            else:
                pipe_data["token_encoder"] = model_symlinks["model"][i]
                pipe_data["tokenizer"]     = model_symlinks["model"][i]                
        elif transformers is not None:
                if i > 0:
                    pipe_data[f"token_encoder_{i}"] = model_symlinks["transformer"][i] 
                    pipe_data[f"tokenizer_{i}"]     = model_symlinks["transformer"][i]
                else:
                    pipe_data["token_encoder"] = model_symlinks["transformer"][i]
                    pipe_data["tokenizer"]     = model_symlinks["transformer"][i]
        else:
            if i > 0:
                pipe_data[f"token_encoder_{i}"] = None 
                pipe_data[f"tokenizer_{i}"]     = None 
            else:
                pipe_data["token_encoder"] = None
                pipe_data["tokenizer"]     = None
    pipe = insta.construct_pipe(model_symlinks["model"], mps_attention_slicing, pipe_data)
    return pipe

@node(path="transform/", name="Compile Pipe", display=True)
def compile_pipe(
    pipe      : ModelType             = None,
    fullgraph : bool                  = True,
    mode      :Literal[*compile_list] = "reduce-overhead", # type: ignore
) -> A[ModelType, Name(f"Compiler")]:
    compile_data["mode"]      = mode
    compile_data["fullgraph"] = fullgraph
    if dynamo == True:
        tensor = insta.compile_model(compile_data)
    return tensor

@node(path="prompt/", name="Encode Vision Prompt", display=True)
def encode_prompt(
    transformers  : ModelType                                          = None,
    queue         : tuple                                              = None,
    padding       : Literal['max_length']                              = "max_length",
    truncation    : bool                                               = True,
    return_tensors: Literal["pt"]                                      = 'pt',
) ->  A[TensorType,Name("Embeddings")]:

    conditioning = {
        "padding"   : padding,
        "truncation": truncation,
         "return_tensors": return_tensors
                    }
    queue = insta.encode_prompt(queue, transformers, conditioning)
    return queue

@node(path="transform/",name="Empty Cache", display=True)
def empty_cache(
    data: TensorType = None,
    element_type : Literal["encoder","lora", "unet","vae"] = "encoder"
    ) -> None: 
    insta.cache_jettison(element_type)

@node(path="transform/", name="Noise Scheduler", display=True)
def noise_scheduler(
        scheduler              : Literal[*algorithms]        = next(iter(algorithms),""), # type: ignore
        algorithm_type         : Literal[*solvers]           = next(iter(solvers),""), # type: ignore
        timesteps              : A[str, Text()]              = None,
        interpolation_type     : Literal["linear"]           = None,
        timestep_spacing       : Literal[*timestep_list]     = None, # type: ignore
        use_beta_sigmas        : bool                        = True,
        use_karras_sigmas      : bool                        = False,
        ays_schedules          : bool                        = False,
        set_alpha_to_one       : bool                        = False,
        rescale_betas_zero_snr : bool                        = False,
        clip_sample            : bool                        = False,
        use_exponential_sigmas : bool                        = False,
        euler_at_final         : bool                        = False,
        lu_lambdas             : bool                        = False,

) -> A[tuple, Name("Scheduler")]:
    scheduler_data = {
        "scheduler"             : scheduler,
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

    }
    return (scheduler, scheduler_data)

@node(path="generate/", name="Generate Image", display=True)
def generate_image(
    pipe                : ModelType                                                         = None,
    queue               : tuple                                                             = None,
    encoding            : TensorType                                                        = None,
    scheduler           : tuple                                                             = None,
    num_inference_steps : A[int, Numerical(min=0, max=250, step=1)]                         = 10,
    guidance_scale      : A[float, Numerical(min=0.00, max=50.00, step=0.01)]               = 5,
    eta                 : A[float, Numerical(min=0.00, max=1.00, step=0.01)]                = 5,
    dynamic_guidance    : bool                                                              = False,
    precision           : Literal[*variant_list]                                            = "F16", # type: ignore
    device              : A[iter,Literal[*spec]]                                            = next(iter(spec), "cpu"), # type: ignore
    offload_method      : Literal[*offload_list]                                            = "none", # type: ignore
    output_type         : Literal["latent"]                                                 = "latent"
) -> A[TensorType, Name("Latent")]:
    insta.set_device(device)
    gen_data["num_inference_steps"] = num_inference_steps
    gen_data["guidance_scale"] = guidance_scale
    gen_data["eta"] = eta
    gen_data["variant"] = precision
    gen_data["output_type"] = output_type
    if offload_method != "none": 
        pipe = insta.offload_to(offload_method)
    if dynamo != False:
        gen_data["return_dict"]   = False
    if dynamic_guidance is True: 
        gen_data["callback_on_step_end"] = insta._dynamic_guidance
        gen_data["callback_on_step_end_tensor_inputs"] = ['prompt_embeds', 'add_text_embeds','add_time_ids']
    latent,pipe_out = insta.diffuse_latent(pipe, queue, gen_data, scheduler[0], scheduler[1], encoding)
    return latent, pipe_out

@node(path="save/", name="Autodecode/Save/Preview", display=True)
def autodecode(
    pipe    : TensorType = None,
    upcast  : bool      = True,
    file_prefix:  A[str, Text(multiline=False, dynamic_prompts=True)] = "Shadowbox-",
    format: A[Literal["png","jpg","optimize"], Dependent(on="temp", when="False")] = "optimize",
    compress_level: A[int, Slider(min=1, max=9, step=1),  Dependent(on="format", when=(not "optimize"))] = 7,
    temp: bool = False,
) -> I[Any]:
    file_prefix = os.path.join(config.get_path("output"), file_prefix)
    queue = insta.decode_latent(pipe, upcast)
    for image in range(queue):
        yield image

@node(path="generate/", name="Timecode", display=True)
def tc() -> I[Any]:
    clock = perf_counter_ns() # 00:00:00
    tc = f"[ {str(datetime.timedelta(milliseconds=(((perf_counter_ns()-clock)*1e-6))))[:-2]} ]"
    yield tc, sleep(0.6)


# self.pipe_element = self.class_converter(model)
#             # elif device is "mps":
#             #     pipe.enable_attention_slicing()
