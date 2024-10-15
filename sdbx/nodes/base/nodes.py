import os
import PIL
from PIL import Image
from collections import defaultdict
from sdbx.config import config
from sdbx.nodes.types import *
from sdbx.nodes.helpers import soft_random, hard_random
import datetime
from time import perf_counter_ns, sleep

system           = config.get_default("spec","data") #needs to be set by system @ launch
spec             = system.get("devices","cpu")
flash_attn       = system.get("flash_attention",False)
dynamo           = system.get("dynamo",False)
algorithms       = config.get_default("algorithms","schedulers")
solvers          = config.get_default("algorithms","solvers")
llms             = config.get_default("index","LLM")
diffusion_models = config.get_default("index","DIF")
lora_models      = config.get_default("index","LOR")
vae_models       = config.get_default("index","VAE")
transformers     = config.get_default("index","TRA")
metadata         = config.get_path("models.metadata")
extensions_list  = [".fp16.safetensors",".safetensors"]
variant_list     = config.TensorDataType.TYPE_T
offload_list     = ["none","sequential", "cpu", "disk"]
compile_list     = ["max-autotune","reduce-overhead"]
timestep_list    = ["trailing", "linear"]
transformers     = []
model_symlinks   = defaultdict(dict)
tokenizers       = defaultdict(dict)
expressions      = defaultdict(dict)
vae              = defaultdict(dict)
pipe_data        = defaultdict(dict)
compile_data     = defaultdict(dict)
gen_data         = defaultdict(dict)
scheduler_data   = defaultdict(dict)
primary_models   = llms | diffusion_models
text_models      = llms | transformers
index            = config.model_indexer
optimize         = config.node_tuner
insta            = config.t2i_pipe

# pipe.transformer.to(memory_format=torch.channels_last)
# pipe.vae.to(memory_format=torch.channels_last)


import os
from llama_cpp import Llama

@node(name="Genesis Node", display=True)
def genesis_node(
     user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
     model: Literal[*primary_models.keys()] = next(iter(primary_models.keys()),""),
 ) -> tuple:
    optimize.determine_tuning(model)
    optimize_expressions = optimize.opt_exp() 
    generator_expressions = optimize.gen_exp(2)#clip skip
    conditional_expressions = optimize.cond_exp()
    pipe_expressions = optimize.pipe_exp()
    vae_expressions = optimize.vae_exp()   
    return optimize_expressions, generator_expressions, conditional_expressions, pipe_expressions

@node(name="GGUF Loader")
def gguf_loader(
    model     : Literal[*llms.keys()]                  = next(iter(llms.keys()),""),
    gpu_layers: A[int, Slider(min=-1, max=35, step=1)] = -1,
    advanced_options: bool = False,
        threads        : A[int, Dependent(on="advanced_options", when=True), Slider(min=0, max=64, step=1)]       = None,
        max_context    : A[int,Dependent(on="advanced_options", when=True), Slider(min=0, max=32767, step=64)]    = None,
        one_time_seed  : A[bool, Dependent(on="advanced_options", when=True)]                                     = False,
        flash_attention: A[bool, Dependent(on="advanced_options", when=True)]                                     = False,
        device         : A[Literal[*spec], Dependent(on="advanced_options", when=True)]                           = next(iter(spec), "cpu"),
        batch          : A[int, Dependent(on="advanced_options", when=True), Numerical(min=0, max=512, step=1), ] = 1,
) -> Llama: 
    llama_expression = {
        "model_path": os.path.join(config.get_path("models.llms"), model),
        'seed': soft_random() if one_time_seed is False else hard_random(),
        "n_gpu_layers": gpu_layers if device != "cpu" else 0,
        }
    if threads         is not None: llama_expression.setdefault("n_threads", threads)
    if max_context     is not None: llama_expression.setdefault("n_ctx", max_context)
    if batch           is not None:  llama_expression.setdefault("n_batch",batch)
    if flash_attention is not None:  llama_expression.setdefault("flash_attn",flash_attention)
    return Llama(**llama_expression)


@node(name="LLM Prompt")
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

@node(name="LLM Print")
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

from transformers import data, TensorType, Cache
from transformers import models as ModelType

@node(name="Text Input", display=True)
def text_input(
    prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
) -> A[str, Name("Text")]:
    return prompt

@node(name="Load Vision Models", display=True)
def load_vision_models(
    transformer_0     : Literal[*text_models.keys()]                                                    = next(iter(text_models.keys()),""),
        transformer_1 : A[Literal[*text_models.keys()], Dependent(on="transformer", when=(not None))]   = None,
        transformer_2 : A[Literal[*text_models.keys()], Dependent(on="transformer_2", when=(not None))] = None,
    precision_0       : Literal[*variant_list]                                                          = "F16",
        precision_1   : A[Literal[*variant_list], Dependent(on="transformer_2", when=(not None))]       = None,
        precision_2   : A[Literal[*variant_list], Dependent(on="transformer_3", when=(not None))]       = None,
    clip_skip         : A[int, Numerical(min=0, max=12, step=1)]                                        = 2,
    device            : Literal[*spec]                                                                  = next(iter(spec), "cpu"),
    flash_attention   : bool                                                                            = flash_attn,
    low_cpu_mem_usage : bool                                                                            = True,
) -> A[ModelType, Name("Encoders")]:
    insta.set_device(device)
    num_hidden_layers = 12 - clip_skip

    for i in range(3):
        transformer_data = globals().get(f"transformer_{i}", None)
        if transformer_data is not None:
            transformers.append(globals().get(f"transformer_{i}"))

    for i in range(len(transformers)):
        expressions[i]["variant"].append(globals().get(f"precision_{i}",None))
        expressions[i]["subfolder"] = f"text_encoder_{i + 1}" if i > 0 else "text_encoder"
        expressions[i]["num_hidden_layers"] = num_hidden_layers
        expressions[i]["low_cpu_mem_usage"] = low_cpu_mem_usage
        tokenizers[i]["subfolder"]  = f"tokenizer_{i + 1}" if i > 0 else "tokenizer"
        if flash_attention is True: expressions[i]["attn_implementation"] = "flash_attention_2"
        model_class = index.fetch_id(transformers[i])
        path        = text_models[transformers[i]][model_class][1] # node wants to know your model's location
        for model_extension in extensions_list:
            tra_extension            = os.path.join("model",model_extension)
            model_symlinks["tra"][i] = optimize.symlinker(path, model_class, tra_extension)

    tensor = insta.declare_encoders(model_symlinks["tra"], tokenizers, expressions)
    return tensor

@node(name="Load LoRA", display=True)
def load_lora(
    lora_0  : Literal[*lora_models.keys()]                                                           = next(iter(lora_models.keys()),None),
        fuse_0  : A[bool,  Dependent(on="lora_0", when=(not None))]                                  = False,
        scale_0: A[float, Numerical(min=0.0, max=1.0, step=0.01), Dependent(on="fuse_0", when=True)] = None,
        lora_1  : A[Literal[*lora_models.keys()],  Dependent(on="lora_0", when=(not None))]          = None,
        fuse_1  : A[bool,  Dependent(on="lora_1", when=(not None))]                                  = False,
        scale_1: A[float, Numerical(min=0.0, max=1.0, step=0.01), Dependent(on="fuse_1", when=True)] = None,
        lora_2  : A[Literal[*lora_models.keys()],  Dependent(on="lora_1", when=(not None))]          = None,
        fuse_2  : A[bool,  Dependent(on="lora_2", when=(not None))]                                  = False,
        scale_2: A[float, Numerical(min=0.0, max=1.0, step=0.01), Dependent(on="fuse_2", when=True)] = None,
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
            tensor = insta.add_lora(lora, weight_name, fuse_data, lora_scale)
    return tensor

@node(name="Force Device", display=True)
def force_device(
    device_name: A[str, Literal[*system]] = next(iter(system), "cpu"),
        gpu_id     : A[int, Dependent(on="device", when=(not "cpu")), Slider(min=0, max=100)] = 0,
) ->  A[str, Name("Device")]:
    if device_name != "cpu": device = f"{device_name}:{gpu_id}"
    return device

@node(name="Load Vae Model", display=True)
def load_vae_model(
    model     : Literal[*vae_models.keys()] = next(iter(vae_models.keys()),""),
    device    : Literal[*spec]              = next(iter(spec), "cpu"),
    precision : Literal[*variant_list]      = "F16",
    low_cpu_mem_usage : bool = True,
) -> A[ModelType, Name("VAE")]:
    insta.set_device(device)
    model_class = index.fetch_id(model)[1]
    vae["config"]           = os.path.join(metadata,  model_class, "vae","config.json")
    vae["local_files_only"] = True
    vae["variant"]          = precision
    vae["low_cpu_mem_usage"] = low_cpu_mem_usage
    tensor = insta.add_vae(model, vae, device)
    return tensor

@node(name="Load Diffusion Model", display=True)
def load_diffusion_model(
    model            : Literal[*diffusion_models.keys()] = next(iter(diffusion_models.keys()),""),
    vae              : ModelType              = None,
    encoders         : ModelType              = None,
    precision        : Literal[*variant_list] = "F16",
    device           : Literal[*spec]         = next(iter(spec), "cpu"),
    low_cpu_mem_usage: bool                   = True,
    safety_checker   : bool                   = False,
) -> A[ModelType, Name("Model")]:
    pipe_data["variant"] = precision
    if safety_checker is False:
        pipe_data["safety_checker"] = None
    if encoders != None:
        for i in encoders:
            pipe_data[f"token_encoder_{i}"] = model_symlinks["tra"][i] if i > 0 else pipe_data["token_encoder"] = model_symlinks["tra"][i]
            pipe_data[f"tokenizer_{i}"]     = model_symlinks["tra"][i] if i > 0 else pipe_data["tokenizer"]     = model_symlinks["tra"][i]
    else:
        pipe_data[f"token_encoder_{i}"] = None if i > 0 else pipe_data["token_encoder"] = None
        pipe_data[f"tokenizer_{i}"]     = None if i > 0 else pipe_data["tokenizer"]     = None
    tensor = insta.construct_pipe(model, pipe_data, device)
    return tensor

@node(name="Compile Model", display=True)
def compile_model(
    model_encoder_or_vae : ModelType                                         = None,
    fullgraph : A[bool,  Dependent(on=compile, when=True)]                   = True,
    mode      : A[Literal[*compile_list],  Dependent(on=compile, when=True)] = "reduce-overhead",
) -> A[ModelType, Name(f"Compiler")]:
    model_class               = index.fetch_id(model_encoder_or_vae)[1]
    compile_data["mode"]      = mode
    compile_data["fullgraph"] = fullgraph
    if spec.get("dynamo",0) != 0:
        tensor = insta.compile_model(model_class, compile_data)
    return tensor

@node(name="Encode Prompt", display=True)
def encode_prompt(
    text_encoders : ModelType                                          = None,
    lora          : ModelType                                          = None,
    prompt        : A[str, Text(multiline=True, dynamic_prompts=True)] = "A slice of a rich and delicious chocolate cake presented on a table in a palace reminiscent of Versailles",
    seed          : A[int, Numerical(min=0, max=0xFFFFFFFFFFFFFF, step=1, randomizable=True)] = soft_random(), # cross compatible with ComfyUI and A1111 seeds
    batch         : A[int, Numerical(min=0, max=512, step=1)]          = 1,
    padding       : Literal['max_length']                              = "max_length",
    truncation    : bool                                               = True,
    return_tensors: Literal["pt"]                                      = 'pt',
) ->  A[TensorType,Name("Embeddings")]:
    queue = insta.queue_manager(prompt,seed)
    cache_tuple = (text_encoders)
    conditioning = {
        "padding"   : padding,
        "truncation": truncation,
         "return_tensors": return_tensors
                    }
    encodings = insta.encode_prompt(text_encoders, cache_tuple)
    return encodings


#if self.opt_exp["flash_attention_2"] is True: self.pipe.enable_xformers_memory_efficient_attention()
# self.pipe_element = self.class_converter(model)
# self.pipe_element.to(memory_format=torch.channels_last)
            # elif device is "mps":
            #     pipe.enable_attention_slicing()

@node(name="Empty Cache", display=True)
def empty_cache(
    cache: Cache,
    ) -> None: 
    insta.cache_jettison()

@node(name="Noise Scheduler", display=True)
def noise_scheduler(
        scheduler              : Literal[*algorithms]        = next(iter(algorithms),""),
        algorithm_type         : Literal[*solvers]           = next(iter(solvers),""),
        use_beta_sigmas        : bool                        = True,
        use_karras_sigmas      : bool                        = False,
        interpolation_type     : Literal["linear"]           = None,
        timestep_spacing       : Literal["none", "trailing"] = None,
        ays_schedules          : bool                        = False,
        set_alpha_to_one       : bool                        = False,
        rescale_betas_zero_snr : bool                        = False,
        clip_sample            : bool                        = False,
        use_exponential_sigmas : bool                        = False,
        euler_at_final         : bool                        = False,
        timesteps              : A[str, Text()]              = None,
) -> A[tuple, Name("Scheduler")]:
    scheduler_data = {
        "scheduler"             : scheduler,
        "algorithm_type"        : algorithm_type,
        "use_beta_sigmas"       : use_beta_sigmas,
        "use_karras_sigmas"     : use_karras_sigmas,
        "interpolation_type"    : interpolation_type,
        "timestep_spacing"      : timestep_spacing,
        "ays_schedules"         : ays_schedules,
        "set_alpha_to_one"      : set_alpha_to_one,
        "rescale_betas_zero_snr": rescale_betas_zero_snr,
        "clip_sample"           : clip_sample,
        "use_exponential_sigmas": use_exponential_sigmas,
        "euler_at_final"        : euler_at_final,
        "timesteps"             : timesteps,
    }
    return (scheduler, scheduler_data)


@node(name="Generate Image", display=True)
def generate_image(
    model_or_compiler   : tuple                                               = None,
    encodings           : TensorType                                          = None,
    prompt              : str                                                 = None,
    scheduler           : tuple                                               = None,
    num_inference_steps : A[int, Numerical(min=0, max=250, step=1)]           = 10,
    guidance_scale      : A[float, Numerical(min=0.00, max=50.00, step=0.01)] = 5,
    eta                 : A[float, Numerical(min=0.00, max=1.00, step=0.01)]  = 5,
    dynamic_guidance    : bool                                                = False,
    precision           : Literal[*variant_list]                              = "F16",
    device              : Literal[*spec]                                      = next(iter(spec), "cpu"),
    offload_method      : Literal[*offload_list]                              = "none",
    output_type         : Literal["latent"]                                   = "latent"
) -> A[TensorType, Name("Latent")]:
    if offload_method != "none": 
        pipe = insta.offload_to(offload_method)
    if dynamo != False:
        gen_data["return_dict"]   = False
    if dynamic_guidance is True: 
        gen_data["callback_on_step_end"] = insta._dynamic_guidance
        gen_data["callback_on_step_end_tensor_inputs"] = ['prompt_embeds', 'add_text_embeds','add_time_ids']  
    gen_data["num_inference_steps"] = num_inference_steps
    gen_data["guidance_scale"] = guidance_scale
    gen_data["variant"] = precision
    gen_data["eta"]
    latent = insta.diffuse_latent(gen_data, scheduler[0], scheduler[1])
    return latent

@node(name="Autodecode", display=True)
def autodecode(
    latent   : TensorType = None,
    vae      : ModelType  = None,
    vae_slice : bool      = False,
    vae_tile  : bool      = False,
    upcast    : bool      = True,
) -> Image:
    
    queue = insta.decode_latent(vae, vae_slice, vae_tile, upcast)
    for image in range(queue):
        return image

@node(name="Save / Preview Image", display=True)
def save_preview_img(
    image: Image,
    file_prefix:  A[str, Text(multiline=False, dynamic_prompts=True)] = "Shadowbox-",
    format: A[Literal["png","jpg","optimize"], Dependent(on="temp", when="False")] = "optimize",
    compress_level: A[int, Slider(min=1, max=9, step=1),  Dependent(on="format", when=(not "optimize"))] = 7,
    temp: bool = False,
) -> I[Any]:
    tempformat="optimize", compress_level="7"
    image = """Inference.postprocess""" #pipe.image_processor.postprocess(image, output_type='pil')[0]
    insta.save_image(image, file_prefix)
    counter = format(len(os.listdir(config.get_path("output")))) #file count
    file_prefix = os.path.join(config.get_path("output"), file_prefix)
    image.save(f'{file_prefix + counter}.png')
    yield image

@node(name="Timecode", display=True)
def tc() -> I[Any]:
    clock = perf_counter_ns() # 00:00:00
    tc = f"[ {str(datetime.timedelta(milliseconds=(((perf_counter_ns()-clock)*1e-6))))[:-2]} ]"
    yield tc, sleep(0.6)

