
# API

## CLASS Config
##### IMPORT from sdbx.config import
##### METHODS get_default, get_path_contents, get_path, model_indexer
##### VARIABLES config_source_location
##### PURPOSE find source directories and data
##### OUTPUT a dict of keys, a dict of files, a path to a file
##### RETURN FORMAT: {key:}, 
##### SYNTAX
```
        config.get_default(filename with no extension, key)               (!cannot find sub-keys on its own)
        config.get_path_contents("string_to_folder.string_to_sub_folder") (see config/config.json, config/directories.json)
        config.get_path("filename") or config.get_path("string_to_folder.filename")
        
        os.path.join(config_source_location,filename)
```
from sdbx.config import model_indexer
from sdbx.indexer import ModelType


## CLASS ReadMeta
##### IMPORT from sdbx.indexer import ReadMeta
##### METHODS data
##### PURPOSE extract metadata from model files

##### SYNTAX 
```
        metadata = ReadMeta(full_path_to_file).data()                 (see config/tuning.json)

```

##### OUTPUT a dict of block data & model measurements for a single model 
##### RETURN FORMAT: {model_tag: }, a dict of extracted metadata as integers and strings
```
metareader = {'filename': 'noosphere_v42.safetensors', 'size': 2132626794, 'path': 'C:\\Users\\woof\\AppData\\Local\\Shadowbox\\models\\image\\noosphere_v42.safetensors', 'dtype': 'F16', 'tensor_params': 1133, 'shape': [], 'data_offsets': [2132471232, 2132471234], 'extension': 'safetensors', 'diffusers_lora': 293, 'unet': 464, 'mmdit': 72, 'hunyuan': 24, 'transformers': 106, 'sd': 254, 'diffusers': 224}
```

<hr>

## CLASS EvalMeta
##### IMPORT from sdbx.indexer import EvalMeta
##### METHODS, process_vae, process_vae_no_12, process_lora, process_tf, process_model 
##### PURPOSE interpret metadata from model files
##### SYNTAX 
```
        index_code = EvalMeta(dict_metadata_from_ReadMeta).data()        (see config/tuning.json)
                        tag = item[0]                   (TRA, LOR, LLM, DIF, VAE)
                        filename = item[1][0]           (base-name only)
                        compatability = item[1][1:2][0] (short code)
                        data = item[1][2:5]             (meta data dict)

```
##### OUTPUT list of strings that identifies a single model as a certain class
##### RETURN FORMAT: [0:] tag code, [1:] file size, [2:] full path (see tuner.json)
```
evaluate = ('DIF', ('noosphere_v42.safetensors', 'STA-15', 2132626794, 'C:\\Users\\woof\\AppData\\Local\\Shadowbox\\models\\image\\noosphere_v42.safetensors', 'F16'))
```

<hr>

## CLASS ModelIndexer
##### IMPORT from sdbx.indexer import IndexManager
##### METHODS write_index, fetch_compatible
##### PURPOSE manage model type lookups, search for compatibility data
##### OUTPUT a .json file of available model info, a dict of compatible models.
##### RETURN FORMAT: { filename: { code: [size, path, dtype] } }
##### SYNTAX 
```

        create_index = config.model_indexer.write_index(optional_filename)       # (defaults to config/index.json)
        fetch = IndexManager().fetch_id(query_as_string)                   # (single id search, first candidate only)
        a,b,c = IndexManager().fetch_compatible(model_class)                     # (automated all type search)
        ## using next(iter(___)):
                a,b,c[0][0] filename
                a,b,c[0][0][1:2] compatability short code

                a,b,c[0][1] size
                a,b,c[0][1][1:2] path
                a,b,c[0][1][2:3] dtype
        filter = parse_compatible(self, query, a/b/c)                      # (show only a type of result)
        fetch = IndexManager().fetch_refiner()                             # Just find STA-XR model only
                                                                           #    template func for controlnet,
                                                                           #    photomaker, other specialized models

```

## CLASS NodeTuner
##### IMPORT from sdbx.nodes.tuner import NodeTuner
##### METHODS get_tuned_parameters, determine_tuning
##### PURPOSE collect model defaults and apply to current node graph
##### OUTPUT a dict of defaults ready to be passed as arguments (save for optimized and model)
        
##### RETURN FORMAT: { file: , variant: , torch_dtype, ... }
##### SYNTAX
```

                        defaults = determine_tuning(self, full_path_to_model)  

dictionary map:

                                                                          system_prompt          
    callback_on_step_end_tensor_inputs  llm-------------.                 temperature           
    guidance_scale                      model----------. |                repeat_penalty               
    num_inference_steps                 lora----------. ||                max_tokens           
    cfg                                 vae----------. |||                context
    output_type ]gen------------.       transformers. ||||        .--llm[ top_p
    return_dict     padding      `------gen          |||||       |        top_k    cache_dir
                    return_tensors ]----conditioning |||||       |                 num_hidden_layers
                    truncation     .----pipe         |||||       | .-transformers[ low_cpu_mem_usage               
                                  | .---optimized   `file        ||                attn_implementation
              torch_dtype         || .--compile     variant      ||                use_safetensors
              variant             ||| .-refiner    torch_dtype==''                         
    seq       tokenizer    ]pipe-' ||| .scheduler      ||     `-----------------model[ class
    cpu       text_encoder         ||||                ||                  
    disk      safety_checker       ||||                | `------------lora[ (reserved)               
    ays       low_cpu_mem_usage    ||||                |       
    device                         ||||                 `----------.                     cache_dir   
    refiner                        ||||                             `---------------vae[ enable_tiling
    sigmas                         ||| `-------------scheduler[ lu_lambdas               disable_tiling             
    dynamo         ]optimized-----' ||                          euler_at_final           enable_slicing
    upcast_vae                      | `--refiner[ available     clip_sample              disable_slicing
    cache_jettison                  |           use_refiner     timesteps
    file_prefix                     |       denoising_start     timestep_spacing          
    algorithm                       |   num_inference_steps     interpolation_type     
    dynamic_cfg                     |        high_noise_fra     use_exponential_sigmas
    fuse_lora_on                    |         denoising_end     use_karras_sigmas
    fuse_lora(lora_scale)           |                           set_alpha_to_one
    fuse_pipe                        `-compile[ mode            use_beta_sigmas
    fuse_unet_only                              fullgraph       rescale_betas_zero_snr
                                                                                                                
scheduler_data
    scheduler
    algorithm_type,
    timesteps,
    interpolation_type,
    timestep_spacing,
    use_beta_sigmas,
    use_karras_sigmas,
    ays_schedules,
    set_alpha_to_one,
    rescale_betas_zero_snr,
    clip_sample,
    use_exponential_sigmas,
    euler_at_final,
    lu_lambdas,
    sigmas,


empty_cache
    data
    encoder
    lora
    pipe
    vae

encode_prompt
    transformers
    queue
    padding
    truncation
    return_tensors:

compile_pipe
    pipe
    fullgraph
    mode

load_lora
    pipe
    lora_0
    fuse_0
    scale_0
    lora_1
    fuse_1
    scale_1
    lora_2
    fuse_2
    scale_2

diffusion_pipe
    model
    use_model_to_encode
    transformers
    vae
    precision
    device
    low_cpu_mem_usage
    safety_checker
    fuse
    padding
    truncation
    return_tensors

load_vae_model
    vae
    device
    precision
    low_cpu_mem_usage
    slicing
    tiling

force_device
    device_name
    gpu_id

load_transformer
    transformer_0
    transformer_1
    transformer_2
    precision_0
    precision_1
    precision_2
    clip_skip
    device
    flash_attention
    low_cpu_mem_usage

text_input
    prompt
    negative_terms
    seed
    batch

generate_image
    pipe
    queue
    encoding
    scheduler
    num_inference_steps
    guidance_scale
    eta
    dynamic_guidance
    precision
    device
    offload_method
    output_type

autodecode
    pipe
    upcast
    file_prefix
    file_format
    compress_level
    temp

load_refiner
    pipe
    high_aesthetic_score
    low_aesthetic_score
    padding
```

## CLASS T2IPipe
##### IMPORT from sdbx.nodes.compute import T2IPipe
##### METHODS declare_encoders, generate_embeddings, encode_prompt, diffuse_latent, decode_latent
##### PURPOSE run text efficiently and continuously through latent diffusion using three separate AI models
##### OUTPUT a picture in `.png` format sent to the designated `output` folder
##### SYNTAX
```
        txt2img = T2IPipe()
        txt2img.declare_encoders(exp):                               (init text transformer models)
                generate_embeddings(prompts, tokenizers, text_encoders, exp)  (configure embedding gen)
                encode_prompt(exp)                                   (run prompt through process)
                construct_pipe(exp)                                  (load main model pipe)
                diffuse_latent(exp)                                  (noise & denoise sample on schedule)
                decode_latent(exp)                                   (convert latent sample to image)
```

- spinoffs
>hf_log, float_converter, algorithm_converter, set_device, queue_manager,

- nodes
>add_dynamic_cfg, _dynamic_guidance, add_lora, cache_jettison, offload_to, compile_model, metrics


### Exception handling
##### IMPORT from sdbx.config import logger
##### METHODS debug, exception
##### OUTPUT detailed error message in log or console
##### SYNTAX
```
        logger.debug(self.path, error_log, , exc_info=True)                 (quiet log)
        logger.exception(self.path, error_log)                              (hard lockup/os freeze only)
```

## MODEL KEY
 - DIF - Diffusion / LLM - Large Language Model / TRA - Text Transformer / LOR -LoRA / VAE - VariableAutoencoder

## CLASS KEY
- AUR-03 Auraflow
- COM-XC Common Canvas XL C
- COM-XN Common Canvas XL NC
- FLU-1D Flux 1 Dev
- FLU-1S Flux 1 Schnell
- HUN-12 HunyuanDit 1.2
- KOL-01 Kolors 1 
- LCM-PIX-AL Pixart Alpha LCM Merge 
- LLM-AYA-23 Aya 23 
- LLM-DEE-02-INS Deepseek Instruct 
- LLM-DOL-25 Dolphin 
- LLM-LLA-03 LLama3 
- LLM-MIS-01 Mistral 
- LLM-MIS-01-INS Mistral Instruct 
- LLM-NEM-04-INS Nemotron Instruct 
- LLM-OPE-12 OpenOrca 
- LLM-PHI-35-INS Phi Instruct 
- LLM-QWE-25-INS Qwen Instruct 
- LLM-SOL-10-INS Solar Instruct 
- LLM-STA-02 Starcoder 
- LLM-STA-02-INS Starcoder 02 Instruct 
- LLM-ZEP-01 Zephyr 
- LORA-FLA-STA-XL Flash XL 
- LORA-LCM-SSD-1B SSD 1B LCM 
- LORA-LCM-STA-15 Stable Diffusion 1.5 LCM 
- LUM-01 Lumina T2I 
- LUM-NS Lumina Next SFT 
- MIT-D1 Mitsua 
- PIX-AL Pixart Alpha 
- PIX-SI Pixart Sigma 
- PLA-25 Playgroud 2.5 
- SD1-TR Stable Diffusion 1.5 Turbo 
- SDX-TR Stable Diffusion XL Turbo 
- SEG-VG Segmind Vega 
- SSD-1B SSD-1B 
- SSD-1L SSD-1B LCM 
- STA-15 Stable Diffusion 1.5 
- STA-3D Stable Diffusion 3 Diffusers 
- STA-3M Stable Diffusion 3 Medium 
- STA-CA Stable Cascade 
- STA-XL Stable Diffusion XL 
- STA-XR Stable Diffusion XL Refiner 
- TIN-SD Tiny Stable Diffusion 1.5 
- WUR-01 Wuerstchen
