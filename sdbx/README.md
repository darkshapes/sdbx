
# API CODE

#### CLASS Config
#### IMPORT from sdbx.config import
#### METHODS get_default, get_path_contents, get_path
#### VARIABLES config_source_location
#### PURPOSE find source directories and data
### RETURNS a dict of keys, a dict of files, a path str
#### SYNTAX
```
        config.get_default(filename with no extension, key)               (!cannot find sub-keys on its own)
        config.get_path_contents("string_to_folder.string_to_sub_folder") (see config/config.json, config/directories.json)
        config.get_path("filename") or config.get_path("string_to_folder.filename")
        
        os.path.join(config_source_location,filename)
```
### OUTPUT contents of a value for a key, file contents of a directory and sub directories, path to a file

#### CLASS ReadMeta
#### IMPORT from sdbx.indexer import ReadMeta
#### METHODS data
#### PURPOSE extract metadata from model files
#### RETURNS a dict of block data & model measurements for a single model
#### SYNTAX 
```
        metadata = ReadMeta(full_path_to_file).data()                 (see config/tuning.json)
```
#### OUTPUT dict of int and str, a form filled model_tag[] 

#### CLASS EvalMeta
#### IMPORT from sdbx.indexer import EvalMeta
#### METHODS, process_vae, process_vae_no_12, process_lora, process_tf, process_model 
#### PURPOSE interpret metadata from model files
#### RETURNS a dict that identifies a single model as a certain type 
#### SYNTAX 
```
        index_code = EvalMeta(dict_metadata_from_ReadMeta).data()        (see config/tuning.json)
                        tag = item[0]                   (TRA, LOR, LLM, DIF, VAE)
                        filename = item[1][0]           (base-name only)
                        compatability = item[1][1:2][0] (short code)
                        data = item[1][2:5]             (meta data dict)
```
#### OUTPUT list of type str: 0: tag code, 1: file size, 2: full path (see tuner.json)

#### CLASS IndexManager
#### IMPORT from sdbx.indexer import IndexManager
#### METHODS write_index, fetch_compatible
#### PURPOSE manage model type lookups, search for compatibility data
#### RETURNS a .json file of available models & info, a dict of models that work with another
#### SYNTAX 
```

        create_index = IndexManager().write_index(optional_filename)       # (defaults to config/index.json)
        fetch = IndexManager().fetch_id(query_as_string)                   # (single id search, first candidate only)
        a,b,c = IndexManager().fetch_compatible(fetch)                     # (automated all type search)
        ## using next(iter(___)):
                a,b,c[0][0] filename
                a,b,c[0][0][1:2] compatability short code

                a,b,c[0][1] size
                a,b,c[0][1][1:2] path
                a,b,c[0][1][2:3] dtype
        filter = parse_compatible(self, query, a/b/c)                      #(show only a type of result)
        fetch = IndexManager().fetch_refiner()                             # Just find STA-XR model only
                                                                           #    template func for controlnet,
                                                                           #    photomaker, other specialized models

```
#### OUTPUT json file with model metadata, a set of dicts with all compatible models, a dict of model compatible codes

#### CLASS NodeTuner
#### IMPORT from sdbx.nodes.tuner import NodeTuner
#### METHODS get_tuned_parameters, determine_tuning
#### PURPOSE collect model defaults and apply to current node graph
#### RETURNS a formatted dict of defaults, a dict of defaults
#### SYNTAX
```

 perf_counter, totals, average, memory
pipe
    model path, tokenizer, text encoder, tokenizr2, text encodr2, tokenizer 3, text encoder 3

expressions:
subroutine

clear_cache, device, dynamic guidance sequential offload,cpu offload, compile(reduce overhead, fullgraph),upcast_vae, vae_tile, vae_slice, file_prefix, output_type, compress_level, config_path, algorithm
queue
     prompt  embeddings, seed
transformers
    tokenizer, text encoder
text_encoders
    torch_dtype torch.float16 ,variant     fp16
conditioning
     prompt, padding="max_length", truncation=True, return_tensors='pt'
lora
    lora(path), weight_name (filename)
scheduler
    timestep spacing, rescale betas zero_snr, clip sample, set alpha to one,
gen
    output_type='latent', timesteps, num_inference_steps, cfg/callback on step end/callback on step end tensor inputs
    embeds
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,negative_pooled_prompt_embeds
vae
   model, dtype, cache_dir
   
 self.subroutine, self.queue, self.transformers, self.conditioning, self.pipe_dict, self.lora_dict, self.fuse, self.schedule, self.gen_dict, self.vae_dict

        defaults = determine_tuning(self, full_path_to_model)            
                    tuning dict :                 system_prompt          
                      llm------------.            temperature           
                      model---------. |           repeat_penalty               
num_inference_steps   vae----------. ||           max_tokens           
cfg                   lora--------. |||           context
output_type ]-.       transformer. ||||   .--llm[ top_p
               `------gen         |||||  |        top_k
                 .----[]          |||||  |                
batch_limit   ]-' .---pipe        `file  | .-transformer[ prompt       class
cache_jettison   | .--compile      size-' |.--------------------model[ stage 
upcast           || .-refiner      dtype-'                             config
compile          ||| .scheduler    ||                  config
file_prefix      ||||              ||                  upcast
use_fast_token   ||||              ||                  slice  
streaming        ||||              | `------------vae[ tile       
dynamic_cfg      ||||               `----------.                class          
path             ||||                           `---------lora[ fuse
                 ||| `-------------scheduler[ algorithm         scale         
strength ]pipe--' ||                          lu_lambdas        unet_only
noise_eta         | `--refiner[ available     euler_at_final
cpu offload       |           use_refiner     clip_sample
sequential_offload|       denoising_start     timesteps            ##### 0-1000
manual_seed       |   num_inference_steps     timestep_spacing     ##### str 
config_path       |        high_noise_fra     interpolation_type
                  |         denoising_end     use_karras_sigmas
                  |                           use_exponential_sigmas
                   `-compile[ mode            use_beta_sigmas
                              fullgraph       rescale_betas_zero_snr
                                              set_alpha_to_one
                                  
                                              
```

#### Exception handling
#### IMPORT from sdbx.config import logger
#### SYNTAX
```
        logger.debug(self.path, error_log, , exc_info=True)                 (quiet log)
        logger.exception(self.path, error_log)                              (hard lockup/os freeze only)
```
#### OUTPUT detailed error message in log or console


#### Model Key
AUR-03 Auraflow
COM-XC Common Canvas XL C
COM-XN Common Canvas XL NC
FLU-1D Flux 1 Dev
FLU-1S Flux 1 Schnell
HUN-12 HunyuanDit 1.2
KOL-01 Kolors 1
LCM-PIX-AL Pixart Alpha LCM Merge
LLM-AYA-23 Aya 23
LLM-DEE-02-INS Deepseek Instruct
LLM-DOL-25 Dolphin
LLM-LLA-03 LLama3
LLM-MIS-01 Mistral
LLM-MIS-01-INS Mistral Instruct
LLM-NEM-04-INS Nemotron Instruct
LLM-OPE-12 OpenOrca
LLM-PHI-35-INS Phi Instruct
LLM-QWE-25-INS Qwen Instruct
LLM-SOL-10-INS Solar Instruct
LLM-STA-02 Starcoder
LLM-STA-02-INS Starcoder 02 Instruct
LLM-ZEP-01 Zephyr
LORA-FLA-STA-XL Flash XL
LORA-LCM-SSD-1B SSD 1B LCM
LORA-LCM-STA-15 Stable Diffusion 1.5 LCM
LUM-01 Lumina T2I
LUM-NS Lumina Next SFT
MIT-D1 Mitsua
PIX-AL Pixart Alpha
PIX-SI Pixart Sigma
PLA-25 Playgroud 2.5
SD1-TR Stable Diffusion 1.5 Turbo
SDX-TR Stable Diffusion XL Turbo
SEG-VG Segmind Vega
SSD-1B SSD-1B
SSD-1L SSD-1B LCM
STA-15 Stable Diffusion 1.5
STA-3D Stable Diffusion 3 Diffusers
STA-3M Stable Diffusion 3 Medium
STA-CA Stable Cascade
STA-XL Stable Diffusion XL
STA-XR Stable Diffusion XL Refiner
TIN-SD Tiny Stable Diffusion 1.5
WUR-01 Wuerstchen
