
# API CODE

#### CLASS Config
#### IMPORT from sdbx import config
#### METHODS get_default, get_path_contents, get_path
#### PURPOSE find source directories and data
#### SYNTAX
```
        config.get_default(filename with no extension, key)               (!cannot find sub-keys on its own)
        config.get_path_contents("string_to_folder.string_to_sub_folder") (see config/config.json, config/directories.json)
        config.get_path("filename") or config.get_path("string_to_folder.filename")
```
### OUTPUT contents of a value for a key, file contents of a directory and sub directories, path to a file

#### CLASS ReadMeta
#### IMPORT from sdbx.indexer import ReadMeta
#### METHODS data
#### PURPOSE extract metadata from model files
#### SYNTAX 
```
        metadata = ReadMeta(full_path_to_file).data()                 (see config/tuning.json)
```
#### OUTPUT dict of int and str, a form filled model_tag[] 

#### CLASS EvalMeta
#### IMPORT from sdbx.indexer import EvalMeta
#### METHODS, process_vae, process_vae_no_12, process_lora, process_tf, process_model 
#### PURPOSE interpret metadata from model files
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
#### IMPORT from sdbx.nodes.tuner import IndexManager
#### METHODS write_index, fetch_compatible
#### PURPOSE manage model type lookups, search for compatibility data
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
        filter = parse_compatible(self, query, a/b/c)                        #(show only a type of result)

```
#### OUTPUT json file with model metadata, a set of dicts with all compatible models, a dict of model compatible codes

#### CLASS NodeTuner
#### IMPORT from sdbx.nodes.tuner import NodeTuner
#### METHODS get_tuned_parameters, tuned_parameters
#### PURPOSE collect model defaults and apply to current node graph
#### SYNTAX
```
        attune = get_tuned(self, metadata, widget_inputs, node_manager, node_id: str,  graph: MultiDiGraph,)
        defaults = tuned_parameters(self, model) 

        dict structure:
                0 model-----------.
                1 vae------.       |
                2 lora----. |      |
                3 tra----. ||      |
                4 pipe-.0 size     |     
                |       1 path     |     
                |       2 dtype--. |
                |                 ||
                |               3 scheduler/scheduler args
                |               4 steps
        0 cache_jettison        5 cfg/dynamic guidance
        1 cpu_offload
        2 sequential_offload

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