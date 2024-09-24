import os
import json
import sdbx.indexer as indexer
from sdbx import config, logger
from sdbx.config import config_source_location
from collections import defaultdict

path_name =  config.get_path("models") #multi read

class IndexManager:
    
    def write_index(self, index_file="index.json"):
        all_data = {
            "DIF": defaultdict(dict),
            "LLM": defaultdict(dict),
            "LOR": defaultdict(dict),
            "TRA": defaultdict(dict),
            "VAE": defaultdict(dict),
                    }  # Collect all data to write at once
        for each in os.listdir(self.path_name):  # SCAN DIRECTORY           #todo - toggle directory scan
            full_path = os.path.join(self.path_name, each)
            if os.path.isfile(full_path):  # Check if it's a file
                self.metareader = indexer.ReadMeta(full_path).data()
                if self.metareader is not None:
                    self.eval_data = indexer.EvalMeta(self.metareader).data()
                    if self.eval_data != None:
                        tag = self.eval_data[0]
                        filename = self.eval_data[1][0]
                        compatability = self.eval_data[1][1:2][0]
                        data = self.eval_data[1][2:5]
                        all_data[tag][filename][compatability] = (data)
                    else:
                        logger.debug(f"No eval: {each}.", exc_info=True)

                else:
                    log = f"No data: {each}."
                    logger.debug(log, exc_info=True)
                    print(log)
        if all_data:
            index_file = os.path.join(config_source_location, index_file)
            print(index_file)
            try:
                os.remove(index_file)
            except FileNotFoundError as error_log:
                logger.debug(f"'Config file absent at write time: {index_file}.'{error_log}", exc_info=True)
                pass
            with open(os.path.join(config_source_location, index_file), "a", encoding="UTF-8") as index:
                json.dump(all_data, index, ensure_ascii=False, indent=4, sort_keys=True)
        else:
            log = "Empty model directory, or no data to write."
            logger.debug(f"{log}{error_log}", exc_info=True)
            print(log)

    def fetch_matching(self, data, query, path=None, index=False):
        if path is None: path = []

        if isinstance(data, dict):
            for key, self.value in data.items():
                self.current = path + [key]
                if self.value == query:
                    return self.__unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self.fetch_matching(self.value, query, self.current)
                    if self.match:
                        return self.match
        elif isinstance(data, list):
            for key, self.value in enumerate(data):
                self.current = path if not index else path + [key]
                if self.value == query:
                    return self.__unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self.fetch_matching(self.value, query, self.current)
                    if self.match:
                        return self.match
                    
    def __unpack(self): 
        iterate = []  
        self.match = self.current, self.value           
        for i in range(len(self.match)-1):
            for j in (self.match[i]):
                iterate.append(j)
        iterate.append(self.match[len(self.match)-1])
        return iterate
    
    def parse_compatible(self, query, index):
        pack = defaultdict(dict)
        for k, v in index.items():
            for code in v.keys():
                if query in code:
                    pack[k, code] = v[code]
                    
        sort = sorted(pack.items(), key=lambda item: item[1])
        return sort


    def fetch_compatible(self, query): 
        clip_data = config.get_default("tuning", "clip_data") 
        vae_index = config.get_default("index", "VAE")
        tra_index = config.get_default("index", "TRA")
        lor_index = config.get_default("index", "LOR")
        model_indexes ={"vae": vae_index,
                        "tra": tra_index, 
                        "lor": lor_index}

        try:
            path = self.fetch_matching(clip_data, query)
        except TypeError as error_log:
            log = f"No match found for {query}"
            logger.debug(f"{log}{error_log}", exc_info=True)
            print(log)
        else:
            if path: # fetch compatible everything
                transformers_list = [each for each in path if each !=query]
                vae = IndexManager().parse_compatible(query, model_indexes["vae"])
                if vae == []: logger.debug(f"No external VAE found compatible with {query}.", exc_info=True)
                lor = IndexManager().parse_compatible(query, model_indexes["lor"])
                if lor == []: logger.debug(f"No compatible LoRA found for {query}.", exc_info=True)
                tra = []
                tra_x = {}
                for i in range(len(transformers_list)):
                    tra_x[i] = IndexManager().parse_compatible(transformers_list[i], model_indexes["tra"])
                    prefix= tra_x[i][0][0]
                    suffix = tra_x[i][0][1]
                    tra.append([prefix,suffix])
                if tra == {}: logger.debug(f"No external text encoder found compatible with {query}.", exc_info=True)
                return vae, lor, tra

# class NodeTuner:
#     def __init__(self, fn):
#         self.fn = fn
#         self.name = fn.info.fname


# find model name in [models]

        # tuned parameters & hyperparameters only!! pcm parameters here 

        # return {
        #     "function name of node": {
        #         "parameter name": "parameter value",
        #         "parameter name 2": "parameter value 2"
        #     }

        # > 1mb embeddings
        # up to 50mb?
        # pt and safetensors
        # <2 tensor params

        # pt files


# if short_code == "LLM":
#     """
#     context_len = model_precision
#     """




# choose pipeline
    # set pipeline params
    #     resolution

# #find compatible vae
# if model_dict[0]: #VAE
#     print(next(iter(model_dict[0]))[1][2] )
#     #remember size
#     #remember dtype
#     # small vae if >75
# else:

#     """
#     set flag for using built-in vae
#     """

# fetch vaeconfig.json
# pipe.upcast_vae() if sdxl

# if model_dict[1]:
#     sort_lora = {}
#     lora_types = ["PCM", "SPO", "TCD", "HYP", "FLA", "LCM", "DMD"]
#     for i in range(len(lora_types)-1):
#         sort_lora.update([each for each in model_dict[1] if lora_types[i] in str(each[0][1])])
#     sort_lora_values = sort_lora.values()
#     print(next(iter(sort_lora_values))[2]) #LORA
    #remember size
    #remember dtype
    #     scheduler (if not already changed)
    #     inference steps (if not already changed)
    #     cfg (if not already changed) # dynamic cfg (rarely)
    # timestep_spacing = "trailing"  # [compatibility] DDIM, PCM "trailing"
    # clip_sample = False  # [compatibility] PCM False
    # set_alpha_to_one = False,  # [compatibility]PCM False
    # rescale_betas_zero_snr = True  # [compatibility] DDIM True 
    # steps x # [compatibility] LCM, LIG, FLA, DPO, PCM, HYP, TCD True
    # cfg x # [compatibility] LCM, LIG, FLA, DPO, PCM, HYP, TCD True
    # dynamic_guidance = True [universal] half cfg @ 50-75%. xl only.no lora/pcm

# else:
#     """
#     dont add lora loader
#     """
    # model_ays = "StableDiffusionTimesteps"
    # model_ays = "StableDiffusionXLTimesteps
    # model_ays = "StableDiffusion3Timesteps

#import torch / torch.cuda.mem_get_info()
#compare mem
# avail_vid_ram = overhead/torch.cuda.mem_get_info()

# avail_vid_ram = 4294836224
# peak_gpu_ram = avail_vid_ram*.95
# print((avail_vid_ram)/1048576)
# print(peak_gpu_ram)
# # overhead = model_size+lora_size+vae+size
# #cpu_ceiling = overhead/psutil.virtual_memory().total
# gpu_ceiling = model_size/avail_vid_ram[1]

#calc ram-specific params
    # try independent unet
    # try independent clips
    # token_encoder_default = "TRA"-found[1] if "TRA"=found[1] else: found[0]# this should autodetect
    # if model_dict[2]: #TRANSFORMER
    #     for num in range(len(model_dict[2])):
    #         print(model_dict[2][num][1][2])
    #     #set size
    #     #set dtype
    # else:
    #     """
    #     set flag to use built-in clip
    #     """
    # look for quant, load_in_4bit=True >75

    # dtypes auto <50
    # dtypes 16 or less >75

    # batch = 1 > 75, 
    # no batch limit <50
    # vae_tile = True  >75 # [compatibility] tile vae input to lower memory
    # vae_slice = False >75  # [compatibility] serialize vae to lower memory

    # sequential_offload = True >75 # [universal] lower vram use (and speed on pascal apparently!!)
    # cpu_offload = False  >90 # [compatibility] lower vram use by forcing to cpu
    # disk_offload = False >90  # [compatibility] last resort, but things work
    # compile_unet = False if unet #[performance] unet only, compile the model for speed, slows first gen only, doesnt work on my end
    # cache jettison - True > 75


# refiner
    #     high_noise_frac = 0.8
    #     image = base(
    #     image = refiner(
    #     high_noise_frac = 0.8
    #     denoising_end=high_noise_frac,
    #     num_inference_steps=n_steps,
    #     denoising_start=high_noise_frac,

#     compile? pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)

# strand types [
#     upscale
#     hiresfix
#     zero conditioning
#     prompt injection
#     x/y map



# query = 'STA-3'
# vae, lora, tra = IndexManager().fetch_compatible(query)
# model_dict = [vae, lora, tra]

#     @cache
#     def get_tuned_parameters(self, widget_inputs, model_types, metadata):
#         max_value = max(metadata.values())
#         largest_keys = [k for k, v in metadata.items() if v == max_value] # collect the keys of the largest pairs
#         ReadMeta.full_data.get("model_size", 0)/psutil.virtual_memory().total

#         torch.device(0)
#         torch.get_default_dtype() (default float)
#         torch.cuda.mem_get_info(device=None) (3522586215, 4294836224)



#     def tuned_parameters(self, node_manager, graph: MultiDiGraph, node_id: str):
#         predecessors = graph.predecessors(node_id)

#         node = graph.nodes[node_id]

#         tuned_parameters = {}
#         for p in predecessors:
#             pnd = graph.nodes[p]  # predecessor node data
#             pfn = node_manager.registry[pnd['fname']]  # predecessor function

#             p_tuned_parameters = pfn.tuner.get_tuned_parameters(pnd['widget_inputs'])[node['fname']]

#             tuned_parameters |= p_tuned_parameters
        
#         return tuned
            