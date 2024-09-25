import os
import json
import sdbx.indexer as indexer
from sdbx import config, logger
from sdbx.config import config_source_location
from collections import defaultdict
import psutil

class IndexManager:

    all_data = {
        "DIF": defaultdict(dict),
        "LLM": defaultdict(dict),
        "LOR": defaultdict(dict),
        "TRA": defaultdict(dict),
        "VAE": defaultdict(dict),
    }
    
    def write_index(self, index_file="index.json"):
        # Collect all data to write at once
        self.directories =  config.get_default("directories","models") #multi read
        self.delete_flag = True
        for each in self.directories:
            self.path_name = config.get_path(f"models.{each}")
            index_file = os.path.join(config_source_location, index_file)
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
                            self.all_data[tag][filename][compatability] = (data)
                        else:
                            logger.debug(f"No eval: {each}.", exc_info=True)
                    else:
                        log = f"No data: {each}."
                        logger.debug(log, exc_info=True)
                        print(log)
        if self.all_data:
            if self.delete_flag:
                try:
                    os.remove(index_file)
                    self.delete_flag =False
                except FileNotFoundError as error_log:
                    logger.debug(f"'Config file absent at write time: {index_file}.'{error_log}", exc_info=True)
                    self.delete_flag =False
                    pass
            with open(os.path.join(config_source_location, index_file), "a", encoding="UTF-8") as index:
                json.dump(self.all_data, index, ensure_ascii=False, indent=4, sort_keys=True)
        else:
            log = "Empty model directory, or no data to write."
            logger.debug(f"{log}{error_log}", exc_info=True)
            print(log)

    def _fetch_matching(self, data, query, path=None, return_index_nums=False):
        if path is None: path = []

        if isinstance(data, dict):
            for key, self.value in data.items():
                self.current = path + [key]
                if self.value == query:
                    return self.__unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self._fetch_matching(self.value, query, self.current)
                    if self.match:
                        return self.match
        elif isinstance(data, list):
            for key, self.value in enumerate(data):
                self.current = path if not return_index_nums else path + [key]
                if self.value == query:
                    return self.__unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self._fetch_matching(self.value, query, self.current)
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
    
    def fetch_id(self, id):
        for each in self.all_data.keys():
            peek_index = config.get_default("index",each)
            sub_index = list(peek_index)
            for i in range(len(sub_index)):
                if id in sub_index[i]:
                    return peek_index[sub_index[i]]
                    
    def fetch_compatible(self, query): 
        self.clip_data = config.get_default("tuning", "clip_data") 
        self.lora_priority = config.get_default("tuning", "lora_priority") 
        self.vae_index = config.get_default("index", "VAE")
        self.tra_index = config.get_default("index", "TRA")
        self.lor_index = config.get_default("index", "LOR")
        self.model_indexes = {
            "vae": self.vae_index,
            "tra": self.tra_index, 
            "lor": self.lor_index
            }
        try:
            path = self._fetch_matching(self.clip_data, query)
        except TypeError as error_log:
            log = f"No match found for {query}"
            logger.debug(f"{log}{error_log}", exc_info=True)
            print(log)
        else:
            if path: # fetch compatible everything
                transformers_list = [each for each in path if each !=query]
                vae = IndexManager().filter_compatible(query, self.model_indexes["vae"])
                if vae == []: logger.debug(f"No external VAE found compatible with {query}.", exc_info=True)
                lor = IndexManager().filter_compatible(query, self.model_indexes["lor"])
                if lor == []: 
                    logger.debug(f"No compatible LoRA found for {query}.", exc_info=True)
                else:
                    j = 0
                    lora = []
                    for each in self.lora_priority:
                        for i in range(len(lor)):
                            if each in lor[i][0][1]:
                                lora.append(lor[i])
                                j += 1
                tra = []
                tra_x = {}
                for i in range(len(transformers_list)):
                    tra_x[i] = IndexManager().filter_compatible(transformers_list[i], self.model_indexes["tra"])
                if tra_x == {}: 
                    logger.debug(f"No external text encoder found compatible with {query}.", exc_info=True)
                else:
                    prefix= tra_x[i][0][0]
                    suffix = tra_x[i][0][1]
                    tra.append((prefix,suffix))
                return vae, lora, tra
            
    def filter_compatible(self, query, index):
        pack = defaultdict(dict)
        #print(index)
        for k, v in index.items():
            for code in v.keys():
                if query in code:
                    pack[k, code] = v[code]
                    
        sort = sorted(pack.items(), key=lambda item: item[1])
        return sort
    

# class NodeTuner:
#     def __init__(self, fn):
#          self.fn = fn
#          self.name = fn.info.fname

pipe = None
model = "tPonynai3_v55"
fetch = IndexManager().fetch_id(model) 
data = list(fetch.keys())[0]
vae, lora, tra = IndexManager().fetch_compatible(data)
model_size = fetch[data][0]
model_file = fetch[data][1]
model_precision = fetch[data][2]
if data == "STA-XL": upcast_vae = True
    compile_unet = False #if unet #[performance] unet only, compile the model for speed, slows first gen only, doesnt work on my end
    pipeline = "type_of_pipeline_from_tuning_file"

if vae:
    vae_size = next(iter(vae[0][1]))
    vae_precision = next(iter(vae[0][1][2:3]))
    vae_file = next(iter(vae[0][1][1:2]))
else :
    vae_file = model_file
    vae_config_path = os.path.join(config.get_path("models.metadata"),"vae",data)
if lora:
    lora_size = next(iter(lora[0][1]))
    lora_precision = next(iter(lora[0][1][2:3]))
    lora_file = next(iter(lora[0][1][1:2]))
    # scheduler_args
    # timestep_spacing = "trailing"  # [compatibility] DDIM, PCM "trailing"
    # clip_sample = False  # [compatibility] PCM False
    # set_alpha_to_one = False,  # [compatibility]PCM False
    # rescale_betas_zero_snr = True  # [compatibility] DDIM True 
    # steps x # [compatibility] LCM, LIG, FLA, DPO, PCM, HYP, TCD True
    # cfg x # [compatibility] LCM, LIG, FLA, DPO, PCM, HYP, TCD True
    # dynamic_guidance = True [universal] half cfg @ 50-75%. xl only.no lora/pcm
else:
    """
    pipe_args
    model_ays = "StableDiffusionTimesteps"
    model_ays = "StableDiffusionXLTimesteps
    model_ays = "StableDiffusion3Timesteps
    inference_steps = 10
"""
if tra:
    tra_size = {}
    tra_precision = {}
    tra_file = {}
    for each in tra:
        tra_size[each] = each[0][1]
        tra_precision[each] = each[0][1][2:3]
        tra_file[each] = each[0][1][1:2]
        larger_tra = each if tra_size[each] > model_size else False
else:
    tra_file = model_file

"""
import torch / torch.cuda.mem_get_info()
"""
avail_vid_ram = 4294836224
peak_gpu_ram = avail_vid_ram*.95
peak_cpu_ram = psutil.virtual_memory().total
overhead = vae_size + lora_size + model_size
cpu_ceiling =  model_size/psutil.virtual_memory().total if not larger_tra else tra_size[each]/psutil.virtual_memory().total
gpu_ceiling = model_size/avail_vid_ram[1] if not larger_tra else tra_size[each]/avail_vid_ram[1]
if overhead > peak_gpu_ram: 
    if overhead > peak_cpu_ram+peak_gpu_ram:
        disk_offload = True  #>90  # [compatibility] last resort, but things work
        cpu_offload = True  #90 # [compatibility] lower vram use by forcing to cpu
        sequential_offload = True # >75 # [universal] lower vram use (and speed on pascal apparently!!)
    elif overhead > peak_cpu_ram:
        disk_offload = False
        cpu_offload = True  #90 # [compatibility] lower vram use by forcing to cpu
        sequential_offload = True
    elif overhead < peak_cpu_ram: # presumably the vast majority of people are here
        disk_offload = False
        cpu_offload = False
        sequential_offload = True
    cache_jettison = True 

else:
    """
    best case scenario, full ram
    """

        # return {
        #     "function name of node": {
        #         "parameter name": "parameter value",
        #         "parameter name 2": "parameter value 2"
        #     }


#compare mem
# avail_vid_ram = overhead/torch.cuda.mem_get_info()


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
            