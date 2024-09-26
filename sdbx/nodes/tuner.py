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
                    return self._unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self._fetch_matching(self.value, query, self.current)
                    if self.match:
                        return self.match
        elif isinstance(data, list):
            for key, self.value in enumerate(data):
                self.current = path if not return_index_nums else path + [key]
                if self.value == query:
                    return self._unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self._fetch_matching(self.value, query, self.current)
                    if self.match:
                        return self.match
                    
    def _unpack(self): 
        iterate = []  
        self.match = self.current, self.value           
        for i in range(len(self.match)-1):
            for j in (self.match[i]):
                iterate.append(j)
        iterate.append(self.match[len(self.match)-1])
        return iterate
    
    def fetch_id(self, search_item):
        for each in self.all_data.keys(): 
            peek_index = config.get_default("index", each)
            if not isinstance(peek_index, dict):
                continue  # Skip if peek_index is not a dict
            if search_item in peek_index:
                return peek_index[search_item]  # Return keys and corresponding value

        return "∅"  # Return "∅" if search_item is not found in any peek_index


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
            tra_sorted = "∅"
        else:
            if path == None: 
                tra_sorted = "∅"
            else:
                transformers_list = [each for each in path if each !=query]
                tra_match = {}
                tra_sorted = []
                for i in range(len(transformers_list)):
                    tra_match[i] = IndexManager().filter_compatible(transformers_list[i], self.model_indexes["tra"])
                try:   
                    if list(tra_match)[0] == 0:
                        logger.debug(f"No external text encoder found compatible with {query}.", exc_info=True)
                    else:
                        for i in range(len(tra_match)):
                            prefix = tra_match[i][0][0]
                            suffix = tra_match[i][0][1]
                            tra_sorted.append((prefix,suffix))
                except IndexError as error_log:
                    logger.debug(f"No external text encoder found compatible with {query}.", exc_info=True)

        vae_sorted = IndexManager().filter_compatible(query, self.model_indexes["vae"])
        lora_match = IndexManager().filter_compatible(query, self.model_indexes["lor"])
        j = 0
        lora_sorted = []
        for each in self.lora_priority:
            for i in range(len(lora_match)):
                if each in lora_match[i][0][1]:
                    lora_sorted.append(lora_match[i])
                    j += 1
        if vae_sorted == []: 
            vae_sorted =str("∅")
            logger.debug(f"No external VAE found compatible with {query}.", exc_info=True)
        if lora_sorted == []: 
            lora_sorted =str("∅")
            logger.debug(f"No compatible LoRA found for {query}.", exc_info=True)
        if tra_sorted == []: 
            tra_sorted =str("∅")
            logger.debug(f"No external text encoder found compatible with {query}.", exc_info=True)
        return vae_sorted, lora_sorted, tra_sorted
            
    def filter_compatible(self, query, index):
        pack = defaultdict(dict)
        if index.items():
            for k, v in index.items():
                for code in v.keys():
                    if query in code:
                        pack[k, code] = v[code]
                        
            sort = sorted(pack.items(), key=lambda item: item[1])
            return sort
        else:
            print("Compatible models not found")
            return "∅"
    

class NodeTuner:
#     def __init__(self, fn):
#          self.fn = fn
#          self.name = fn.info.fname

    #@cache
    def get_tuned(self, model):# (metadata, widget_inputs, node_manager, node_id: str,  graph: MultiDiGraph, ):

        # predecessors = graph.predecessors(node_id)
        # node = graph.nodes[node_id]

    
            # for p in predecessors:
            #     pnd = graph.nodes[p]  # predecessor node data
            #     pfn = node_manager.registry[pnd['fname']]  # predecessor function
            #     p_tuned_parameters = pfn.tuner.get_tuned_parameters(pnd['widget_inputs'])[node['fname']]
            #     tuned_parameters |= p_tuned_parameters
        self.tuned_parameters()
        """
        # return {
        #     "function name of node": {
        #         "parameter name": "parameter value",
        #         "parameter name 2": "parameter value 2"
        #     }
        """

    def tuned_parameters(self, model): 
        self.pipe = None
        self.model = model
        self.fetch = IndexManager().fetch_id(self.model)
        #print(self.fetch[list(self.fetch)[0]])
        if self.fetch != "∅" or None:
            self.category = list(self.fetch)[0]
            self.fetch = self.fetch[list(self.fetch)[0]]
            if self.category == "LLM":
                """
                do LLM things
                """
            else:
                self.params = defaultdict(dict)
                if "LLM" not in self.category:
                    self.vae, self.lora, self.tra = IndexManager().fetch_compatible(self.category)
                    self.num = 2
                    self.params["model"] = self.fetch
                    if self.vae != "∅":  
                        self.params["vae"] = self.vae[0][1]
                    if self.lora !=  "∅": 
                        self.params["lora"] = self.lora[0][1]
                    else :
                        if "STA-15" == self.category: self.model_ays = "StableDiffusionTimesteps"
                        elif "STA-XL" == self.category: self.model_ays = "StableDiffusionTimesteps"
                        elif "STA-3" in self.category: self.model_ays = "StableDiffusionTimesteps"
                        self.inference_steps = 10
                    if self.tra != "∅": 
                        for i in range(len(self.tra[0])-1):
                            for e in self.tra:
                                self.params[f"tra{i}"] = self.tra[i]
                                if self.tra[i] > self.params["model"][0]: larger_tra = f"tra{i}"

                if self.category == "STA-XL": 
                    upcast_vae = True #force vae to f32 by default, because the default vae is broken
                    compile_unet = False #if unet #[performance] unet only, compile the model for speed, slows first gen only
                # refiner
                # high_noise_frac = 0.8
                # denoising_end=high_noise_frac,
                # num_inference_steps=n_steps,
                # denoising_start=high_noise_frac,

                # scheduler_args
                # timestep_spacing = "trailing"  # [compatibility] DDIM, PCM "trailing"
                #clip_sample = False if "PCM" not in next(iter(lora[0][0][1:2]))  # [compatibility] PCM False
                # set_alpha_to_one = False,  # [compatibility]PCM False
                # rescale_betas_zero_snr = True  # [compatibility] DDIM True 
                # steps x # [compatibility] LCM, LIG, FLA, DPO, PCM, HYP, TCD True
                # cfg x # [compatibility] LCM, LIG, FLA, DPO, PCM, HYP, TCD True
                # dynamic_guidance = True [universal] half cfg @ 50-75%. xl only.no lora/pcm
                #import torch / torch.cuda.mem_get_info()
                vae_config_path = os.path.join(config.get_path("models.metadata"),"vae",self.category)
                avail_vid_ram = 4294836224 #accomodate list of graphics card ram
                peak_gpu_ram = avail_vid_ram*.95
                peak_cpu_ram = psutil.virtual_memory().total
                overhead = self.params["model"][0] if not larger_tra else self.params[larger_tra][0]
                cpu_ceiling =  overhead/psutil.virtual_memory().total
                gpu_ceiling = overhead/avail_vid_ram
                oh_no = [False, False, False, False,]
                if overhead > peak_gpu_ram: 
                    oh_no[2] = True
                    if overhead > peak_cpu_ram+peak_gpu_ram:
                        oh_no[1] = True
                    elif overhead > peak_cpu_ram:
                        oh_no[0] = True
                vae_tile = True  >75 # [compatibility] tile vae input to lower memory
                vae_slice = False >75  # [compatibility] serialize vae to lower memory
                #print(oh_no)
                cache_jettison = oh_no[2]
                cpu_offload = oh_no[1]
                sequential_offload = oh_no[0]
                # batch = 1 > 75, 
                # no batch limit <50
                token_encoder_default = self.params["model"][1]
                #if unet - compile? pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
                #look for quant, load_in_4bit=True >75

                
                # strand types [
                #     upscale
                #     hiresfix
                #     zero conditioning
                #     prompt injection
                #     x/y map

                return self.params

#index = IndexManager().write_index()
path = config.get_path("models.image")
for each in os.listdir(path):
    if not os.path.isdir(each):
        #full_path=os.path.join(path,each)
        default = NodeTuner().tuned_parameters(each)
        print(default)
var_list = [
    "__doc__",
    "__name__",
    "__package__",
    "__loader__",
    "__spec__",
    "__annotations__",
    "__builtins__",
    "__file__",
    "__cached__",
    "config",
    "indexer",
    "json",
    "os",
    "defaultdict",
    "IndexManager",
    "logger",
    "psutil",
    "var_list",
    "i"
    ]
variables = dict(locals())
for each in variables:
    if each not in var_list:
        print(f"{each} = {variables[each]}")




            