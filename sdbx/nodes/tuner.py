import os
import numbers
from sdbx.indexer import IndexManager
from sdbx import logger
from sdbx.config import config, Precision, cache, TensorDataType as D
from sdbx.nodes.helpers import soft_random
from collections import defaultdict
from diffusers.schedulers import AysSchedules


# def collect_tuned_parameters(self, node_manager, graph: MultiDiGraph, node_id: str):
#     predecessors = graph.predecessors(node_id)

#     node = graph.nodes[node_id]

#     tuned_parameters = {}
#     for p in predecessors:
#         pnd = graph.nodes[p]  # predecessor node data
#         pfn = node_manager.registry[pnd['fname']]  # predecessor function

#         p_tuned_parameters = pfn.tuner.get_tuned_parameters(pnd['widget_inputs'])[node['fname']]

#         tuned_parameters |= p_tuned_parameters
    
#     return tuned
# @cache
#     def get_tuned_parameters(self, widget_inputs, model_types, metadata):
#       
#     def tuned_parameters(self, node_manager, graph: MultiDiGraph, node_id: str):
#         predecessors = graph.predecessors(node_id)
#         node = graph.nodes[node_id]
#         tuned_parameters = {}
#         for p in predecessors:
#             pnd = graph.nodes[p]  # predecessor node data
#             pfn = node_manager.registry[pnd['fname']]  # predecessor function

class NodeTuner:

    spec = config.get_default("spec","data")
    algorithms = list(config.get_default("algorithms","schedulers"))
    solvers = list(config.get_default("algorithms","solvers"))
    metadata_path = config.get_path("models.metadata")
    first_device = next(iter(spec.get("devices","cpu")),"cpu")



    def symlinker(self, true_file, class_name, filename, full_path=False):
        symlink_head      = os.path.join(config.get_path("models.metadata"), class_name)
        symlink_full_path = os.path.join(symlink_head, filename)
        try:
            os.remove(symlink_full_path) 
        except:
            pass
        os.symlink(true_file, symlink_full_path)
        return symlink_head if full_path == False else symlink_full_path
    
    @cache        
    def determine_tuning(self, model):
        self.spec = config.get_default("spec","data")
        self.algorithms = list(config.get_default("algorithms","schedulers"))
        self.solvers = list(config.get_default("algorithms","solvers"))
        self.sort, self.category, self.fetch = IndexManager().fetch_id(model)
        self.metadata_path = config.get_path("models.metadata")
        if self.fetch != "∅" and self.fetch != None:

            self.optimized     = defaultdict(dict)
            self.model         = defaultdict(dict)
            self.queue         = defaultdict(dict)
            self.pipe          = defaultdict(dict)
            self.transformers  = defaultdict(dict)
            self.conditioning  = defaultdict(dict)
            self.lora_unsorted = defaultdict(dict)
            self.lora_sorted   = defaultdict(dict)
            self.lora          = defaultdict(dict)
            self.fuse          = defaultdict(dict)
            self.schedule      = defaultdict(dict)
            self.gen           = defaultdict(dict)
            self.refiner       = defaultdict(dict)
            self.vae           = defaultdict(dict)
            self.dynamo        = defaultdict(dict)
            self.unet          = defaultdict(dict)

        if self.sort == "LLM":
            """LLM-specific tuning"""
        elif self.sort in ["VAE", "TRA", "LOR"]:           
            """Handle VAE, TRA, LOR types when sent from individual nodes"""
        else:
            self.model["class"] = self.category
            self.model["variant"] = list(self.fetch)[2]
            self.sym_suffix = "diffusion_pytorch_model"
            self.extensions_list = [".fp16.safetensors",".safetensors"]
            self.unet["model"]      = list(self.fetch)[1]
            for model_extension in self.extensions_list:
                model_filename  = os.path.join("unet",self.sym_suffix + model_extension)
                link = self.symlinker(self.unet["model"], self.category ,model_filename)
            self.model["file"] = link
            self.unet["class"] = self.category
            self._vae, self._tra, self._lora =  IndexManager().fetch_compatible(self.category)
            self.pipe["local_files_only"]          = True
            self.pipe["low_cpu_mem_usage"]         = True
            self.pipe["low_cpu_mem_usage"]         = True
            self.transformers["local_files_only"]  = True
            self.transformers["use_safetensors"]   = True
            self.transformers["low_cpu_mem_usage"] = True
            self.vae["local_files_only"]           = True
            self.vae["config"] = os.path.join(config.get_path("models.metadata"),self.category,"vae","config.json")


    @cache
    def opt_exp(self):
        peak_cpu     = self.spec["devices"].get("cpu",1)
        if self.first_device != "cpu": 
            peak_gpu = self.spec["devices"].get(self.first_device,1)  #accommodate list of graphics card ram
        overhead    = list(self.fetch)[0]
        cpu_ceiling = overhead / peak_cpu
        if peak_gpu:
            gpu_ceiling = overhead / peak_gpu 
        else:
            gpu_ceiling = 1
        total_ceiling = overhead / (peak_cpu + peak_gpu)

        # size?  >50%   >100%  >cpu  >cpu+gpu , overhead condition numbers
        self.oh_no = [False, False, False, False,]
        if gpu_ceiling > .99: 
            self.oh_no[1] = True #look for quant, load_in_4bit=True/load_in_8bit=True
            if total_ceiling > 1:
                self.oh_no[3] = True
            elif cpu_ceiling > 1:
                self.oh_no[2] = True
        elif gpu_ceiling < .5:
            self.oh_no[0] = True

        if self.first_device == "cuda":
            compute_capability = config.device.cuda.get_device_capability()[0]
            if compute_capability < 7 & self.oh_no[0] == False:
                self.model["variant"] = "F16"
                self.transformer[self.category]["variant"] = "F16"
                self.vae["variant"] = "F16"            
        elif self.first_device == "cpu":
            self.model["variant"] = "F32"
            self.tra["variant"] = "F32"
            self.vae["variant"] = "F32"

        self.optimized["cache_jettison"] = self.oh_no[1]
        self.optimized["device"]         = self.first_device
        self.optimized["dynamic_cfg"]    = False
        self.optimized["seq"]            = self.oh_no[1]
        self.optimized["cpu"]            = self.oh_no[2]
        self.optimized["disk"]           = self.oh_no[3]
        self.optimized["file_prefix"]    = "Shadowbox-"
        self.optimized["refiner"]        = IndexManager().fetch_refiner()
        self.optimized["upcast_vae"]     = True if self.category == "STA-XL" else False # f32, fp16 broken by default
        self.optimized["fuse_lora_on"]   = False
        #self.optimized["fuse_pipe"]     = True #pipe.fuse_qkv_projections(), untested
        self.optimized["fuse_unet_only"] = False # todo - add conditions where this is useful
        if self.spec.get("flash_attention_2", 0) == True:
                    self.optimized["flash_attention_2"] = True
        if self.spec.get("dynamo",False):
            self.optimized["sigmas"] = True #
            self.optimized["dynamo"] = self.spec["dynamo"]
        return self.optimized
  
    @cache
    def pipe_exp(self):
        if self.model.get("variant",0) != "F32" and self.model.get("variant,0") != "F16": self.pipe["variant"] = self.model["variant"]
        self.pipe["tokenizer"]      = None
        self.pipe["text_encoder"]   = None
        self.pipe["tokenizer_2"]    = None
        self.pipe["text_encoder_2"] = None
        self.unet["variant"]        = self.pipe["variant"]

        #self.pipe["device_map"] = None
        if "STA-15" in self.category: 
            self.pipe["safety_checker"] = None
        return self.pipe, self.model["file"] #, self.model["file"], self.unet["model"]
    
    @cache
    def cond_exp(self):
        self.conditioning["padding"]        = "max_length"
        self.conditioning["truncation"]     = True
        self.conditioning["return_tensors"] = 'pt'
        return self.conditioning

    @cache
    def vae_exp(self):
        # ensure value returned
        print(self._vae)
        if self._vae != "∅":  
            # for vae_extension in self.extensions_list:
            #     vae_filename  = os.path.join(os.sep,"vae", self.sym_suffix + vae_extension)
            #     link          = self.symlinker(self._vae[0][1][1], self.category, vae_filename, full_path=True)
            self.optimized["vae"] = self._vae[0][1][1]
            if self.vae.get("variant",0) == 0: self.vae["variant"] = self._vae[0][1][2]
        else:
            self.optimized["vae"] = self.model["file"]
            self.vae["torch_dtype"] = "auto"
    
        if self.oh_no[2]: 
            self.vae["enable_tiling"] = True
        else: 
            self.vae["disable_tiling"] = True  # [compatibility] tile vae input to lower memory
        
        if self.oh_no[2]: 
            self.vae["enable_slicing"]  = True
        else: 
            self.vae["disable_slicing"] = True # [compatibility] serialize vae to lower memory

        return self.optimized, self.vae
    
    def _get_step_from_filename(self, key, val):
        try:
            self.get_filename = key[0]
            self.lower_filename = self.get_filename.lower()
            self.step_num_index = self.lower_filename.rindex("step")
        except ValueError as error_log:
            logger.debug(f"LoRA not named with steps {error_log}.", exc_info=True)
            return 0
        else:
            if self.step_num_index is not None:
                self.crop_steps = str(self.get_filename[self.step_num_index - 2:self.step_num_index])
                self.steps_val  = int(self.crop_steps) if self.crop_steps.isdigit() else int(self.crop_steps[1:])
                return self.steps_val
            else:
                return 0

    def prioritize_loras(self):
        self.step_priority = {}
        self.lora_priority = {}
        self.lora_priority = config.get_default("algorithms", "lora_priority")
        self.step_priority = config.get_default("algorithms", "step_priority")
        for items in self.lora_priority:
            for each in self.step_priority:
                if isinstance(each, numbers.Real):
                    for key, val in self._lora.items():
                        self.steps = self._get_step_from_filename(key, val)
                        if self.steps == each:
                            self.lora_unsorted[key] = self.steps

                        if str(items).upper() in key[1].upper():
                            self.lora_sorted[key] = val
                            if self.lora_unsorted.get(key,None) != None:
                                self.gen["num_inference_steps"] = self.lora_unsorted[key]
                                return val[1], key[1]
                            
                                        
                if next(iter(self.lora_sorted.items()), 0) != 0:
                    each, item = next(iter(self.lora_sorted.items()),0)
                    self.step_no = self._get_step_from_filename(each, item)
                    if self.step_no != 0:
                        self.gen["num_inference_steps"] = self.step_no
                    else:
                        self.gen["num_inference_steps"] = 20
                    return item[1], each[1]
            
            if next(iter(self.lora_unsorted.items()), 0) != 0:
                each, item = next(iter(self.lora_unsorted.items()),0)
                if not isinstance(val(len(val)-1), numbers.Real):
                    self.gen["num_inference_steps"] = 20
                else: 
                    self.gen["num_inference_steps"] = val(len(val)-1)
                return each[1][1], each[0][1]  
            else:
                logger.debug(f"LoRA not found?", exc_info=True)

    @cache       
    def gen_exp(self, skip):
        self.skip =  12 - int(skip)
        if self._tra != "∅" and self._tra != {}:
            i=0
            for each in self._tra:
                for tra_extension in self.extensions_list:
                    if i==0:
                        tra_filename = os.path.join("text_encoder","model" + tra_extension)
                    else:
                        tra_filename = os.path.join(f"text_encoder_{i+1}","model" + tra_extension)                        
                    link = self.symlinker(self._tra[each][1], self.category, tra_filename)
                self.optimized["transformer"][i] = link
                i+=1
                if self.first_device == "cpu":
                    self.transformers[each]["variant"] = "F32"
                else: 
                    self.transformers[each]["variant"] = self._tra[each][2]

                self.transformers[each]["num_hidden_layers"] = self.skip
                if self.spec.get("flash_attention_2",False)  == True: 
                    self.transformers[each]["attn_implementation"] = "flash_attention_2" #dont add unless necessary
        else:
                for tra_extension in self.extensions_list:
                    tra_filename = os.path.join("text_encoder", self.sym_suffix + tra_extension)
                    link = self.symlinker(list(self.fetch)[1],self.category, tra_filename)
                self.optimized["transformer"] = link
                self.transformers[self.category]["variant"] = self.model["variant"]
                self.transformers[self.category]["num_hidden_layers"] = self.skip

        if next(iter(self._lora.items()),0) != 0:
            self.optimized["lora"], self.optimized["lora_class"] = self.prioritize_loras()
            # cfg enabled here
            if "PCM" in self.optimized["lora_class"]:
                self.optimized["algorithm"]                     = self.algorithms[5] #DDIM
                self.optimized["scheduler"]["timestep_spacing"] = "trailing"
                self.optimized["scheduler"]["set_alpha_to_one"] = False  # [compatibility]PCM False
                self.optimized["scheduler"]["clip_sample"] = False
                self.gen["guidance_scale"] = 5 if "normal" in str(self.optimized["lora"]).lower() else 2
            elif "SPO" in self.optimized["lora_class"]:
                if "STA-15" in self.optimized["lora_class"]:
                    self.gen["guidance_scale"] = 7.5
                elif "STA-XL" in self.optimized["lora_class"]:
                    self.gen["guidance_scale"] = 5
            else:
            #lora parameters
            # cfg disabled below this line
                self.gen["guidance_scale"] = 0
                if "TCD" in self.optimized["lora_class"]:
                    self.gen["num_inference_steps"] = 8
                    self.optimized["algorithm"] = self.algorithms[7] #TCD sampler
                    self.gen["eta"] = 0.3
                elif "LIG" in self.optimized["lora_class"]: #4 step model pref
                    self.optimized["algorithm"]                       = self.algorithms[0] #Euler sampler
                    self.optimized["scheduler"]["interpolation_type"] = "linear" #sgm_uniform/simple
                    self.optimized["scheduler"]["timestep_spacing"]   = "trailing"
                elif "DMD" in self.optimized["lora_class"]:
                    self.optimized["fuse_lora_on"] = False
                    self.optimized["algorithm"]                    = self.algorithms[6] #LCM
                    self.optimized["scheduler"]["timesteps"]       = [999, 749, 499, 249]
                    self.optimized["scheduler"]["use_beta_sigmas"] = True
                elif "LCM" in self.optimized["lora_class"] or "RT" in self.optimized["lora_class"]:
                    self.optimized["algorithm"]     = self.algorithms[6] #LCM
                    self.gen["num_inference_steps"] = 4
                elif "FLA" in self.optimized["lora_class"]:
                    self.optimized["algorithm"] = self.algorithms[6] #LCM
                    self.gen["num_inference_steps"] = 4
                    self.optimized["scheduler"]["timestep_spacing"] = "trailing"
                    if "STA-3" in self.optimized["lora_class"]:
                        self.optimized["algorithm"] = self.algorithms[9] #LCM
                        for each in self.transformers["file"].lower():
                            if "t5" in each:
                                for items in self.transformer:
                                    try:
                                        self.transformers[items].pop(each)
                                    except KeyError as error_log:
                                        logger.debug(f"No key for  {error_log}.", exc_info=True)
                elif "HYP" in self.optimized["lora_class"]:
                    if self.gen["num_inference_steps"] == 1:
                        self.optimized["algorithm"]               = self.algorithms[7] #TCD FOR ONE STEP
                        self.optimized["fuse_lora"]["lora_scale"] = 1.0
                        self.gen["eta"]                           = 1.0
                        if "STA-XL" in self.optimized["lora_class"]: #unet only
                            self.optimized["scheduler"]["timesteps"] = 800
                    else:
                        if "CFG" in str(self.optimized["lora"]).upper():
                            if self.category == "STA-XL":
                                self.gen["guidance_scale"] = 5
                            elif self.category == "STA-15":
                                self.gen["guidance_scale"] = 7.5                               
                        if ("FLU" in self.optimized["lora_class"]
                        or "STA-3" in self.optimized["lora_class"]):
                            self.optimized["fuse_lora"]["lora_scale"]=0.125
                        self.optimized["algorithm"]                     = self.algorithms[5] #DDIM
                        self.optimized["scheduler"]["timestep_spacing"] = "trailing"
        else :   #if no lora
            self.gen["num_inference_steps"]                  = 20
            self.gen["guidance_scale"]                       = 7
            self.optimized["scheduler"]["use_karras_sigmas"] = True
            if ("LUM" in self.category
            or "STA-3" in self.category):
                self.optimized["scheduler"]["interpolation type"] = "linear" #sgm_uniform/simple
                self.optimized["scheduler"]["timestep_spacing"]   = "trailing"
            if self.category in ["STA-15", "STA-XL", "STA3"]:
                self.optimized["algorithm"]     = self.algorithms[4] #DPMAlignYourSteps
                self.gen["algorithm_type"]      = self.solvers[0]
                self.optimized["scheduler"]["euler_at_final"] = True
                self.gen["num_inference_steps"] = 10
                if "STA-15" == self.category: 
                    self.optimized["scheduler"]["timesteps"] = AysSchedules[ "StableDiffusionTimesteps"]
                elif "STA-XL" == self.category:
                    self.optimized["scheduler"]["timesteps"] = AysSchedules["StableDiffusionXLTimesteps"]
                elif "STA-3" in self.category:
                    self.gen["guidance_scale"] = 4
                    self.optimized["scheduler"]["timesteps"] = AysSchedules["StableDiffusion3Timesteps"]
            elif "PLA" in self.category:
                self.optimized["algorithm"] = self.algorithms[3] #EDMDPM
            elif ("FLU" in self.category
            or "AUR" in self.category):
                self.optimized["algorithm"] = self.algorithms[2] #EulerAncestralAliens

        self.gen["output_type"] = "latent"
        self.gen["low_cpu_mem_usage"] = self.oh_no[0]
        if self.optimized.get("algorithm",0) == 0: self.optimized["algorithm"] = self.algorithms[0] #Euler
        return  self.transformers, self.gen, self.optimized
    
    @cache      
    def refiner_exp(self):
        self.refiner = defaultdict(dict)
        if self.optimized["refiner"] != None and self.optimized["refiner"] != "∅":
            self.refiner["use_refiner"]         = False,                           # refiner
            self.refiner["high_noise_fra"]      = 0.8,                             #end noise
            self.refiner["denoising_end"]       = self.refiner["high_noise_fra"]
            self.refiner["num_inference_steps"] = self.gen["num_inference_steps"], #begin step
            self.refiner["denoising_start"]     = self.refiner["high_noise_fra"],  #begin noise
        return self.refiner

    @cache
    def dynamo_exp(self):
        self.dynamo= defaultdict(dict)
        self.gen["return_dict"]             = False
        self.dynamo["compile"]["fullgraph"] = True
        self.dynamo["compile"]["mode"]      = "reduce-overhead" #switches to max-autotune if cuda device
        return self.dynamo
            