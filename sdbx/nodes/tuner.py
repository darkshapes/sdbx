import os
from sdbx.indexer import IndexManager
from sdbx import logger
from sdbx.config import config, Precision, cache, TensorDataType as D
from sdbx.nodes.helpers import soft_random
from collections import defaultdict
from diffusers.schedulers import AysSchedules
import psutil

class NodeTuner:
    @cache        
    def determine_tuning(self, model):
        self.spec = config.get_default("spec","data")
        self.algorithms = list(config.get_default("algorithms","schedulers"))
        self.solvers = list(config.get_default("algorithms","solvers"))
        self.sort, self.category, self.fetch = IndexManager().fetch_id(model)
        self.metadata_path = config.get_path("models.metadata")
        self.clip_skip = 2
        if self.fetch != "∅" and self.fetch != None:

            self.optimized = defaultdict(dict)
            self.model = defaultdict(dict) 
            self.queue = defaultdict(dict)
            self.transformers = defaultdict(dict)
            self.conditioning = defaultdict(dict)
            self.schedule = defaultdict(dict)
            self.gen = defaultdict(dict)

        if self.sort == "LLM":
            """
            llm stuff
            """
        
        elif (self.sort  == "VAE"
        or self.sort  == "TRA"
        or self.sort  == "LOR"):
            
            """
            say less fam
            """
        else:
            self.model["file"] = list(self.fetch)[1]
            self.model["class"] = self.category
            self._vae, self._tra, self._lora = IndexManager().fetch_compatible(self.category)
    @cache
    def pipe_exp(self):
        self.pipe = defaultdict(dict)
        if self.oh_no[0]: self.pipe["low_cpu_mem_usage"] = self.oh_no[0]
        if not self.oh_no[0]:self.pipe["variant"] = list(self.fetch)[2]
        else:self.pipe["torch_dtype"] = "auto"
        self.pipe["tokenizer"] = None
        self.pipe["text_encoder"] = None
        self.pipe["tokenizer_2"] = None
        self.pipe["text_encoder_2"] = None
        if "STA-15" in self.category: self.pipe["safety_checker"] = None
        return self.pipe, self.model
    
    @cache
    def opt_exp(self):
        peak_gpu = self.spec["gpu_ram"] #accomodate list of graphics card ram
        overhead =  list(self.fetch)[0]
        peak_cpu = self.spec["cpu_ram"]
        cpu_ceiling =  overhead/peak_cpu
        gpu_ceiling = overhead/peak_gpu
        total_ceiling = overhead/(peak_cpu+peak_gpu)

        # size?  >50%   >100%  >cpu  >cpu+gpu , overhead condition numbers
        self.oh_no = [False, False, False, False,]
        if gpu_ceiling > 1: 
            self.oh_no[1] = True #look for quant, load_in_4bit=True/load_in_8bit=True
            if total_ceiling > 1:
                self.oh_no[3] = True
            elif cpu_ceiling > 1:
                self.oh_no[2] = True
        elif gpu_ceiling < .5:
            self.oh_no[0] = True

        self.optimized["cache_jettison"] = self.oh_no[1]
        self.optimized["device"] = self.spec["devices"][0]
        if self.spec["flash_attention_2"] == True: self.optimized["attn_implementation"]="flash_attention_2"
        self.optimized["dynamic_cfg"] = False
        self.optimized["seq"] = self.oh_no[1]
        self.optimized["cpu"] = self.oh_no[2]
        self.optimized["disk"] = self.oh_no[3]
        self.optimized["file_prefix"] = "Shadowbox-"
        self.optimized["fuse_lora_on"] = False
        #self.optimized["fuse_pipe"] = True pipe.fuse_qkv_projections()
        self.optimized["compile_transformer"] = self.spec["compile_transformer"]
        #if self.spec["compile_unet"]: 
        #    self.optimized["compile_unet"] = self.spec["compile_unet"]
        #    self.optimized["sigmas"] = True
        #    self.gen["return_dict"]=False
        self.optimized["compile_vae"] = self.spec["compile_vae"]
        self.optimized["compile"]["fullgraph"] = True
        self.optimized["compile"]["mode"] = "reduce-overhead"
        self.optimized["refiner"] = IndexManager().fetch_refiner()
        self.optimized["upcast_vae"] = True if self.category == "STA-XL" else False # f32, fp16 broken by default
        skip = 12 - (self.clip_skip-1)
        self.optimized["num_hidden_layers"] = int(skip)
        return self.optimized
  
    @cache
    def cond_exp(self):
        self.conditioning["padding"] = "max_length"
        self.conditioning["truncation"] = True
        self.conditioning["return_tensors"] = 'pt'
        return self.conditioning

    @cache
    def vae_exp(self):
        # ensure value returned
        if self._vae != "∅":  
            self.vae = defaultdict(dict)
            self.optimized["vae"] = self._vae[0][1][1]
            if not self.oh_no[0]: 
                self.vae["variant"] = self._vae[0][1][2] 
            else:
                self.vae["torch_dtype"] = "auto"
            
            if self.oh_no[1]: 
                self.vae["enable_tiling"] = True
            else: 
                self.vae["disable_tiling"] = True  # [compatibility] tile vae input to lower memory
            
            if self.oh_no[2]: 
                self.vae["enable_slicing"]  = True
            else: 
                self.vae["disable_slicing"] = True # [compatibility] serialize vae to lower memory
            self.vae["cache_dir"] = "vae_"

            return self.optimized, self.vae
    
    def prioritize_loras(self):
        self.lora_unsorted = defaultdict(dict)
        self.lora_sorted = defaultdict(dict)
        self.lora = defaultdict(dict)
        self.step_priority = {}
        self.lora_priority = {}
        self.lora_priority = config.get_default("algorithms", "lora_priority")
        self.step_priority = config.get_default("algorithms", "step_priority") 
        for item in self.lora_priority:
            for each in self.step_priority:
                for key, val in self._lora.items():
                    try:
                        self.get_filename = key[0]
                        self.lower_filename = self.get_filename.lower()
                        self.step_num_index = self.lower_filename.rindex("step")
                    except ValueError as error_log:
                        logger.debug(f"LoRA not named with steps {error_log}.", exc_info=True)
                        continue  # skip to next iteration on error
                    else:
                        if self.step_num_index is not None:
                            self.crop_steps = str(key[0][self.step_num_index - 2:self.step_num_index])
                            self.steps = int(self.crop_steps) if self.crop_steps.isdigit() else int(self.crop_steps[1:])

                            if str(item) in str(key[1]):
                                self.lora_sorted[key[1]] = self.steps

                            if self.steps == each:
                                self.lora_unsorted[key[1]] = self.steps
                                if self.lora_sorted.get(key[1],None) != None:
                                    self.gen["num_inference_steps"] = each
                                    return { key[1]: val }      

        if next(iter(self.lora_sorted.items()), 0) != 0:
            key, val = zip(**self.lora_sorted.items())
            self.gen["num_inference_steps"] = val[0]
            return {key[0]: val[0]}
        
        elif next(iter(self.lora_unsorted.items()), 0) != 0:
            self.gen["num_inference_steps"] = 20
            key, val = zip(**self.lora_unsorted.items())
            return {key[0]: val[0]}         
        else:
            logger.debug(f"LoRA not found?", exc_info=True)

    @cache       
    def gen_exp(self):
        self.conditioning_list = ["tokenizer", "text_encoder", "tokenizer_2", "text_encoder_2","tokenizer_3", "text_encoder_3"]
        if self._tra != "∅" and self._tra != {}: 
                i = 0
                for each in self._tra:
                    #self.transformers["tokenizer][each] = self._tra[each][1]
                    self.transformers["text_encoder"][each] = self._tra[each][1]
                    if not self.oh_no[0]:
                        self.transformers["variant"][each] = self._tra[each][2] 
                    else: 
                        self.transformers["torch_dtype"] = "auto"
                        self.transformers["low_cpu_mem_usage"][each] = self.oh_no[0]

        if next(iter(self._lora.items()),0) != 0:
            self.lora_sorted = self.prioritize_loras()
            self.optimized["lora"] = next(iter(self.lora_sorted.items()),0)[1][1]
            self.optimized["lora_class"] = next(iter(self.lora_sorted.items()),0)[0]
            # cfg enabled here
            if "PCM" in self.optimized["lora_class"]:
                self.optimized["algorithm"] = self.algorithms[5] #DDIM
                self.optimized["scheduler"]["timestep_spacing"] = "trailing"
                self.optimized["scheduler"]["set_alpha_to_one"] = False  # [compatibility]PCM False
                self.optimized["scheduler"]["rescale_betas_zero_snr"] = True  # [compatibility] DDIM True 
                self.optimized["scheduler"]["clip_sample"] = False
                self.gen["guidance_scale"] = 5 if "normal" in str(self.optimized["lora"]).lower() else 3
            elif "SPO" in self.optimized["lora_class"]:
                if "STA-15" in self.optimized["lora_class"]:
                    self.gen["guidance_scale"] = 7.5
                elif "STA-XL" in self.optimized["lora_class"]:
                    self.gen["guidance_scale"] = 5
            else:
            #lora parameters
            # cfg disabled below this line
                self.optimized["fuse_lora_on"] = True
                self.optimized["fuse_unet_only"] = False # x todo - add conditions where this is useful
                self.gen["guidance_scale"] = 0
                if "TCD" in self.optimized["lora_class"]:
                    self.gen["num_inference_steps"] = 4
                    self.optimized["algorithm"] = self.algorithms[7] #TCD sampler
                    self.gen["eta"] = 0.3
                    self.gen["strength"] = .99
                elif "LIG" in self.optimized["lora_class"]: #4 step model pref
                    self.optimized["algorithm"] = self.algorithms[0] #Euler sampler
                    self.optimized["scheduler"]["interpolation_type"] = "linear" #sgm_uniform/simple
                    self.optimized["scheduler"]["timestep_spacing"] = "trailing"                        
                elif "DMD" in self.optimized["lora_class"]:
                    self.optimized["fuse_lora_on"] = False
                    self.optimized["algorithm"] = self.algorithms[6] #LCM
                    self.optimized["scheduler"]["timesteps"] = [999, 749, 499, 249]
                    self.optimized["scheduler"]["use_beta_sigmas"] = True         
                elif "LCM" in self.optimized["lora_class"] or "RT" in self.optimized["lora_class"]:
                    self.optimized["algorithm"] = self.algorithms[6] #LCM
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
                        self.optimized["algorithm"] = self.algorithms[7] #tcd FOR ONE STEP
                        self.optimized["fuse_lora_on"] = True
                        self.optimized["fuse_lora"]["lora_scale"]=1.0
                        self.gen["eta"] = 1.0
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
                        self.optimized["algorithm"] = self.algorithms[5] #DDIM
                        self.optimized["scheduler"]["timestep_spacing"] = "trailing"
        else :   #if no lora
            self.optimized["algorithm"] = self.algorithms[4]
            self.gen["num_inference_steps"] = 20
            self.gen["guidance_scale"] = 7
            self.optimized["scheduler"]["use_karras_sigmas"] = True
            if ("LUM" in self.category
            or "STA-3" in self.category):
                self.optimized["scheduler"]["interpolation type"] = "linear", #sgm_uniform/simple
                self.optimized["scheduler"]["timestep_spacing"] = "trailing", 
            if (self.category == "STA-15"
            or self.category == "STA-XL"
            or self.category == "STA3"):
                self.optimized["algorithm"] = self.algorithms[8] #AlignYourSteps
                if "STA-15" == self.category: 
                    self.optimized["ays"] = "StableDiffusionTimesteps"
                    self.optimized["scheduler"]["timesteps"] = AysSchedules[self.ays]
                elif "STA-XL" == self.category:
                    self.gen["num_inference_steps"] = 10
                    self.gen["guidance_scale"] = 5 
                    self.optimized["ays"] = "StableDiffusionXLTimesteps"
                    self.optimized["dynamic_cfg"] = True # half cfg @ 50-75%. xl only.no lora accels
                    self.gen["callback_on_step_end_tensor_inputs"]=['prompt_embeds', 'add_text_embeds','add_time_ids']

                elif "STA-3" in self.category:
                    self.ays = "StableDiffusion3Timesteps"
                    self.gen["num_inference_steps"] = 10
                    self.gen["guidance_scale"] = 4
            elif "PLA" in self.category:
                self.optimized["algorithm"] = self.algorithms[3] #EDMDPM
            elif ("FLU" in self.category
            or "AUR" in self.category):
                self.optimized["algorithm"] = self.algorithms[2] #EulerAncestralAliens

        self.gen["output_type"] = "latent"
        self.gen["low_cpu_mem_usage"] = self.spec["low_cpu_mem_usage"]
        return  self.transformers, self.gen, self.optimized
    
    @cache      
    def refiner_exp(self):
        self.refiner = defaultdict(dict)
        if self.optimized["refiner"] != None and self.optimized["refiner"] != "∅":
            self.refiner["use_refiner"] = False, # refiner      
            self.refiner["high_noise_fra"] = 0.8, #end noise 
            self.refiner["denoising_end"] = self.refiner["high_noise_fra"]
            self.refiner["num_inference_steps"] = self.gen["num_inference_steps"], #begin step
            self.refiner["denoising_start"] = self.refiner["high_noise_fra"], #begin noise

        return self.refiner

            