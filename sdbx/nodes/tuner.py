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
            self.pipe_dict = defaultdict(dict)
            self.transformers = defaultdict(dict)
            self.conditioning = defaultdict(dict)
            self.lora_dict = defaultdict(dict)
            self.fuse = defaultdict(dict)
            self.schedule = defaultdict(dict)
            self.gen_dict = defaultdict(dict)
            self.refiner = defaultdict(dict)
            self.vae_dict = defaultdict(dict)

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
        self.pipe_dict["variant"] = list(self.fetch)[2]
        self.pipe_dict["tokenizer"] = None
        self.pipe_dict["text_encoder"] = None
        self.pipe_dict["tokenizer_2"] = None
        self.pipe_dict["text_encoder_2"] = None
        if "STA-15" in self.category: self.pipe_dict["safety_checker"] = None
        return self.pipe_dict, self.model
    
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
        self.optimized["dynamic_cfg"] = False
        self.optimized["seq"] = self.oh_no[1]
        self.optimized["cpu"] = self.oh_no[2]
        self.optimized["disk"] = self.oh_no[3]
        self.optimized["file_prefix"] = "Shadowbox-"
        self.optimized["fuse"] = False
        self.optimized["compile_unet"] == False
        self.optimized["compile"]["fullgraph"] = True
        self.optimized["compile"]["mode"] = "reduce-overhead"       
        self.optimized["refiner"] = IndexManager().fetch_refiner()
        self.optimized["upcast_vae"] = True if self.category == "STA-XL" else False #force vae to f32 by default, because the default vae is broken       
        skip = 12 - (self.clip_skip-1)
        self.optimized["num_hidden_layers"] = int(skip)
        return self.optimized
  
    @cache
    def cond_exp(self):
        #self.queue["prompt"] = "A slice of a rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles"
        self.conditioning["padding"] = "max_length"
        self.conditioning["truncation"] = True
        self.conditioning["return_tensors"] = 'pt'
        self.num = 2

    @cache
    def vae_exp(self):
        # ensure value returned
        if self._vae != "∅":  
            self.optimized["vae"] = self._vae[0][1][1]
            self.vae_dict["variant"] = self._vae[0][1][2]
            self.vae_dict["cache_dir"] = "vae_"
            if self.oh_no[1]: self.vae_dict["enable_tiling"] = True
            else: self.vae_dict["disable_tiling"] = True  # [compatibility] tile vae input to lower memory
            if self.oh_no[1]: self.vae_dict["enable_slicing"]  = True
            else: self.vae_dict["disable_slicing"] = True # [compatibility] serialize vae to lower memory

            return self.optimized, self.vae_dict
        
    @cache       
    def gen_exp(self):
        self.conditioning_list = ["tokenizer", "text_encoder", "tokenizer_2", "text_encoder_2","tokenizer_3", "text_encoder_3"]
        if self._tra != "∅" and self._tra != {}: 
                i = 0
                for each in self._tra:
                    #self.transformers["tokenizer][each] = self._tra[each][1]
                    self.transformers["text_encoder"][each] = self._tra[each][1]
                    self.transformers["variant"][each] = self._tra[each][2]

        if next(iter(self._lora.items()),0) != 0: 
            self.lora_dict["file"] = next(iter(self._lora.items()),0)[1][1]
            self.lora_dict["class"] = next(iter(self._lora.items()),0)[0][1]
            #get steps from filename if possible (yet to determine another reliable way)
            #print(self.lora_dict["file"])
            try:
                self.step_title = self.lora_dict["file"].lower()
                self.step  = self.step_title.rindex("step")
            except ValueError as error_log:
                logger.debug(f"LoRA not named with steps {error_log}.", exc_info=True)
                self.gen_dict["num_inference_steps"] = 20
            else:
                if self.step != None:
                    self.isteps = str(self.lora_dict["file"][self.step-2:self.step])
                    self.gen_dict["num_inference_steps"] = int(self.isteps) if self.isteps.isdigit() else int(self.isteps[1:])
                else:
                    self.gen_dict["num_inference_steps"] = 20

            # cfg enabled here
            if "PCM" in self.lora_dict["class"]:
                self.optimized["scheduler"] = self.algorithms[5] #DDIM
                self.schedule["timestep_spacing"] = "trailing"
                self.schedule["set_alpha_to_one"] = False,  # [compatibility]PCM False
                self.schedule["rescale_betas_zero_snr"] = True,  # [compatibility] DDIM True 
                self.schedule["clip_sample"] = False
                self.gen_dict["guidance_scale"] = 5 if "normal" in str(self.lora_dict["file"]).lower() else 3
            elif "SPO" in self.lora_dict["class"]:
                if "STA-15" in self.lora_dict["class"]:
                    self.gen_dict["guidance_scale"] = 7.5
                elif "STA-XL" in self.lora_dict["class"]:
                    self.gen_dict["guidance_scale"] = 5
            else:
            #lora parameters
            # cfg disabled below this line
                self.optimized["fuse"] = True
                self.fuse["lora_scale"] = 1.0
                self.optimized["fuse_unet_only"] = False # x todo - add conditions where this is useful
                self.gen_dict["guidance_scale"] = 0
                if "TCD" in self.lora_dict["class"]:
                    self.gen_dict["num_inference_steps"] = 4
                    self.optimized["scheduler"] = self.algorithms[7] #TCD sampler
                    self.gen_dict["eta"] = 0.3
                    self.gen_dict["strength"] = .99
                elif "LIG" in self.lora_dict["class"]: #4 step model pref
                    self.optimized["scheduler"] = self.algorithms[0] #Euler sampler
                    self.schedule["interpolation_type"] = "Linear", #sgm_uniform/simple
                    self.schedule["timestep_spacing"] = "trailing",                         
                elif "DMD" in self.lora_dict["class"]:
                    self.optimized["fuse"] = False
                    self.optimized["scheduler"] = self.algorithms[6] #LCM
                    self.schedule["timesteps"] = [999, 749, 499, 249]
                    self.schedule["use_beta_sigmas"] = True         
                elif "LCM" in self.lora_dict["class"] or "RT" in self.lora_dict["class"]:
                    self.optimized["scheduler"] = self.algorithms[6] #LCM
                    self.gen_dict["num_inference_steps"] = 4
                elif "FLA" in self.lora_dict["class"]:
                    self.optimized["scheduler"] = self.algorithms[6] #LCM
                    self.gen_dict["num_inference_steps"] = 4
                    self.schedule["timestep_spacing"] = "trailing"
                    if "STA-3" in self.lora_dict["class"]:
                        self.optimized["scheduler"] = self.algorithms[9] #LCM
                        for each in self.transformers["file"].lower():
                            if "t5" in each:
                                for items in self.transformer:
                                    try:
                                        self.transformers[items].pop(each)
                                    except KeyError as error_log:
                                        logger.debug(f"No key for  {error_log}.", exc_info=True)
                elif "HYP" in self.lora_dict["class"]:
                    if self.gen_dict["num_inference_steps"] == 1:
                        self.optimized["scheduler"] = self.algorithms[7] #tcd FOR ONE STEP
                        self.schedule["timestep_spacing"] = "trailing"
                    if "CFG" in str(self.lora_dict["file"]).upper():
                        if self.category == "STA-XL":
                            self.gen_dict["guidance_scale"] = 5
                        elif self.category == "STA-15":
                            self.gen_dict["guidance_scale"] = 7.5                               
                    if ("FLU" in self.lora_dict["class"]
                    or "STA-3" in self.lora_dict["class"]):
                        self.fuse["lora_scale"]=0.125
                    elif "STA-XL" in self.lora_dict["class"]:
                        if self.gen_dict["num_inference_steps"] == 1:
                            self.schedule["timesteps"] = 800
                        else:
                            self.optimized["scheduler"] = self.algorithms[5] #DDIM
                            self.schedule["timestep_spacing"] = "trailing"
                            self.gen_dict["eta"] = 1.0
        else :   #if no lora
            self.optimized["scheduler"] = self.algorithms[4]
            self.gen_dict["num_inference_steps"] = 20
            self.gen_dict["guidance_scale"] = 7
            self.schedule["use_karras_sigmas"] = True
            if ("LUM" in self.category
            or "STA-3" in self.category):
                self.schedule["interpolation type"] = "Linear", #sgm_uniform/simple
                self.schedule["timestep_spacing"] = "trailing", 
            if (self.category == "STA-15"
            or self.category == "STA-XL"
            or self.category == "STA3"):
                self.optimized["scheduler"] = self.algorithms[8] #AlignYourSteps
                if "STA-15" == self.category: 
                    self.optimized["ays"] = "StableDiffusionTimesteps"
                    self.schedule["timesteps"] = AysSchedules[self.ays]
                elif "STA-XL" == self.category:
                    self.gen_dict["num_inference_steps"] = 10
                    self.gen_dict["guidance_scale"] = 5 
                    self.optimized["ays"] = "StableDiffusionXLTimesteps"
                    self.optimized["dynamic_cfg"] = True # half cfg @ 50-75%. xl only.no lora accels
                    self.gen_dict["callback_on_step_end_tensor_inputs"]=['prompt_embeds', 'add_text_embeds','add_time_ids']

                elif "STA-3" in self.category:
                    self.ays = "StableDiffusion3Timesteps"
                    self.gen_dict["num_inference_steps"] = 10
                    self.gen_dict["guidance_scale"] = 4
            elif "PLA" in self.category:
                self.gen_dict["schedule"] = self.algorithms[3] #EDMDPM
            elif ("FLU" in self.category
            or "AUR" in self.category):
                self.optimized["scheduler"] = self.algorithms[2] #EulerAncestralAliens

        self.gen_dict["output_type"] = "latent"
        return  self.transformers, self.gen_dict, self.optimized, self.lora_dict, self.fuse, self.schedule 
    
    @cache      
    def refiner_exp(self):
        if self.category == "STA-XL": 
            if self.refiner["file"] != None and self.refiner["file"] != "∅":
                self.refiner["use_refiner"] = False, # refiner      
                self.refiner["high_noise_fra"] = 0.8, #end noise 
                self.refiner["denoising_end"] = self.refiner["high_noise_fra"]
                self.refiner["num_inference_steps"] = self.gen_dict["num_inference_steps"], #begin step
                self.refiner["denoising_start"] = self.refiner["high_noise_fra"], #begin noise
                return self.refiner

            return self.pipe_dict

            