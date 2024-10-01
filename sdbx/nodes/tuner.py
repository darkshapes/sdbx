import os
from sdbx.indexer import IndexManager
from sdbx import logger
from sdbx.config import config, Precision, cache, TensorDataType as D
from sdbx.nodes.helpers import soft_random
from collections import defaultdict
import psutil

class NodeTuner:

    @cache        
    def determine_tuning(self, model): 
        self.spec = config.get_default("spec","data")
        self.algorithms = list(config.get_default("algorithms","schedulers"))
        self.solvers = list(config.get_default("algorithms","solvers"))
        self.sort, self.category, self.fetch = IndexManager().fetch_id(model)
        #print(self.sort, self.category, self.fetch)
        if self.fetch != "∅" and self.fetch != None:
            #print(self.category)
            self.params = defaultdict(dict)
            self.params["transformer"]["file"] = {}
            self.params["transformer"]["size"] = {}
            self.params["transformer"]["dtype"] = {}
            self.params["transformer"]["use_fast"] = {}
            self.params["transformer"]["config"] = {}
            self.params["refiner"] = {}
            self.params["negative"] = {}
            self.params["compile"]["unet"] = {}
            self.params["model"]["file"] = list(self.fetch)[1]
            self.params["model"]["size"] = list(self.fetch)[0]
            self.params["model"]["dtype"] = list(self.fetch)[2]
            self.params["model"]["stage"] = self.sort
            self.params["model"]["class"] = self.category

         
            peak_gpu = self.spec["gpu_ram"] #accomodate list of graphics card ram
            overhead = self.params["model"]["size"]
            peak_cpu = self.spec["cpu_ram"]
            cpu_ceiling =  overhead/peak_cpu
            gpu_ceiling = overhead/peak_gpu
            total_ceiling = overhead/peak_cpu+peak_gpu
            
            #give a number to each overhead condition
            # size?  >50%   >100%  >cpu  >cpu+gpu
            oh_no = [False, False, False, False,]
            if gpu_ceiling > 1: 
                #look for quant, load_in_4bit=True >75
                oh_no[1] = True
                if total_ceiling > 1:
                    oh_no[3] = True
                elif cpu_ceiling > 1:
                    oh_no[2] = True
            elif gpu_ceiling < .5:
                oh_no[0] = True
            
            self.params["pipe"]["cache_jettison"] = oh_no[3]
            self.params["pipe"]["cpu_offload"] = oh_no[2]
            self.params["pipe"]["sequential_offload"] = oh_no[1]
            self.params["pipe"]["max_batch"] = oh_no[2]
            self.params["pipe"]["seed"] = int(soft_random())
            self.params["pipe"]["low_cpu_mem_usage"] = oh_no[0]#
            self.params["pipe"]["dynamic_cfg"] = False  
            self.params["pipe"]["file_prefix"] = "Shadowbox-"

            if self.params["model"]["stage"] == "LLM":
                self.params["model"]["context"] = self.params["model"].pop("dtype")
                self.params["model"]["top_p"] = .95
                self.params["model"]["top_k"] = 40
                self.params["model"]["repeat_penalty"] =1
                self.params["model"]["temperature"] = 0.2
                self.params["model"]["max_tokens"] = 256
                self.params["model"]["streaming"] = True   
                self.params["pipe"]["prompt"] = "Tell a story about-"
                self.params["pipe"]["system_prompt"] = "You are an assistant who gives an accurate and concise reply to the best of your ability."         

            elif (self.params["model"]["stage"] == "VAE"
            or self.params["model"]["stage"] == "TRA"
            or self.params["model"]["stage"] == "LOR"):
                
                """
                say less fam
                """
            else:
                self.params["pipe"]["config_path"] = os.path.join(config.get_path("models.metadata"),self.params["model"]["class"])
                self.params["model"]["config"] = os.path.join(self.params["model"]["class"],"model.safetensors")
                self.params["model"]["yaml"] = os.path.join(config.get_path("models.metadata"),self.params["model"]["class"])
                self.params["pipe"]["prompt"] = "A slice of a rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles"
                self.vae,  self.tra, self.lora = IndexManager().fetch_compatible(self.params["model"]["class"])
                self.num = 2

                # ensure value returned
                if self.vae != "∅":  
                    self.params["vae"]["file"] = self.vae[0][1][1]
                    self.params["vae"]["size"] = self.vae[0][1][0]
                    self.params["vae"]["dtype"] = self.vae[0][1][2]
                    self.params["vae"]["tile"] = oh_no[1]   # [compatibility] tile vae input to lower memory
                    self.params["vae"]["slice"] = oh_no[2]  # [compatibility] serialize vae to lower memory
                self.params["vae"]["config"] = os.path.join(config.get_path("models.metadata"),self.params["model"]["class"],"config.json")
                self.params["vae"]["upcast"] = True if self.params["model"]["class"] == "STA-XL" else False #force vae to f32 by default, because the default vae is broken

                if self.tra != "∅" and self.tra != {}: 
                        for each in self.tra:
                            self.params["transformer"]["file"][each] = self.tra[each][1]
                            self.params["transformer"]["size"][each] = self.tra[each][0]
                            self.params["transformer"]["dtype"][each] = self.tra[each][2]
                            self.params["transformer"]["use_fast"][each] = True
                            self.params["transformer"]["config"][each] = os.path.join(each,"model.fp16.safetensors") if self.params["transformer"]["dtype"][each] == "F16" else os.path.join(each,"model.safetensors")

                    
                if next(iter(self.lora.items()),0) != 0: 
                    self.params["lora"]["file"] = next(iter(self.lora.items()),0)[1][1]
                    self.params["lora"]["size"] = next(iter(self.lora.items()),0)[1][0] 
                    self.params["lora"]["dtype"] = next(iter(self.lora.items()),0)[1][2] 
                    self.params["lora"]["class"] = next(iter(self.lora.items()),0)[0][1]
                    #get steps from filename if possible (yet to determine another reliable way)
                    print(self.params["lora"]["file"])
                    try:
                        self.step_title = self.params["lora"]["file"].lower()
                        self.step  = self.step_title.rindex("step")
                    except ValueError as error_log:
                        logger.debug(f"LoRA not named with steps {error_log}.", exc_info=True)
                        self.params["pipe"]["num_inference_steps"] = 20

                    else:
                        if self.step != None:
                            self.isteps = str(self.params["lora"]["file"][self.step-2:self.step])
                            self.params["pipe"]["num_inference_steps"] = int(self.isteps) if self.isteps.isdigit() else int(self.isteps[1:])
                        else:
                            self.params["pipe"]["num_inference_steps"] = 20
                    self.params["scheduler"]["config"] = self.params["pipe"]["config_path"]
                    self.params["scheduler"]["timestep_spacing"] ="linspace"                      
                    self.params["scheduler"]["use_beta_sigmas"] = True
                    self.params["scheduler"]["lu_lambdas"]= False
                    self.params["scheduler"]["euler_at_final"] = False
                    #get lora prams

                    # cfg enabled here
                    self.params["lora"]["unet_only"] = False # x todo - add conditions where this is useful
                    self.params["lora"]["fuse"] = False
                    if "PCM" in self.params["lora"]["class"]:
                        self.params["pipe"]["algorithm"] = self.algorithms[5] #DDIM
                        self.params["scheduler"]["timestep_spacing"] = "trailing"
                        self.params["scheduler"]["set_alpha_to_one"] = False  # [compatibility]PCM False
                        self.params["scheduler"]["rescale_betas_zero_snr"] = True  # [compatibility] DDIM True 
                        self.params["scheduler"]["clip_sample"] = False
                        self.params["pipe"]["cfg"] = 5 if "normal" in str(self.params["lora"]["file"]).lower() else 3
                    elif "SPO" in self.params["lora"]["class"]:
                        if "STA-15" in self.params["lora"]["class"]:
                            self.params["pipe"]["cfg"] = 7.5
                        elif "STA-XL" in self.params["lora"]["class"]:
                            self.params["pipe"]["cfg"] = 5 
                    else:
                    #lora parameters
                    # cfg disabled below this line
                        self.params["pipe"]["cfg"] = 0
                        self.params["lora"]["fuse"] = True
                        self.params["lora"]["scale"] = 1.0
                        self.params["lora"]["unet_only"] = False # x todo - add conditions where this is useful
                        if "TCD" in self.params["lora"]["class"]:
                            self.params["pipe"]["num_inference_steps"] = 4
                            self.params["pipe"]["algorithm"] = self.algorithms[7] #TCD sampler
                            self.params["pipe"]["noise_eta"] = 0.3
                            self.params["pipe"]["strength"] = .99
                        elif "LIG" in self.params["lora"]["class"]: #4 step model pref
                            self.params["pipe"]["algorithm"] = self.algorithms[0] #Euler sampler
                            self.params["scheduler"]["interpolation_type"] = "Linear" #sgm_uniform/simple
                            self.params["scheduler"]["timestep_spacing"] = "trailing"                         
                        elif "DMD" in self.params["lora"]["class"]:
                            self.params["lora"]["fuse"] = False
                            self.params["pipe"]["algorithm"] = self.algorithms[6] #LCM
                            self.params["scheduler"]["timesteps"] = [999, 749, 499, 249]
                            self.params["scheduler"]["use_beta_sigmas"] = True         
                        elif "LCM" in self.params["lora"]["class"] or "RT" in self.params["lora"]["class"]:
                            self.params["pipe"]["algorithm"] = self.algorithms[6] #LCM
                            self.params["pipe"]["num_inference_steps"] = 4
                        elif "FLA" in self.params["lora"]["class"]:
                            self.params["pipe"]["algorithm"] = self.algorithms[6] #LCM
                            self.params["pipe"]["num_inference_steps"] = 4
                            self.params["scheduler"]["timestep_spacing"] = "trailing"
                            if "STA-3" in self.params["lora"]["class"]:
                                self.params["pipe"]["algorithm"] = self.algorithms[9] #LCM
                                for each in self.params["transformer"]["file"].lower():
                                    if "t5" in each:
                                        for items in self.params["transformer"]:
                                            self.params["transformer"][items].pop(each)
                        elif "HYP" in self.params["lora"]["class"]:
                            if self.params["pipe"]["num_inference_steps"] == 1:
                                self.params["pipe"]["algorithm"] = self.algorithms[7] #tcd FOR ONE STEP
                                self.params["scheduler"]["timestep_spacing"] = "trailing"
                            if "CFG" in str(self.params["lora"]["file"]).upper():
                                if self.params["model"]["class"] == "STA-XL":
                                    self.params["pipe"]["cfg"] = 5
                                elif self.params["model"]["class"] == "STA-15":
                                    self.params["pipe"]["cfg"] = 7.5                                   
                            if ("FLU" in self.params["lora"]["class"]
                            or "STA-3" in self.params["lora"]["class"]):
                                self.params["lora"]["scale"]=0.125
                            elif "STA-XL" in self.params["lora"]["class"]:
                                if self.params["pipe"]["num_inference_steps"] == 1:
                                    self.params["scheduler"]["timesteps"] = 800
                                else:
                                    self.params["pipe"]["algorithm"] = self.algorithms[5] #DDIM
                                    self.params["scheduler"]["timestep_spacing"] = "trailing"
                                    self.params["pipe"]["noise_eta"] = 1.0       

                else :   #if no lora
                    if ("LUM" in self.params["model"]["class"]
                    or "STA-3" in self.params["model"]["class"]):
                        self.params["scheduler"]["interpolation type"] = "Linear" #sgm_uniform/simple
                        self.params["scheduler"]["timestep_spacing"] = "trailing" 
                    if (self.params["model"]["class"] == "STA-15"
                    or self.params["model"]["class"] == "STA-XL"
                    or self.params["model"]["class"] == "STA3"):
                        self.params["pipe"]["algorithm"] = self.algorithms[8] #AlignYourSteps
                        if "STA-15" == self.params["model"]["class"]: 
                            self.params["scheduler"]["model_ays"] = "StableDiffusionTimesteps"
                        elif "STA-XL" == self.params["model"]["class"]:
                            self.params["pipe"]["num_inference_steps"] = 10
                            self.params["pipe"]["cfg"] = 5 
                            self.params["scheduler"]["model_ays"] = "StableDiffusionXLTimesteps"
                            self.params["pipe"]["dynamic_cfg"] = True # half cfg @ 50-75%. xl only.no lora accels
                        elif "STA-3" in self.params["model"]["class"]:
                            self.params["scheduler"]["lu_lambdas"]= True
                            self.params["scheduler"]["euler_at_final"] = True
                            self.params["scheduler"]["model_ays"] = "StableDiffusion3Timesteps"
                            self.params["pipe"]["algorithm"] = "AysSchedules"
                            self.params["pipe"]["algorithm_type"]= self.solvers[4] # DPM
                            self.params["scheduler"]["use_karras_sigmas"] = True 
                    elif "PLA" in self.params["model"]["class"]:
                        self.params["pipe"]["algorithm"] = self.algorithms[3] #EDMDPM
                    elif ("FLU" in self.params["model"]["class"]
                    or "AUR" in self.params["model"]["class"]):
                        self.params["pipe"]["algorithm"] = self.algorithms[2] #EulerAncestral (Ancient Aliens)

                self.params["compile"]["unet"] == True
                self.params["compile"]["fullgraph"] = True
                self.params["compile"]["mode"] = "reduce-overhead"  #if unet and possibly only on higher-end cards #[performance] unet only, compile the model for speed, slows first gen only 

                if self.params["model"]["class"] == "STA-XL": 
                    self.params["refiner"]["available"] = IndexManager().fetch_refiner()
                    if self.params["refiner"]["available"] != None and self.params["refiner"]["available"] != "∅":
                        self.params["refiner"]["use_refiner"] = False # refiner      
                        self.params["refiner"]["high_noise_fra"] = 0.8 #end noise 
                        self.params["refiner"]["denoising_end"] = self.params["pipe"]["refiner"]["high_noise_fra"]
                        self.params["refiner"]["num_num_inference_steps"] = self.params["pipe"]["num_inference_steps"] #begin step
                        self.params["refiner"]["denoising_start"] = self.params["pipe"]["refiner"]["high_noise_fra"] #begin noise
                    else: 
                        self.params["refiner"]["available"] = False
            
                return self.params


            