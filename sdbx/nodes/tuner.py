import os
from sdbx.indexer import IndexManager
from sdbx import config
from sdbx.config import config
from collections import defaultdict
import psutil
import platform

class NodeTuner:
#     def __init__(self, fn):
#          self.fn = fn
#          self.name = fn.info.fname

    #@cache
    #def get_tuned(self, model):# (metadata, widget_inputs, node_manager, node_id: str,  graph: MultiDiGraph, ):
       # get_tuned(self, metadata, widget_inputs, node_manager, node_id: str,  graph: MultiDiGraph,)
        # predecessors = graph.predecessors(node_id)
        # node = graph.nodes[node_id]

    
            # for p in predecessors:
            #     pnd = graph.nodes[p]  # predecessor node data
            #     pfn = node_manager.registry[pnd['fname']]  # predecessor function
            #     p_tuned_parameters = pfn.tuner.get_tuned_parameters(pnd['widget_inputs'])[node['fname']]
            #     tuned_parameters |= p_tuned_parameters
        # self.determine_tuning()
        # """
        # # return {
        # #     "function name of node": {
        # #         "parameter name": "parameter value",
        # #         "parameter name 2": "parameter value 2"
        # #     }
        # """

    def determine_tuning(self, model): 
        self.algorithms = list(config.get_default("algorithms","schedulers"))
        self.solvers = list(config.get_default("algorithms","solvers"))
        self.category, self.fetch = IndexManager().fetch_id(model)
        self.category = list(self.category)[0]
        if self.fetch != "∅" or None:
            self.fetch = self.fetch[list(self.fetch)[0]]
            print()
            if self.category == "LLM":
                """
                do LLM things
                """
            elif (self.category == "VAE"
            or self.category == "TRA"
            or self.category == "LOR"):
                """
                say less fam
                """
            else:
                self.params = defaultdict(dict)
                self.params["pipe"]["class"] = self.category
                self.params["pipe"]["transformer"] = {}
                self.params["pipe"]["transformer"]["torch.compile"] = {}
                self.params["pipe"]["scheduler"] = {}
                self.params["pipe"]["inference_steps"] = {}
                self.params["pipe"]["refiner"] = {}

                if "LLM" not in self.params["pipe"]["class"]:
                    self.vae, self.lora, self.tra = IndexManager().fetch_compatible(self.category)
                    self.num = 2
                    
                    self.params["model"] = self.fetch
                    if self.vae != "∅":  
                        self.params["vae"] = self.vae[0][1]
                    if self.lora !=  "∅": 
                        self.params["lora"] = self.lora[0][1]
                        self.params["pipe"]["scheduler"]["timestep_spacing"] = "trailing"
                        self.params["pipe"]["scheduler"]["use_beta_sigmas"] = True
                        if "PCM" in self.params["lora"]:
                            self.params["pipe"]["scheduler"]["timesteps"] = "trailing"
                            self.params["pipe"]["scheduler"]["set_alpha_to_one"] = False,  # [compatibility]PCM False
                            self.params["pipe"]["scheduler"]["rescale_betas_zero_snr"] = True  # [compatibility] DDIM True 
                            self.params["pipe"]["scheduler"] = self.algorithms[5] #DDIM
                            self.params["pipe"]["scheduler"]["clip_sample"] = False
                        elif "LCM" in self.params["lora"]:
                            self.params["pipe"]["scheduler"] = self.algorithms[6] #LCM
                            self.params["pipe"]["inference_steps"] = """find where the steps are in the filename"""
                            self.params["pipe"]["cfg"] = 4
                        elif "HYP" in self.params["lora"]:
                            self.params["pipe"]["scheduler"] = self.algorithms[6] #LCM
                            self.params["pipe"]["inference_steps"] = """find where the steps are in the filename"""
                            self.params["pipe"]["cfg"] = """0 if cfg not in filename else use model default"""
                            if "FLU" in self.params["pipe"]["class"]:
                                self.params["pipe"]["inference_steps"] = 3.5
                                self.params["pipe"]["lora"]["fuse"] = True
                                self.params["pipe"]["lora"]["fuse"]["lora_scale"]=0.125
                            if "SD3" in self.params["pipe"]["class"]:
                                self.params["pipe"]["inference_steps"] = 5
                                self.params["pipe"]["lora"]["fuse"]["lora_scale"]=0.125
                            # if sdxl
                                # if 1 step model as determined by filename
                                    # timesteps=[800]
                                    #if unet lcm scheduler self.algorithms[6]
                                #else 
                                    # tcd scheduler self.algorithms[7]
                                    # self.params["pipe"]["eta"] = 1.0
                                    # self.params["pipe"]["inference_steps"] determined by filename
                                self.params["pipe"]["lora"]["fuse"] = True
                                self.params["pipe"]["cfg"] = 0

                        elif "TCD" in self.params["lora"]:
                            self.params["pipe"]["scheduler"] = self.algorithms[7] 
                    else :
                        if ("LUM" in self.params["pipe"]["class"]
                        or "STA-3" in self.params["pipe"]["class"]):
                            self.params["pipe"]["interpolation type"] = "Linear" #sgm_uniform/simple
                            self.params["pipe"]["scheduler"]["timesteps"] = "trailing" 
                        if (self.params["pipe"]["class"] == "STA-15"
                        or self.params["pipe"]["class"] == "STA-XL"
                        or self.params["pipe"]["class"] == "STA3"):
                            self.params["pipe"]["scheduler"] = self.algorithms[8] #AlignYourSteps
                            if "STA-15" == self.params["pipe"]["class"]: 
                                self.params["pipe"]["scheduler"]["model_ays"] = "StableDiffusionTimesteps"
                            elif "STA-XL" == self.params["pipe"]["class"]: 
                                self.params["pipe"]["inference_steps"] = 10
                                self.params["pipe"]["cfg"] = 5 
                                self.params["pipe"]["scheduler"]["model_ays"] = "StableDiffusionXLTimesteps"
                                self.params["pipe"]["dynamic_guidance"] = True # half cfg @ 50-75%. xl only.no lora accels
                            elif "STA-3" in self.params["pipe"]["class"]:
                                self.params["pipe"]["scheduler"]["model_ays"] = "StableDiffusion3Timesteps"
                                self.params["pipe"]["scheduler"] = "DPMSolverMultistepScheduler"
                                self.params["pipe"]["scheduler"]["algorithm_type"]= self.solvers[0]
                                self.params["pipe"]["scheduler"]["use_karras_sigmas"] = True 
                        elif "PLA" in self.params["pipe"]["class"]:
                            self.params["pipe"]["scheduler"] = "EDMDPMSolverMultistepScheduler2M"
                        elif ("FLU" in self.params["pipe"]["class"]
                        or "AUR" in self.params["pipe"]["class"]):
                            self.params["pipe"]["scheduler"] = "FlowMatchEulerDiscreteScheduler"

                    #3.5 flux hyper
                    if self.tra != "∅": 
                        for i in range(len(self.tra[0])-1):
                            for e in self.tra:
                                print(e)
                                self.params["tra"][i] = self.tra[i]
                                if self.tra[i] > self.params["model"][0]: larger_tra = i
                    else:
                        self.params["tra"] = self.fetch
                        larger_tra = False

                avail_vid_ram = 4294836224 #accomodate list of graphics card ram
                overhead = self.params["model"][0] if not larger_tra else larger_tra
                peak_cpu = psutil.virtual_memory().total
                cpu_ceiling =  overhead/psutil.virtual_memory().total
                peak_gpu = avail_vid_ram*.95
                gpu_ceiling = overhead/avail_vid_ram
                total_ceiling = overhead/peak_cpu+peak_gpu
                #look for quant, load_in_4bit=True >75

                oh_no = [False, False, False, False,]
                if gpu_ceiling > 1: 
                    oh_no[2] = True
                    if total_ceiling > 1:
                        oh_no[0] = True
                    elif cpu_ceiling > 1:
                        oh_no[1] = True
                self.params["pipe"]["vae_tile"] = oh_no[2]   # [compatibility] tile vae input to lower memory
                self.params["pipe"]["vae_slice"] = oh_no[1]  # [compatibility] serialize vae to lower memory
                self.params["pipe"]["cache_jettison"] = oh_no[2]
                self.params["pipe"]["cpu_offload"] = oh_no[1]
                self.params["pipe"]["sequential_offload"] = oh_no[0]
                self.params["pipe"]["max_batch"] = oh_no[1]
                for i in range(len(self.params["tra"])):
                    print(self.params["model"])
                    print(self.params["tra"])
                    print(self.params["tra"][i])

                    self.params["pipe"]["transformer"][i]= self.params["tra"][i][1]

                self.params["pipe"]["vae_config"] = os.path.join(config.get_path("models.metadata"),"vae",self.category)
                self.params["pipe"]["compile_unet"] = False #if unet and possibly only on higher-end cards #[performance] unet only, compile the model for speed, slows first gen only 
                self.params["pipe"]["transformer"]["torch.compile"]["mode"] = "reduce-overhead"
                self.params["pipe"]["transformer"]["torch.compile"]["fullgraph"] = True

                #import torch / torch.cuda.mem_get_info()

                if self.params["pipe"]["class"] == "STA-XL": 
                    self.params["pipe"]["upcast_vae"] = True #force vae to f32 by default, because the default vae is broken
                    self.params["pipe"]["compile_unet"] = False #if unet #[performance] unet only, compile the model for speed, slows first gen only                   
                    self.params["pipe"]["refiner"]["use_refiner"] = False # refiner
                    self.params["pipe"]["refiner"]["high_noise_fra"] = 0.8
                    self.params["pipe"]["refiner"]["denoising_end"]= self.params["pipe"]["refiner"]["high_noise_fra"]
                    self.params["pipe"]["refiner"]["num_inference_steps"]= self.params["pipe"]["inference_steps"]
                    self.params["pipe"]["refiner"]["denoising_start"] = self.params["pipe"]["refiner"]["high_noise_fra"]
                
                # if platform.system().lower() == 'linux':
                #     if "cuda" in device:
                #         torch.cuda.memory._record_memory_history() # [diagnostic] mem use measurement

                return self.params

#index = IndexManager().write_index()
path = config.get_path("models.image")
for each in os.listdir(path):
    if not os.path.isdir(each):
        #full_path=os.path.join(path,each)
        default = NodeTuner().determine_tuning(each)
        #print(default)
# var_list = [
#     "__doc__",
#     "__name__",
#     "__package__",
#     "__loader__",
#     "__spec__",
#     "__annotations__",
#     "__builtins__",
#     "__file__",
#     "__cached__",
#     "config",
#     "indexer",
#     "json",
#     "os",
#     "defaultdict",
#     "IndexManager",
#     "logger",
#     "psutil",
#     "var_list",
#     "i"
#     ]
# variables = dict(locals())
# for each in variables:
#     if each not in var_list:
#         print(f"{each} = {variables[each]}")
#print(default.keys())




            