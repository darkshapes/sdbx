import os
import numbers
from sdbx.indexer import IndexManager
from sdbx import logger
from sdbx.config import config, Precision, cache, TensorDataType as D
from sdbx.nodes.helpers import soft_random
from collections import defaultdict
from diffusers.schedulers.scheduling_utils import AysSchedules


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

    @cache
    def check_memory_fit(self, peak_gpu, **file_sizes):
        """ Fit files in vram, extend memory to cpu if required.
        File sizes are in bytes.
        Return dict with flags for appropriate workflow. """

        total_size = sum(file_sizes.values())
        if total_size <= peak_gpu:
            return {"fit": "all", "memory": "gpu"}

        file_list = list(file_sizes.items())
        for i in range(len(file_list) - 1, 0, -1):
            total_size = sum(size for _, size in file_list[:i])
            if total_size <= peak_gpu:
                threshold = {"fit": file_list[i-1][0], "memory": "gpu"}
                return threshold

        import shutil
        disk_utilization = shutil.disk_usage(config.get_path("output"))
        disk_free = disk_utilization[2]
        model, model_size = file_list[-1]
        offloading = { "cpu":1073741824, "sequential":3221225472, "disk": disk_free }  # Extend to cpu offloading

        for offload_type, additional_memory in offloading.items():
            if model_size <= (peak_gpu + additional_memory):
                threshold = {"fit": "model", "memory": offload_type}
                return threshold

        return threshold

    @cache
    def system_profile(self, first_device, main_model_size, vae_size = 0, apex = False, tra_size= None):
        """ Choose a working flow specific to this system's requirements.
        This routine makes a _conservative_ estimate. To trade stability for performance,
        pass 'apex' flag. To turn off system profiling, use 'manual' in config file """

        peak_cpu =  self.spec["devices"].get("cpu",1)
        if first_device    != "cpu":
            peak_gpu = self.spec["devices"].get(self.first_device,1)  #todo: accommodate list of graphics card ram
            model_sizes_dict = {}
            model_sizes_dict["model"] = main_model_size
            if tra_size is not None:
                model_sizes_dict["tra"] = tra_size
            if vae_size is not None:
                model_sizes_dict["vae"] = vae_size
            threshold = self.check_memory_fit(peak_gpu, **model_sizes_dict)
        else:
            if main_model_size > peak_cpu:
                threshold = {"memory": "disk"}
            self.pipe_data["precision"] = "F32"
            self.transformer_data["precision"] = "F32"
            self.vae_data["precision"] = "F32"

        if threshold.get("fit",None) == "all":
            self.vae_data["torch_dtype"] = "auto"
            self.pipe_data["torch_dtype"] = "auto"

        if threshold.get("fit", None) in ["vae", "all"]:
            self.pipe_data["padding"]        = "max_length"
            self.pipe_data["truncation"]     = True
            self.pipe_data["return_tensors"] = 'pt'

        if threshold.get("fit", None) == "vae":
            self.cache_data["stage"]["generate"] = 1,1,1,0

        #treat any other case the same way
        if threshold.get("fit", None) not in [None, "all"]:
            self.cache_data["stage"]["encoder"] = 1,0,0,0
            self.cache_data["stage"]["generate"] = 0,1,1,0
            self.gen_data["output_type"] = "latent"
            self.pipe_data["transformer_models"] = None
            self.pipe_data["precision"] = "F16"
            self.vae_data["precision"] = "F16"
            self.encode_data["padding"]        = "max_length"
            self.encode_data["truncation"]     = True
            self.encode_data["return_tensors"] = 'pt'
            if self.transformers_data is not None:
                for i in range(len(self._tra)):
                    self.transformers_data[f"precision_{i}"] = "F16"

        if threshold.get("memory","gpu") != "gpu":
            self.gen_data["offload_method"] = threshold.get("memory","none")
            if  self.gen_data["offload_method"] == "sequential":
                self.cache_data["stage"]["head"] = 1,0,0,0
                self.vae_data["tiling"] = True
                self.vae_data["slicing"] = False
            elif  self.gen_data["offload_method"] == "disk":
                self.cache_data["stage"]["head"] = 1,0,0,0
                self.cache_data["stage"]["tail"] = 0,0,0,1
                self.vae_data["tiling"] = True
                self.vae_data["slicing"]  = True
            else: #if it equals cpu
                self.vae_data["tiling"] = False
                self.vae_data["slicing"] = False
        else:
            self.gen_data["offload_method"] = "none"
            self.vae_data["tiling"] = False
            self.vae_data["slicing"] = False

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
    @cache
    def prioritize_loras(self):
        self.info["lora_priority"] = config.get_default("algorithms", "lora_priority")
        self.info["step_priority"] = config.get_default("algorithms", "step_priority")
        for items in self.info["lora_priority"]:
            for each in self.info["step_priority"]:
                if isinstance(each, numbers.Real):
                    for key, val in self._lora.items():
                        self.steps = self._get_step_from_filename(key, val)
                        if self.steps == each:
                            self.info["lora_unsorted"][key] = self.steps

                        if str(items).upper() in key[1].upper():
                            self.info["lora_sorted"][key] = val
                            if self.info["lora_unsorted"].get(key,None) != None:
                                self.gen_data["num_inference_steps"] = self.info["lora_unsorted"][key]
                                return val[1], key[1]


                if next(iter(self.info["lora_sorted"].items()), 0) != 0:
                    each, item = next(iter(self.info["lora_sorted"].items()),0)
                    self.step_no = self._get_step_from_filename(each, item)
                    if self.step_no != 0:
                        self.gen_data["num_inference_steps"] = self.step_no
                    else:
                        self.gen_data["num_inference_steps"] = 20
                    return item[1], each[1]

            if next(iter(self.info["lora_unsorted"].items()), 0) != 0:
                each, item = next(iter(self.info["lora_unsorted"].items()),0)
                if not isinstance(val(len(val)-1), numbers.Real):
                    self.gen_data["num_inference_steps"] = 20
                else:
                    self.gen_data["num_inference_steps"] = val(len(val)-1)
                return each[1][1], each[0][1]
            else:
                logger.debug(f"LoRA not found?", exc_info=True)


    def symlinker(self, true_file, class_name, filename, full_path=False):
        symlink_head      = os.path.join(config.get_path("models.metadata"), class_name)
        symlink_full_path = os.path.join(symlink_head, filename)
        try:
            os.remove(symlink_full_path) 
        except FileNotFoundError:
            pass
        os.symlink(true_file, symlink_full_path)
        return symlink_head if full_path == False else symlink_full_path

    @cache
    def determine_tuning(self, model):
        self.device_data       = defaultdict(dict)
        self.cache_data        = defaultdict(dict)
        self.queue_data        = defaultdict(dict)
        self.pipe_data         = defaultdict(dict)
        self.transformers_data = defaultdict(dict)
        self.vae_data          = defaultdict(dict)
        self.lora_data         = defaultdict(dict)
        self.scheduler_data    = defaultdict(dict)
        self.gen_data          = defaultdict(dict)
        self.image_data        = defaultdict(dict)
        self.encode_data       = defaultdict(dict)
        self.refiner_data      = defaultdict(dict)
        self.tuned_defaults    = defaultdict(dict)
        self.compile_data      = defaultdict(dict)

        #dicts that do not leave the class
        self.info              = defaultdict(dict)
        self.pipe_data["model"]             = model

        self.info["model_category"], self.info["model_class"], self.info["model_details"] = config.model_indexer.fetch_id(self.pipe_data["model"])
        self.info["model_size"]             = self.info["model_details"][0]
        self.pipe_data["device"]            = self.first_device
        self.pipe_data["low_cpu_mem_usage"] = True
        self.gen_data["dynamic_guidance"]   = False
        self.pipe_data["add_watermarker"]   = False
        self.device_data["device_name"]     = self.first_device
        self.device_data["gpu_id"]          = 0
        self.lora_data["fuse_0"]            = True
        #self.gen_data["low_cpu_mem_usage"]  = True
        self.image_data["file_prefix"]      = "Shadowbox-"
        self.image_data["upcast"]       = True if self.info["model_class"] == "STA-XL" else False # f32, fp16 broken by default
        #self.pipe["device_map"] = None    #todo: work on use case
        if self.info["model_class"] in ["STA-15", "STA-XL", "STA-3M", "STA-3C", "STA-3Q"]:
            self.pipe_data["safety_checker"] = False

        self.cache_data["stage"]["head"]  = None
        self.cache_data["stage"]["encoder"]  = None
        self.cache_data["stage"]["generate"] = None
        self.cache_data["stage"]["tail"] = None

        # self.pipe_data["fuse_pipe"]     = True # todo: pipe.fuse_qkv_projections(), untested
        # self.pipe_data["fuse_unet_only"] = False # todo - add conditions where this is useful
        #self.pipe_data["unet_file"]    = list(self.info["model_details"])[1] full path to file
        self.refiner_data["model"] = config.model_indexer.fetch_refiner()
        # if self.spec.get("flash_attention", 0) == True:
        #     self.transformers_data["flash_attention"] = True
        if self.spec.get("dynamo",False):
            self.scheduler_data["sigmas"]  = [] # custom sigma data
            self.gen_data["return_dict"]   = False
            self.compile_data["fullgraph"] = True
            self.compile_data["mode"]      = "reduce-overhead" #switches to max-autotune if cuda device
        else:
            self.compile_data = None

        if self.info["model_details"] != "∅" and self.info["model_details"] != None:
            if self.info["model_category"] == "LLM":
                """LLM-specific tuning
                generally handled by llama_cpp_python
                but we can spawn the 3 llm nodes here"""

            elif self.info["model_category"] in ["VAE", "TRA", "LOR"]:
                """Handle VAE, TRA, LOR types when sent from individual nodes
                complicated in execution and maybe out of scope"""
            else:
                self.queue_data = {
                        "prompt":"A slice of a rich and delicious chocolate cake presented on a table in a palace reminiscent of Versailles",
                        "seed" : soft_random(),
                        "batch": 1,
                    }
                self._vae, self._tra, self._lora =  IndexManager().fetch_compatible(self.info["model_class"])

                if self._tra != "∅" and self._tra != {}:
                    i=0
                    for each in self._tra:
                        self.transformers_data[f"transformer_{i}"]  = os.path.basename(self._tra[each][1])
                        self.transformers_data[f"precision_{i}"]    = self._tra[each][2]
                        self.transformers_data["clip_skip"]         = 2
                        self.transformers_data["device"]            = self.first_device
                        self.transformers_data["low_cpu_mem_usage"] = True
                        self.transformers_data["flash_attention"]   = self.spec.get("flash_attention",False) #dont add param unless necessary
                        self.info["tra_size"][i] = self._tra[each][0]
                        i += 1
                else:
                        self.transformers_data = None
                        self.pipe_data["use_model_to_encode"] = True
                        self.pipe_data["num_hidden_layers"]   = self.skip
                        if self.spec.get("flash_attention",False)  == True:
                            self.pipe_data["attn_implementation"] = "flash_attention"

                if self._vae != "∅":
                    self.vae_data["vae"]               = self._vae[0][0][0]
                    self.info["vae_size"]              = self._vae[0][1][0]
                    self.vae_data["device"]            = self.first_device
                    self.vae_data["low_cpu_mem_usage"] = True
                else:
                    self.vae_data = None

        if next(iter(self._lora.items()),0) != 0:
            self.lora_data["lora_0"], self.info["lora_class"] = self.prioritize_loras()
            # cfg enabled here
            if "PCM" in self.info["lora_class"]:
                self.scheduler_data["scheduler"]        = self.algorithms[5] #DDIM
                self.scheduler_data["timestep_spacing"] = "trailing"
                self.scheduler_data["set_alpha_to_one"] = False  # [compatibility]PCM False
                self.scheduler_data["clip_sample"]      = False
                self.gen_data["guidance_scale"]         = 5 if "normal" in str(self.lora_data["lora_0"]).lower() else 2
            elif "SPO" in self.info["lora_class"]:
                if "STA-15" in self.info["lora_class"]:
                    self.gen_data["guidance_scale"] = 7.5
                elif "STA-XL" in self.info["lora_class"]:
                    self.gen_data["guidance_scale"] = 5
            else:
            #lora parameters
            # cfg disabled below this line
                self.gen_data["guidance_scale"] = 0
                if "TCD" in self.info["lora_class"]:
                    self.gen_data["num_inference_steps"] = 8
                    self.scheduler_data["scheduler"]     = self.algorithms[7] #TCD sampler
                    self.gen_data["eta"]                 = 0.3
                elif "LIG" in self.info["lora_class"]: #4 step model pref
                    self.scheduler_data["scheduler"]          = self.algorithms[0] #Euler sampler
                    self.scheduler_data["interpolation_type"] = "linear" #sgm_uniform/simple
                    self.scheduler_data["timestep_spacing"]   = "trailing"
                elif "DMD" in self.info["lora_class"]:
                    self.lora_data["fuse_0"]               = False
                    self.scheduler_data["scheduler"]       = self.algorithms[6] #LCM
                    self.scheduler_data["timesteps"]       = [999, 749, 499, 249]
                    self.scheduler_data["use_beta_sigmas"] = True
                elif "LCM" in self.info["lora_class"] or "RT" in self.info["lora_class"]:
                    self.scheduler_data["scheduler"]     = self.algorithms[6] #LCM
                    self.gen_data["num_inference_steps"] = 4
                elif "FLA" in self.info["lora_class"]:
                    self.scheduler_data["scheduler"] = self.algorithms[6] #LCM
                    self.gen_data["num_inference_steps"]    = 4
                    self.scheduler_data["timestep_spacing"] = "trailing"
                    if "STA-3" in self.info["lora_class"]:
                        self.scheduler_data["scheduler"] = self.algorithms[9] #LCM
                        for each in self.transformers["file"].lower():
                            if "t5" in each:
                                for i in self.transformers_data:
                                    try:
                                        self.transformers_data.pop(i)
                                    except KeyError as error_log:
                                        logger.debug(f"No key for  {error_log}.", exc_info=True)
                elif "HYP" in self.info["lora_class"]:
                    if self.gen_data["num_inference_steps"] == 1:
                        self.scheduler_data["scheduler"] = self.algorithms[7] #TCD FOR ONE STEP
                        self.lora_data["scale_0"]        = 1.0
                        self.gen_data["eta"]             = 1.0
                        if "STA-XL" in self.info["lora_class"]: #unet only
                            self.scheduler_data["timesteps"] = 800
                    else:
                        if "CFG" in str(self.lora_data["lora_0"]).upper():
                            if self.info["model_class"] == "STA-XL":
                                self.gen_data["guidance_scale"] = 5
                            elif self.info["model_class"] == "STA-15":
                                self.gen_data["guidance_scale"] = 7.5
                        if ("FLU" in self.info["lora_class"]
                        or "STA-3" in self.info["lora_class"]):
                            self.lora_data["scale_0"] = 0.125
                        self.scheduler_data["scheduler"]        = self.algorithms[5] #DDIM
                        self.scheduler_data["timestep_spacing"] = "trailing"
        else :   #if no lora
            self.lora_data = None
            self.gen_data["num_inference_steps"]     = 20
            self.gen_data["guidance_scale"]          = 7
            self.scheduler_data["use_karras_sigmas"] = True
            if ("LUM" in self.info["model_class"]
            or "STA-3" in self.info["model_class"]):
                self.scheduler_data["interpolation type"] = "linear" #sgm_uniform/simple
                self.scheduler_data["timestep_spacing"]   = "trailing"
            if self.info["model_class"] in ["STA-15", "STA-XL", "STA3"]:
                self.scheduler_data["scheduler"]      = self.algorithms[4] #DPMAlignYourSteps
                self.gen_data["algorithm_type"]       = self.solvers[0]
                self.scheduler_data["euler_at_final"] = True
                self.gen_data["num_inference_steps"]  = 10
                if "STA-15" == self.info["model_class"]:
                    self.scheduler_data["timesteps"] = AysSchedules[ "StableDiffusionTimesteps"]
                elif "STA-XL" == self.info["model_class"]:
                    self.scheduler_data["timesteps"] = AysSchedules["StableDiffusionXLTimesteps"]
                elif "STA-3" in self.info["model_class"]:
                    self.gen_data["guidance_scale"] = 4
                    self.scheduler_data["timesteps"] = AysSchedules["StableDiffusion3Timesteps"]
            elif "PLA" in self.info["model_class"]:
                self.scheduler_data["scheduler"] = self.algorithms[3] #EDMDPM
            elif ("FLU" in self.info["model_class"]
            or "AUR" in self.info["model_class"]):
                self.scheduler_data["scheduler"] = self.algorithms[2] #EulerAncestralAliens

        if self.scheduler_data.get("scheduler",0) == 0: self.scheduler_data["scheduler"] = self.algorithms[0] #Euler

        if self.refiner_data["model"] != None and self.refiner_data["model"] != "∅":
            self.refiner_data["use_refiner"]         = False,                           # refiner
            self.refiner_data["high_noise_fra"]      = 0.8,                             #end noise
            self.refiner_data["denoising_end"]       = self.refiner_data["high_noise_fra"]
            self.refiner_data["num_inference_steps"] = self.gen_data["num_inference_steps"], #begin step
            self.refiner_data["denoising_start"]     = self.refiner_data["high_noise_fra"],  #begin noise
        else:
            self.refiner_data = None

            manual_profile = False #todo: make this a system flag in config.toml
            if manual_profile == False:
                tra_size = sum(self.info["tra_size"].values())
                profile = self.system_profile(self.first_device, int(self.info["model_size"]), int(self.info["vae_size"]), tra_size=tra_size)

        self.tuned_defaults = {
            "noise_scheduler" : self.scheduler_data,
            "empty_cache"     : self.cache_data,
            "encode_prompt"   : self.encode_data,
            "compile_pipe"    : self.compile_data,
            "load_lora"       : self.lora_data,
            "diffusion_pipe"  : self.pipe_data,
            "load_vae_model"  : self.vae_data,
            "force_device"    : self.device_data,
            "load_transformer": self.transformers_data,
            "text_input"      : self.queue_data,
            "generate_image"  : self.gen_data,
            "autodecode"      : self.image_data,
            "load_refiner"    : self.refiner_data
        }

        name, value = zip(*self.tuned_defaults.items())
        for each in name:
            if self.tuned_defaults.get(each, None) == None:
                self.tuned_defaults.pop(each)

        return self.tuned_defaults