import os
from collections import defaultdict

import psutil
from diffusers.schedulers import AysSchedules

from sdbx import config, logger
from sdbx.config import Precision, TensorDataType as D, cache
from sdbx.nodes.helpers import soft_random


class NodeTuner:
    def __init__(self):
        self.spec = config.get_default("spec", "data")
        self.algorithms = list(config.get_default("algorithms", "schedulers"))
        self.solvers = list(config.get_default("algorithms", "solvers"))
        self.metadata_path = config.get_path("models.metadata")
        self.clip_skip = 2

    @cache
    def determine_tuning(self, model):
        self.sort, self.category, self.fetch = config.model_index.fetch_id(model)

        if self.fetch is not None:
            # Initialize dictionaries
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

            if self.sort == ModelType.LANGUAGE.value:
                # LLM-specific tuning
                pass

            elif self.sort in [ModelType.VAE.value, ModelType.TRANSFORMER.value, ModelType.LORA.value]:
                # Handle VAE, TRA, LOR types
                pass

            else:
                # Default case for other model types
                self.model["file"] = self.fetch[1]
                self.model["class"] = self.category
                self._vae, self._tra, self._lora = config.model_index.fetch_compatible(self.category)

    @cache
    def pipe_exp(self):
        self.pipe_dict["variant"] = self.fetch[2]  # dtype_or_context_length
        self.pipe_dict["tokenizer"] = None
        self.pipe_dict["text_encoder"] = None
        self.pipe_dict["tokenizer_2"] = None
        self.pipe_dict["text_encoder_2"] = None
        if "STA-15" in self.category:
            self.pipe_dict["safety_checker"] = None
        return self.pipe_dict, self.model

    @cache
    def opt_exp(self):
        peak_gpu = self.spec["gpu_ram"]  # Accommodate list of graphics card RAM
        overhead = self.fetch[0]  # size
        peak_cpu = self.spec["cpu_ram"]
        cpu_ceiling = overhead / peak_cpu
        gpu_ceiling = overhead / peak_gpu
        total_ceiling = overhead / (peak_cpu + peak_gpu)

        # Overhead condition numbers
        self.oh_no = [False, False, False, False]
        if gpu_ceiling > 1:
            self.oh_no[1] = True  # Consider quantization, load_in_4bit=True/load_in_8bit=True
            if total_ceiling > 1:
                self.oh_no[3] = True
            elif cpu_ceiling > 1:
                self.oh_no[2] = True
        elif gpu_ceiling < 0.5:
            self.oh_no[0] = True

        self.optimized["cache_jettison"] = self.oh_no[1]
        self.optimized["device"] = self.spec["devices"][0]
        self.optimized["dynamic_cfg"] = False
        self.optimized["seq"] = self.oh_no[1]
        self.optimized["cpu"] = self.oh_no[2]
        self.optimized["disk"] = self.oh_no[3]
        self.optimized["file_prefix"] = "Shadowbox-"
        self.optimized["fuse"] = False
        self.optimized["compile_unet"] = False
        self.optimized["compile"] = {"fullgraph": True, "mode": "reduce-overhead"}
        self.refiner = config.model_index.fetch_refiner()
        self.optimized["upcast_vae"] = True if self.category == "STA-XL" else False
        skip = 12 - (self.clip_skip - 1)
        self.optimized["num_hidden_layers"] = int(skip)
        return self.optimized

    @cache
    def cond_exp(self):
        # Example prompt (commented out)
        # self.queue["prompt"] = "A slice of a rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles"
        self.conditioning["padding"] = "max_length"
        self.conditioning["truncation"] = True
        self.conditioning["return_tensors"] = 'pt'
        self.num = 2

    @cache
    def vae_exp(self):
        # Ensure value returned
        if self._vae:
            self.optimized["vae"] = self._vae[0][1][1]  # Path to VAE model
            self.vae_dict["variant"] = self._vae[0][1][2]
            self.vae_dict["cache_dir"] = "vae_"
            if self.oh_no[1]:
                self.vae_dict["enable_tiling"] = True
                self.vae_dict["enable_slicing"] = True
            else:
                self.vae_dict["disable_tiling"] = True
                self.vae_dict["disable_slicing"] = True
            return self.optimized, self.vae_dict

    @cache
    def gen_exp(self):
        self.conditioning_list = ["tokenizer", "text_encoder", "tokenizer_2", "text_encoder_2", "tokenizer_3", "text_encoder_3"]
        if self._tra:
            for (filename, code), value in self._tra:
                # Assuming value is [size, path, dtype_or_context_length]
                self.transformers["text_encoder"][code] = value[1]  # Path
                self.transformers["variant"][code] = value[2]  # dtype_or_context_length

        if self._lora and len(self._lora) > 0:
            (filename, code), value = self._lora[0]
            self.lora_dict["file"] = value[1]  # Path
            self.lora_dict["class"] = code

            # Get steps from filename if possible
            try:
                self.step_title = os.path.basename(self.lora_dict["file"]).lower()
                self.step_index = self.step_title.rindex("step")
            except ValueError as error_log:
                logger.debug(f"LoRA not named with steps {error_log}.", exc_info=True)
                self.gen_dict["num_inference_steps"] = 20
            else:
                if self.step_index is not None:
                    self.isteps_str = self.step_title[self.step_index - 2:self.step_index]
                    if self.isteps_str.isdigit():
                        self.gen_dict["num_inference_steps"] = int(self.isteps_str)
                    else:
                        self.gen_dict["num_inference_steps"] = int(self.isteps_str[1:])
                else:
                    self.gen_dict["num_inference_steps"] = 20

            # CFG enabled here
            if "PCM" in self.lora_dict["class"]:
                self.optimized["scheduler"] = self.algorithms[5]  # DDIM
                self.schedule["timestep_spacing"] = "trailing"
                self.schedule["set_alpha_to_one"] = False  # PCM False
                self.schedule["rescale_betas_zero_snr"] = True  # DDIM True
                self.schedule["clip_sample"] = False
                if "normal" in self.lora_dict["file"].lower():
                    self.gen_dict["guidance_scale"] = 5
                else:
                    self.gen_dict["guidance_scale"] = 3
            elif "SPO" in self.lora_dict["class"]:
                if "STA-15" in self.lora_dict["class"]:
                    self.gen_dict["guidance_scale"] = 7.5
                elif "STA-XL" in self.lora_dict["class"]:
                    self.gen_dict["guidance_scale"] = 5
            else:
                # LoRA parameters
                # CFG disabled below this line
                self.optimized["fuse"] = True
                self.fuse["lora_scale"] = 1.0
                self.optimized["fuse_unet_only"] = False  # ToDo: Add conditions where this is useful
                self.gen_dict["guidance_scale"] = 0
                if "TCD" in self.lora_dict["class"]:
                    self.gen_dict["num_inference_steps"] = 4
                    self.optimized["scheduler"] = self.algorithms[7]  # TCD sampler
                    self.gen_dict["eta"] = 0.3
                    self.gen_dict["strength"] = 0.99
                elif "LIG" in self.lora_dict["class"]:  # 4-step model preference
                    self.optimized["scheduler"] = self.algorithms[0]  # Euler sampler
                    self.schedule["interpolation_type"] = "Linear"  # sgm_uniform/simple
                    self.schedule["timestep_spacing"] = "trailing"
                elif "DMD" in self.lora_dict["class"]:
                    self.optimized["fuse"] = False
                    self.optimized["scheduler"] = self.algorithms[6]  # LCM
                    self.schedule["timesteps"] = [999, 749, 499, 249]
                    self.schedule["use_beta_sigmas"] = True
                elif "LCM" in self.lora_dict["class"] or "RT" in self.lora_dict["class"]:
                    self.optimized["scheduler"] = self.algorithms[6]  # LCM
                    self.gen_dict["num_inference_steps"] = 4
                elif "FLA" in self.lora_dict["class"]:
                    self.optimized["scheduler"] = self.algorithms[6]  # LCM
                    self.gen_dict["num_inference_steps"] = 4
                    self.schedule["timestep_spacing"] = "trailing"
                    if "STA-3" in self.lora_dict["class"]:
                        self.optimized["scheduler"] = self.algorithms[9]  # LCM
                        for each in self.transformers.get("file", {}).keys():
                            if "t5" in each.lower():
                                for items in self.transformers:
                                    try:
                                        self.transformers[items].pop(each)
                                    except KeyError as error_log:
                                        logger.debug(f"No key for {error_log}.", exc_info=True)
                elif "HYP" in self.lora_dict["class"]:
                    if self.gen_dict.get("num_inference_steps") == 1:
                        self.optimized["scheduler"] = self.algorithms[7]  # TCD for one step
                        self.schedule["timestep_spacing"] = "trailing"
                    if "CFG" in self.lora_dict["file"].upper():
                        if self.category == "STA-XL":
                            self.gen_dict["guidance_scale"] = 5
                        elif self.category == "STA-15":
                            self.gen_dict["guidance_scale"] = 7.5
                    if "FLU" in self.lora_dict["class"] or "STA-3" in self.lora_dict["class"]:
                        self.fuse["lora_scale"] = 0.125
                    elif "STA-XL" in self.lora_dict["class"]:
                        if self.gen_dict.get("num_inference_steps") == 1:
                            self.schedule["timesteps"] = 800
                        else:
                            self.optimized["scheduler"] = self.algorithms[5]  # DDIM
                            self.schedule["timestep_spacing"] = "trailing"
                            self.gen_dict["eta"] = 1.0
        else:
            # If no LoRA
            self.optimized["scheduler"] = self.algorithms[4]
            self.gen_dict["num_inference_steps"] = 20
            self.gen_dict["guidance_scale"] = 7
            self.schedule["use_karras_sigmas"] = True
            if "LUM" in self.category or "STA-3" in self.category:
                self.schedule["interpolation type"] = "Linear"
                self.schedule["timestep_spacing"] = "trailing"
            if self.category in ["STA-15", "STA-XL", "STA3"]:
                self.optimized["scheduler"] = self.algorithms[8]  # AlignYourSteps
                if self.category == "STA-15":
                    self.optimized["ays"] = "StableDiffusionTimesteps"
                    self.schedule["timesteps"] = AysSchedules[self.optimized["ays"]]
                elif self.category == "STA-XL":
                    self.gen_dict["num_inference_steps"] = 10
                    self.gen_dict["guidance_scale"] = 5
                    self.optimized["ays"] = "StableDiffusionXLTimesteps"
                    self.optimized["dynamic_cfg"] = True  # Half CFG @ 50-75%. XL only, no LoRA accelerators
                    self.gen_dict["callback_on_step_end_tensor_inputs"] = ['prompt_embeds', 'add_text_embeds', 'add_time_ids']
                elif self.category == "STA-3":
                    self.optimized["ays"] = "StableDiffusion3Timesteps"
                    self.gen_dict["num_inference_steps"] = 10
                    self.gen_dict["guidance_scale"] = 4
            elif "PLA" in self.category:
                self.gen_dict["schedule"] = self.algorithms[3]  # EDMDPM
            elif "FLU" in self.category or "AUR" in self.category:
                self.optimized["scheduler"] = self.algorithms[2]  # EulerAncestralAliens

        self.gen_dict["output_type"] = "latent"
        return self.transformers, self.gen_dict, self.optimized, self.lora_dict, self.fuse, self.schedule

    @cache
    def refiner_exp(self):
        if self.category == "STA-XL":
            if self.refiner and self.refiner != "∅":
                self.refiner["use_refiner"] = False  # Refiner
                self.refiner["high_noise_fra"] = 0.8  # End noise
                self.refiner["denoising_end"] = self.refiner["high_noise_fra"]
                self.refiner["num_inference_steps"] = self.gen_dict["num_inference_steps"]  # Begin step
                self.refiner["denoising_start"] = self.refiner["high_noise_fra"]  # Begin noise
                return self.refiner

        return self.pipe_dict