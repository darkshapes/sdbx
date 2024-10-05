"""
Credits:
Felixsans
"""

import gc
import os
import torch
import datetime
from time import perf_counter_ns
from diffusers import AutoPipelineForText2Image, AutoencoderKL
from diffusers.schedulers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    EDMDPMSolverMultistepScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    LCMScheduler,
    TCDScheduler,
    AysSchedules,          
    HeunDiscreteScheduler,
    UniPCMultistepScheduler,
    LMSDiscreteScheduler,
    DEISMultistepScheduler,
     )
from diffusers.utils import logging as df_log
from transformers import logging as tf_log
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import accelerate
from sdbx import logger
from sdbx.nodes.tuner import NodeTuner
#from sdbx.indexer import IndexManager
from sdbx.config import config
from sdbx.nodes.helpers import seed_planter, soft_random


class T2IPipe:
    #do not put an __init__ in this class! only __call__ can be used. https://huggingface.co/docs/diffusers/main/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForText2Image
    bug_off=False
    config_path = config.get_path("models.metadata")

############## TIMECODE
    def tc(self, clock, string, debug=False): return print(f"[ {str(datetime.timedelta(milliseconds=(((perf_counter_ns()-clock)*1e-6))))[:-2]} ] {string}") if not debug else print("", end="")

############## STFU HUGGINGFACE
    def hf_log(self, on=False, fatal=False):
        if on:
            tf_log.enable_default_handler()
            df_log.enable_default_handler()
            tf_log.set_verbosity_warning()
            df_log.set_verbosity_warning()
        if fatal:
            TORCH_LOGS="-all"
            tf_log.disable_default_handler()
            df_log.disable_default_handler()
            tf_log.set_verbosity(tf_log.FATAL)
            tf_log.set_verbosity(df_log.FATAL)
        else:
            tf_log.set_verbosity_error()
            df_log.set_verbosity_error()

############## TORCH DATATYPE
    def float_converter(self, old_index):
        float_chart = {
                "F64": ["fp64", torch.float64],
                "F32": ["fp32", torch.float32],
                "F16": ["fp16", torch.float16],
                "BF16": ["bf16", torch.bfloat16],
                "F8_E4M3": ["fp8e4m3fn", torch.float8_e4m3fn],
                "F8_E5M2": ["fp8e5m2", torch.float8_e5m2],
                "I64": ["i64", torch.int64],
                "I32": ["i32", torch.int32],
                "I16": ["i16", torch.int16],
                "I8": ["i8", torch.int8],
                "U8": ["u8", torch.uint8],                                       
                "NF4": ["nf4", "nf4"],
        }
        for key, val in float_chart.items():
            if old_index == key:
                return val[0], val[1]

############## SCHEDULER
    def algorithm_converter(self, non_constant, exp):
        self.non_constant = non_constant
        self.exp = exp
        self.schedule_chart = {
            "EulerDiscreteScheduler" : EulerDiscreteScheduler.from_config(self.pipe.scheduler.config,**exp),           
            "EulerAncestralDiscreteScheduler" : EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config,**exp),
            "FlowMatchEulerDiscreteScheduler" : FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config,**exp),     
            "EDMDPMSolverMultistepScheduler" : EDMDPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config,**exp), 
            "DPMSolverMultistepScheduler" : DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config,**exp),        
            "DDIMScheduler" : DDIMScheduler.from_config(self.pipe.scheduler.config,**exp),
            "LCMScheduler" : LCMScheduler.from_config(self.pipe.scheduler.config,**exp),
            "TCDScheduler" : TCDScheduler.from_config(self.pipe.scheduler.config,**exp),
            "AysSchedules": AysSchedules,
            "HeunDiscreteScheduler" : HeunDiscreteScheduler.from_config(self.pipe.scheduler.config,**exp),
            "UniPCMultistepScheduler" : UniPCMultistepScheduler.from_config(self.pipe.scheduler.config,**exp),
            "LMSDiscreteScheduler" : LMSDiscreteScheduler.from_config(self.pipe.scheduler.config,**exp),  
            "DEISMultistepScheduler" : DEISMultistepScheduler.from_config(self.pipe.scheduler.config,**exp),
        }
        if self.non_constant in self.schedule_chart:
            self.pipe.scheduler = self.schedule_chart[self.non_constant]
            return self.pipe.scheduler
        else:
            try:
                raise ValueError(f"Scheduler '{self.non_constant}' not supported")
            except ValueError as error_log:
                logger.debug(f"Scheduler error {error_log}.", exc_info=True)


############## DEVICE
    def set_device(self, device=None):
        self.clock = perf_counter_ns() # 00:00:00
        self.tc(self.clock, " ")
        if device==None:
            if torch.cuda.is_available(): self.device = "cuda" # https://pytorch.org/docs/stable/torch_cuda_memory.html
            else: self.device = "mps" if (torch.backends.mps.is_available() & torch.backends.mps.is_built()) else "cpu"# https://pytorch.org/docs/master/notes/mps.html
        else: self.device = device

############## QUEUE
    def queue_manager(self, prompt, seed):
        self.tc(self.clock, "determining device type...", self.bug_off)
        self.queue = []    

        self.queue.extend([{
            "prompt": prompt,
            "seed": seed,
            }])
        
############## ENCODERS
    def declare_encoders(self, exp, transformer="stabilityai/stable-diffusion-xl-base-1.0"):
        tformer, gen, self.enc_opt = exp
        tformer_dict = {}
        if tformer.get("variant",0):
            var, dtype = self.float_converter(tformer["variant"][next(iter(tformer["variant"]),0)])
            tformer_dict.setdefault("variant",var)
            tformer_dict.setdefault("torch_dtype", dtype)
        tformer_dict.setdefault("num_hidden_layers",self.enc_opt["num_hidden_layers"])
        if self.enc_opt.get("attn_implementation",0): tformer_dict.setdefault("attn_implementation", self.enc_opt["attn_implementation"])

        self.tokenizer = CLIPTokenizer.from_pretrained(
            transformer,
            subfolder='tokenizer',
        )
        self.hf_log(fatal=True) #suppress layer skip messages
        self.text_encoder = CLIPTextModel.from_pretrained(
            transformer,
            subfolder='text_encoder',
            use_safetensors=True,
            **tformer_dict,
        ).to(self.device)
        self.hf_log(on=True) #return to normal
      
        if self.enc_opt.get("compile_transformer",0): self.compile_model(self.text_encoder, self.enc_opt["compile"])

        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            transformer,
            subfolder='tokenizer_2',
        )

        self.hf_log(fatal=True) #suppress layer skip messages
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            transformer,
            subfolder='text_encoder_2',
            use_safetensors=True,
            **tformer_dict,
        ).to(self.device)
        self.hf_log(on=True) #return to normal
        if self.enc_opt.get("compile_transformer",0): self.compile_model(self.text_encoder_2, self.enc_opt["compile"])

############## EMBEDDINGS
    def generate_embeddings(self, prompts, tokenizers, text_encoders, exp):
        self.tc(self.clock, f"encoding prompt with device: {self.device}...", self.bug_off)
        embeddings_list = []
        #for prompt, tokenizer, text_encoder in zip(prompts, self.tokenizer.values(), self.text_encoder.values()):
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            cond_input = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            **exp,
        )
            prompt_embeds = text_encoder(cond_input.input_ids.to(self.device), output_hidden_states=True)

            pooled_prompt_embeds = prompt_embeds[0]
            embeddings_list.append(prompt_embeds.hidden_states[-2])

            prompt_embeds = torch.concat(embeddings_list, dim=-1)

        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(1 * 1, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

############## PROMPT
    def encode_prompt(self, exp):
        self.exp = exp
        with torch.no_grad():
            for generation in self.queue:
                generation['embeddings'] = self.generate_embeddings(
                    [generation['prompt'], generation['prompt']],
                    [self.tokenizer, self.tokenizer_2],
                    [self.text_encoder, self.text_encoder_2], self.exp
                    )

############## CACHE MANAGEMENT
    def cache_jettison(self, encoder=False, lora=False, unet=False, vae=False):
        self.tc(self.clock, f"empty cache...", self.bug_off)
        if encoder: del self.tokenizer, self.text_encoder,  self.tokenizer_2, self.text_encoder_2, self.exp
        if lora: self.pipe.unload_lora_weights(), self.exp
        if unet: del self.pipe.unet
        if vae: del self.pipe.vae
        gc.collect()
        if self.device == "cuda": torch.cuda.empty_cache()
        if self.device == "mps": torch.mps.empty_cache()
 

############## PIPE
    def construct_pipe(self, exp, model="stabilityai/stable-diffusion-xl-base-1.0"):
        self.model = model
        self.exp, custom_model = exp
        if self.exp.get("variant",0):
            var, dtype = self.float_converter(self.exp["variant"])
            self.exp["variant"] = var
            self.exp.setdefault("torch_dtype", dtype)
        self.tc(self.clock, f"precision set for: {var}, using {dtype}", self.bug_off)

        ###### going to need this code soon...

        #symlnk_path = os.path.join(self.config_path,model["class"])
        #checkpoint = os.path.basename(custom_model["file"])
        #symlnk_path = model["file"].replace(checkpoint,"")
        #symlink_config = os.path.join(symlnk_path,"config.json")
        #symlnk_file = os.path.join(symlnk_path,"diffusion_pytorch_model.fp16.safetensors") if var == "fp16" else os.path.join(symlnk_path,"diffusion_pytorch_model.safetensors")
        
        self.tc(self.clock, f"load model {os.path.basename(self.model)}...", self.bug_off)
        #if os.path.isfile(symlnk_file): os.remove(symlnk_file)
        #os.symlink(checkpoint,symlnk_file) #note: no 'i' in 'symlnk'
        self.pipe = AutoPipelineForText2Image.from_pretrained(model, **self.exp).to(self.device)


############## LORA
    def add_lora(self, exp, fuse, opt):
        self.exp = exp
        self.lora_opt = opt
        lora = os.path.basename(self.exp)
        lora_path = self.exp.replace(lora,"")
        self.tc(self.clock, f"set lora to {lora}", self.bug_off)  # lora2
        self.pipe.load_lora_weights(lora_path, weight_name=lora)
        if fuse: self.pipe.fuse_lora(**self.lora_opt) #add unet only possibility

############## MEMORY OFFLOADING
    def offload_to(self, seq=False, cpu=False, disk=False):
        self.tc(self.clock, f"set offload as {cpu|disk} and sequential as {seq} for {self.device} device", self.bug_off) 
        if self.device=="cuda":
            if seq: self.pipe.enable_sequential_cpu_offload()
            if cpu: self.pipe.enable_model_cpu_offload() 
        if disk: accelerate.disk_offload() 

 ### cue lag spike

############## COMPILE
    def compile_model(self, model, exp):
        self.model = model
        if "cuda" in self.device:
            exp.setdefault("mode","max-autotune")
        self.model = torch.compile(self.model, **exp)
        return self.model

############## CFG CUTOFF
    def _dynamic_guidance(self, pipe, step_index, timestep, callback_key):
        if step_index == int(pipe.num_timesteps * 0.5):
            callback_key['prompt_embeds'] = callback_key['prompt_embeds'].chunk(2)[-1]
            callback_key['add_text_embeds'] = callback_key['add_text_embeds'].chunk(2)[-1]
            callback_key['add_time_ids'] = callback_key['add_time_ids'].chunk(2)[-1]
            pipe._guidance_scale = 0.0
        return callback_key
    
############## CFG CUTOFF CALLBACK
    def add_dynamic_cfg(self):
            self.tc(self.clock, "set dynamic cfg")
            self.gen_dict.setdefault("callback_on_step_end",self._dynamic_guidance)
            self.gen_dict.setdefault("callback_on_step_end_tensor_inputs",['prompt_embeds', 'add_text_embeds','add_time_ids'])

############## INFERENCE
    def diffuse_latent(self, exp):
        tformer, self.gen_dict, self.gen_opt = exp
        self.debugger(locals())
        self.tc(self.clock, f"set scheduler", self.bug_off)
        self.pipe.scheduler = self.algorithm_converter(self.gen_opt["algorithm"], self.gen_opt["scheduluer"])
   
        
        if self.gen_opt.get("lora",0): self.add_lora(self.gen_opt["lora"], self.gen_opt["fuse_lora_on"], self.gen_opt["fuse_lora"])
        if self.gen_opt.get("dynamic_cfg",0): self.add_dynamic_cfg()
        if self.gen_opt.get("compile_unet",0): self.compile_model(self.pipe.unet, self.gen_opt["compile"])

        self.tc(self.clock, f"set generator", self.bug_off) 
        generator = torch.Generator(device=self.device)
        self.tc(self.clock, "activating device. this may take a moment...")
        if self.gen_opt.get("seq",0) or self.gen_opt.get("cpu",0) or self.gen_opt.get("disk",0): self.offload_to(self.gen_opt["seq"], self.gen_opt["cpu"], self.gen_opt["disk"])

        self.tc(self.clock, f"entering loop...")
        self.image_start = perf_counter_ns()
        self.individual_totals = []
        for i, generation in enumerate(self.queue, start=1):
            self.tc(self.clock, f"planting seed {generation['seed']}...", self.bug_off)
            seed_planter(generation['seed'])
            generator.manual_seed(generation['seed'])
            self.individual_start = perf_counter_ns()
            self.tc(self.clock, f"inference device: {self.device}....", self.bug_off)
            self.tc(self.image_start, f"{i} of {len(self.queue)}", self.bug_off) 
            generation['latents'] = self.pipe(
                prompt_embeds=generation['embeddings'][0],
                negative_prompt_embeds =generation['embeddings'][1],
                pooled_prompt_embeds=generation['embeddings'][2],
                negative_pooled_prompt_embeds=generation['embeddings'][3],
                generator=generator,
                **self.gen_dict,
            ).images
            self.individual_totals.append(self.individual_start)

############## AUTODECODE
    def decode_latent(self, opt):
        self.vae_opt, self.vae_exp = opt
        self.autoencoder = self.vae_opt["vae"] #autoencoder wants full path and filename 
        if self.vae_exp.get("variant",0):
            var, dtype = self.float_converter(self.vae_exp["variant"])
            self.vae_exp["variant"] = var
            self.vae_exp.setdefault("torch_dtype", dtype)
        file_prefix = self.vae_opt["file_prefix"] + self.vae_opt["lora_class"] + self.vae_opt["algorithm"]
        self.tc(self.clock, f"decode configured for {os.path.basename(self.autoencoder)}...", self.bug_off)
        self.autoencoder = AutoencoderKL.from_single_file(self.autoencoder,**self.vae_exp).to(self.device)
        self.pipe.vae = self.autoencoder
        self.debugger(locals())
        if self.vae_opt.get("upcast_vae",0): self.pipe.upcast_vae()
        # if self.vae_opt.get("compile_vae",0): 
        #    self.pipe.vae.decode = self.compile_model(self.pipe.vae.decode, self.vae_opt["compile"])
        with torch.no_grad():
            counter = [s.endswith('png') for s in os.listdir(config.get_path("output"))].count(True) # get existing images
            self.tc(self.clock, f"decoding...")
            for i, generation in enumerate(self.queue, start=1):
                self.seed = generation['seed']
                self.tc(self.image_start, f"{i} of {len(self.queue)}", self.bug_off) 
                generation['latents'] = generation['latents'].to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)


                image = self.pipe.vae.decode(
                    generation['latents'] / self.pipe.vae.config.scaling_factor,
                    return_dict=False,
                )[0]

                image = self.pipe.image_processor.postprocess(image)[0] #, output_type='pil')[0]

                self.tc(self.clock, f"saving...")     
                counter += 1
                filename = f"{file_prefix}-{self.seed}-{counter}-batch-{i}.png"

                image.save(os.path.join(config.get_path("output"), filename)) # optimize=True,     

############## MEASUREMENT SUMMARY
    def metrics(self):

        if self.device == "cuda":
            max_memory = round(torch.cuda.max_memory_allocated(self.device) * 1e-9, 2)
            self.tc(self.clock, f"Max. memory used: {max_memory} GB", self.bug_off)
        
        if not self.bug_off:
            for i,num in enumerate(self.individual_totals):
                self.tc(self.individual_totals[i], f"image {i+1}")

############## DEBUG TOOLS
    def debugger(self, variables):
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
        "diffusers"
        "transformers"
        "torch"
        "vae"
        "AutoencoderKL"
        "generation"
        "tensor"
        ]
        for each in variables:
            if each not in var_list:
                print(f"{each} = {variables[each]}")
  
# create_index = IndexManager().write_index()       # (defaults to config/index.json)

#genesis node
optimize = NodeTuner()
optimize.determine_tuning("ponyFaetality_v11.safetensors")
opt_exp = optimize.opt_exp()
insta = T2IPipe()
insta.set_device()

#loader node transformer class
gen_exp = optimize.gen_exp()
insta.declare_encoders(gen_exp)

#prompt node
prompt = "A slice of a rich and delicious chocolate cake presented on a table in a palace reminiscent of Versailles owned by a vampire"
seed = int(soft_random())
insta.queue_manager(prompt,seed)

#enocde node
cond_exp = optimize.cond_exp()
insta.encode_prompt(cond_exp)

#cache ctrl node
insta.cache_jettison(encoder=True)

#t2i

pipe_exp = optimize.pipe_exp()
insta.construct_pipe(pipe_exp)
insta.diffuse_latent(gen_exp)

#cache ctrl node
insta.cache_jettison(lora=True)

#vae node
vae_exp = optimize.vae_exp()
insta.decode_latent(vae_exp)

#cache ctrl node
insta.cache_jettison(vae=True)
#metrics node
insta.metrics()


