"""
Credits:
Felixsans
"""
import gc
import os
import torch
import datetime
from time import perf_counter_ns
from collections import defaultdict

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
from diffusers import AutoencoderKL, AutoPipelineForText2Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import accelerate

from sdbx import logger
from sdbx.config import config
from sdbx.nodes.helpers import seed_planter

class T2IPipe:
    # __call__? NO __init__! ONLY __call__. https://huggingface.co/docs/diffusers/main/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForText2Image

    config_path = config.get_path("models.metadata")
    spec = config.get_default("spec","data")

############## STFU HUGGINGFACE
    def hf_log(self, on=False, fatal=False):
        if on is True:
            tf_log.enable_default_handler()
            df_log.enable_default_handler()
            tf_log.set_verbosity_warning()
            df_log.set_verbosity_warning()
        if fatal is True:
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
            if old_index is key:
                return val[0], val[1]

    def class_converter(self,class_name):
        class_name_chart = {
            "VAE": self.pipe.vae.decode,
            "TRA": self.pipe.transformer, 
            "DIF": self.pipe.unet
            }
        for key, val in class_name_chart.items():
            if class_name is key:
                return val
            
############## SCHEDULER
    def algorithm_converter(self, non_constant, exp):
        self.non_constant = non_constant
        self.algo_exp = exp
        self.schedule_chart = {
            "EulerDiscreteScheduler" : EulerDiscreteScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp),           
            "EulerAncestralDiscreteScheduler" : EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp),
            "FlowMatchEulerDiscreteScheduler" : FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp),     
            "EDMDPMSolverMultistepScheduler" : EDMDPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp), 
            "DPMSolverMultistepScheduler" : DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp),        
            "DDIMScheduler" : DDIMScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp),
            "LCMScheduler" : LCMScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp),
            "TCDScheduler" : TCDScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp),
            "AysSchedules": AysSchedules,
            "HeunDiscreteScheduler" : HeunDiscreteScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp),
            "UniPCMultistepScheduler" : UniPCMultistepScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp),
            "LMSDiscreteScheduler" : LMSDiscreteScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp),  
            "DEISMultistepScheduler" : DEISMultistepScheduler.from_config(self.pipe.scheduler.config,**self.algo_exp),
        }
        if self.non_constant in self.schedule_chart:
            self.pipe.scheduler = self.schedule_chart[self.non_constant]
            self.tc(self.clock)
            return self.pipe.scheduler
        else:
            try:
                raise ValueError(f"Scheduler '{self.non_constant}' not supported")
            except ValueError as error_log:
                logger.debug(f"Scheduler error {error_log}.", exc_info=True)

############## QUEUE
    def queue_manager(self, prompt, seed):
        self.queue = []    

        self.queue.extend([{
            "prompt": prompt,
            "seed": seed,
            }])
        return self.queue
    
############## ENCODERS
    def declare_encoders(self, model_symlinks, tokenizers, expressions):
        self.tformer_models = model_symlinks
        self.tokenizer_subfolders = tokenizers 
        self.encoder_expressions = expressions
        self.tokenizer = []
        self.text_encoder = []
        for i in self.tformer_models:
            if self.encoder_expressions[i].get("variant",0) != 0:
                var, dtype = self.float_converter(self.encoder_expressions[i]["variant"])
                self.encoder_expressions[i]["variant"] = var
                self.encoder_expressions[i].setdefault("torch_dtype", dtype)

            self.tokenizer[i] = CLIPTokenizer.from_pretrained(
                self.tformer_models[i],
                self.tokenizer_subfolder[i]
            )
            self.hf_log(fatal=True) #suppress layer skip messages
            self.text_encoder[i] = CLIPTextModel.from_pretrained(
                self.tformer_models[i],
                **self.encoder_expressions[i]
            ).to(self.device)
            self.hf_log(on=True) #return to normal

        return self.tokenizer, self.text_encoder

############## EMBEDDINGS
    def generate_embeddings(self, prompts, tokenizers, text_encoders, exp):
        self.emb_exp = exp
        embeddings_list = []
        #for prompt, tokenizer, text_encoder in zip(prompts, self.tokenizer.values(), self.text_encoder.values()):
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            cond_input = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            **self.emb_exp
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

############## ENCODE
    def encode_prompt(self, enc_exp):
        with torch.no_grad():
            for generation in self.queue:
                generation['embeddings'] = self.generate_embeddings(
                    [generation['prompt'], generation['prompt']],
                    self.tokenizer, self.text_encoder, self.enc_exp
                    )

 ############## VAE PT1
    def add_vae(self, model, vae, device):
        self.vae = vae
        if self.vae_exp.get("variant",0) != 0:
            var, dtype = self.float_converter(self.vae["variant"])
            self.vae["variant"] = var
            self.vae_exp.setdefault("torch_dtype", dtype)
        self.autoencoder = AutoencoderKL.from_single_file(model,**self.vae).to(device)
        return self.autoencoder
    
############## PIPE
    def construct_pipe(self, model, pipe_data, device):
        self.pipe_data = pipe_data
        if self.pipe_data.get("variant",0):
            var, dtype = self.float_converter(self.pipe_data["variant"])
            self.pipe_data["variant"] = var
            self.pipe_data.setdefault("torch_dtype", dtype)

        self.pipe = AutoPipelineForText2Image.from_pretrained(model, **self.pipe_data).to(device)
        if device is "mps":
            self.pipe.enable_attention_slicing()
        return self.pipe

############## LORA
    def add_lora(self, lora, weight_name, fuse, scale):
        self.pipe.load_lora_weights(lora, weight_name=weight_name)
        if fuse: 
            self.pipe.fuse_lora(scale) #add unet only possibility
        return self.pipe

############## INFERENCE
    def diffuse_latent(self, gen_data, scheduler, scheduler_data):
        self.pipe.scheduler = self.algorithm_converter(scheduler, scheduler_data )
        generator = torch.Generator(device=self.device)
        if self.spec.get("dynamo",False) is True:
            for i, generation in enumerate(self.queue, start=1):
                seed_planter(generation['seed'])
                generator.manual_seed(generation['seed'])
                
                generation['latents'] = self.pipe(
                    prompt_embeds=generation['embeddings'][0],
                    negative_prompt_embeds =generation['embeddings'][1],
                    pooled_prompt_embeds=generation['embeddings'][2],
                    negative_pooled_prompt_embeds=generation['embeddings'][3],
                    **self.gen_data
                )[0] # return individual for compiled
        else:
            for i, generation in enumerate(self.queue, start=1):
                seed_planter(generation['seed'])
                generator.manual_seed(generation['seed'])
                
                generation['latents'] = self.pipe(
                    prompt_embeds=generation['embeddings'][0],
                    negative_prompt_embeds =generation['embeddings'][1],
                    pooled_prompt_embeds=generation['embeddings'][2],
                    negative_pooled_prompt_embeds=generation['embeddings'][3],
                    **self.gen_data
                ).images #return entire batch at once

############## AUTODECODE
    def decode_latent(self, upcast):
        self.pipe.upcast_vae()
        with torch.no_grad():
            counter = [s.endswith('png') for s in os.listdir(config.get_path("output"))].count(True) # get existing images

            for i, generation in enumerate(self.queue, start=1):
                self.seed = generation['seed']
                generation['latents'] = generation['latents'].to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

                image = self.pipe.vae.decode(
                    generation['latents'] / self.pipe.vae.config.scaling_factor,
                    return_dict=False,
                )[0]

############## OUTPUT
    def save_image(self, queue, opt):
        self.vae_opt, self.vae_exp = opt
        for i, generation in enumerate(self.queue, start=1):
            file_prefix = f"{self.vae_opt['file_prefix']}-{self.vae_opt['lora_class']}-{self.vae_opt['algorithm']}"
            image = self.pipe.image_processor.postprocess(image)[0] #, output_type='pil')[0]

            counter += 1
            filename = f"{file_prefix}-{self.seed}-{counter}-batch-{i}.png"

            image.save(os.path.join(config.get_path("output"), filename)) # optimize=True,     
            self.tc(self.clock)


############## SET DEVICE
    def set_device(self, device=None):
        if device is None: 
            self.device = next(iter(self.spec.get("devices","cpu")),"cpu")
        else: 
            self.device = device
            if device is "cuda":
                torch.backends.cudnn.allow_tf32 = self.spec["torch.backends.cudnn.allow_tf32"]
                torch.backends.cuda.enable_flash_sdp(self.spec["flash_attention_2"])

############## MEMORY OFFLOADING
    def offload_to(self, offload_method):
        if not "cpu" in self.device:
            if offload_method is "sequential": self.pipe.enable_sequential_cpu_offload()
            elif offload_method is "cpu": self.pipe.enable_model_cpu_offload() 
        elif offload_method is "disk": accelerate.disk_offload()
        return self.pipe
    
############## CFG CUTOFF
    def _dynamic_guidance(self, pipe, step_index, timestep, callback_key):
        if step_index is int(pipe.num_timesteps * 0.5):
            callback_key['prompt_embeds'] = callback_key['prompt_embeds'].chunk(2)[-1]
            callback_key['add_text_embeds'] = callback_key['add_text_embeds'].chunk(2)[-1]
            callback_key['add_time_ids'] = callback_key['add_time_ids'].chunk(2)[-1]
            pipe._guidance_scale = 0.0
        return callback_key

############## COMPILE
    def compile_model(self, model, compile_data):
        self.pipe_element = self.class_converter(model)
        self.pipe_element = torch.compile(self.pipe_element, **compile_data) #compile needs to be put in another routine
        return self.pipe

############## CACHE MANAGEMENT
    def cache_jettison(self, encoder=False, lora=False, unet=False, vae=False):
        self.tc(self.clock)
        if encoder: del self.tokenizer, self.text_encoder,  self.tokenizer_2, self.text_encoder_2
        if lora: self.pipe.unload_lora_weights()
        if unet: del self.pipe.unet
        if vae: del self.pipe.vae
        gc.collect()
        if self.device is "cuda": torch.cuda.empty_cache()
        if self.device is "mps": torch.mps.empty_cache()
        if self.device is "xpu": torch.xpu.empty_cache()
                      
############## MEASUREMENT SUMMARY
    def metrics(self):
        if "cuda" in self.device:
            memory = round(torch.cuda.max_memory_allocated(self.device) * 1e-9, 2)
            logger.debug(f"Total mem use: {memory}.", exc_info=True)
            # self.tc(self.clock)
