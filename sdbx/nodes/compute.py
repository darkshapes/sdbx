"""
Credits:
Felixsans
"""

import gc
import os
from time import perf_counter_ns, perf_counter
from sdbx.nodes.tuner import NodeTuner
from diffusers import DiffusionPipeline, AutoPipelineForText2Image , AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, TCDScheduler, HeunDiscreteScheduler, UniPCMultistepScheduler, DEISMultistepScheduler
from diffusers.schedulers import AysSchedules, FlowMatchEulerDiscreteScheduler, EDMDPMSolverMultistepScheduler, DPMSolverMultistepScheduler, LCMScheduler, LMSDiscreteScheduler
from sdbx.config import config
from sdbx.nodes.helpers import seed_planter, soft_random
import torch
from accelerate import Accelerator
import peft
import platform
import datetime
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

class T2IPipe:
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
            
    def algorithm_converter(self, non_constant):
        schedule_chart =  [
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
        ]
        for each in schedule_chart:
            if non_constant == each:
                return each
    
    def tc(self, clock, string): return print(f"[ {str(datetime.timedelta(milliseconds=(((perf_counter_ns()-clock)*1e-6))))[:-4]} ] {string}")

    def set_device(self, device=None):
        if device==None:
            if torch.cuda.is_available(): self.device = "cuda" # https://pytorch.org/docs/stable/torch_cuda_memory.html
            else: self.device = "mps" if (torch.backends.mps.is_available() & torch.backends.mps.is_built()) else "cpu"# https://pytorch.org/docs/master/notes/mps.html
        else: self.device = device      

    def queue_manager(self, prompt, seed):
        self.clock = perf_counter_ns() 
        self.tc(self.clock, "determining device type...")
        self.queue = []    

        self.queue.extend([{
            "prompt": prompt,
            "seed": seed,
            }])
        
    def declare_encoders(self, transformer="stabilityai/stable-diffusion-xl-base-1.0"):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            transformer,
            subfolder='tokenizer',
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            transformer,
            subfolder='text_encoder',
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant='fp16',
        ).to(self.device)

        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            transformer,
            subfolder='tokenizer_2',
        )

        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            transformer,
            subfolder='text_encoder_2',
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant='fp16',
        ).to(self.device)

    def generate_embeddings(self, prompts, tokenizers, text_encoders):
        self.tc(self.clock, f"encoding prompt with device: {self.device}...")

        embeddings_list = []
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            cond_input = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
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

    def encode_prompt(self):
        with torch.no_grad():
            for generation in self.queue:
                generation['embeddings'] = self.generate_embeddings(
                    [generation['prompt'], generation['prompt']],
                    [self.tokenizer, self.tokenizer_2],
                    [self.text_encoder, self.text_encoder_2],
        )

    def cache_jettison(self, encoder=False, lora=False, unet=False, vae=False):
        self.tc(self.clock, f"empty cache...")
        if encoder: del self.tokenizer, self.text_encoder, self.tokenizer_2, self.text_encoder_2
        if lora: self.pipe.unload_lora_weights()
        if unet: del self.pipe.unet
        if vae: del self.pipe.vae
        gc.collect()
        if self.device == "cuda": torch.cuda.empty_cache()
        if self.device == "mps": torch.mps.empty_cache()
 
    def construct_pipe(self, exp, model="stabilityai/stable-diffusion-xl-base-1.0"):
        self.tc(self.clock, f"precision set for: {exp["variant"]}, using {exp["torch_dtype"]}")
        var, dtype = self.float_converter(exp["variant"])
        exp["variant"] = var
        exp.setdefault("torch_dtype", dtype)
        self.tc(self.clock, f"load model {model}...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(model, **exp).to(self.device)

    def add_lora(self, lora="pcm_sdxl_normalcfg_16step_converted_fp16.safetensors", path="models.lora"):
        lora_path = config.get_path(path)
        self.tc(self.clock, f"set scheduler, lora = {os.path.join(lora_path, lora)}")  # lora2
        lora = lora #lora needs to be explicitly declared
        self.pipe.load_lora_weights(self,lora_path, weight_name=lora)

   
    def build_scheduler(self):
        #self.schedule =  EulerDiscreteScheduler
        self.scheduler_args = {  
            }
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, **self.scheduler_args)

    def offload(self, sequential=False, cpu=False, disk=False):
        self.tc(self.clock, f"set offload as {cpu|disk} and sequential as {sequential} for {self.device} device") 
        if self.add_loradevice=="cuda":
            if sequential: self.pipe.enable_sequential_cpu_offload()
            if cpu: self.pipe.enable_model_cpu_offload() 
            if disk: self.pipe.enable_disk_offload() 

    #self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

    def add_dynamic_cfg(self):
            self.tc(self.clock, "set dynamic cfg")
            self.gen_args.setdefault("callback_on_step_end",self._dynamic_guidance)
            self.gen_args.setdefault("callback_on_step_end_tensor_inputs",['prompt_embeds', 'add_text_embeds','add_time_ids'])

    def _dynamic_guidance(self, pipe, step_index, timestep, callback_key):
        if step_index == int(pipe.num_timesteps * 0.5):
            callback_key['prompt_embeds'] = callback_key['prompt_embeds'].chunk(2)[-1]
            callback_key['add_text_embeds'] = callback_key['add_text_embeds'].chunk(2)[-1]
            callback_key['add_time_ids'] = callback_key['add_time_ids'].chunk(2)[-1]
            pipe._guidance_scale = 0.0
        return callback_key
    
    def diffuse_latent(self, dynamic_cfg=False):
        self.gen_args = { 
            "num_inference_steps": "16",
            "guidance_scale": "5"
        }
        if dynamic_cfg: self.add_dynamic_cfg()

        self.tc(self.clock, f"set generator") 
        generator = torch.Generator(device=self.device)

        self.tc(self.clock, f"begin queue loop...")
        for i, generation in enumerate(self.queue, start=1):

            self.tc(self.clock, f"planting seed {generation['seed']}...")
            seed_planter(generation['seed'])
            generator.manual_seed(generation['seed'])

            self.tc(self.clock, f"inference device: {self.device}....")
            self.image_start = perf_counter()
            generation['latents'] = self.pipe(
                prompt_embeds=generation['embeddings'][0],
                negative_prompt_embeds =generation['embeddings'][1],
                pooled_prompt_embeds=generation['embeddings'][2],
                negative_pooled_prompt_embeds=generation['embeddings'][3],
                generator=generator,
                output_type='latent',
                **self.gen_args,
            ).images 

    def decode_latent(self, autoencoder="flatpiecexlVAE_baseonA1579.safetensors", prefix="Shadowbox-", compress_level=4, vae_tile=False, vae_slice=False):
        vae_path = config.get_path("models.image")
        autoencoder = os.path.join(vae_path,autoencoder) #autoencoder wants full path and filename

        vae_args = {}
        vae_args["torch_dtype"]= torch.float16
        vae_args["cache_dir"]="vae_"
        if vae_tile: vae_args["enable_tiling"]= True
        else: vae_args["disable_tiling"]=True
        if vae_slice:vae_args["enable_slicing"]= True
        else: vae_args["disable_slicing"]=True

        self.tc(self.clock, f"decoding using {autoencoder}...")
        vae = AutoencoderKL.from_single_file(autoencoder,**vae_args).to(self.device)
        self.pipe.vae=vae
        self.pipe.upcast_vae()

        with torch.no_grad():
            counter = [s.endswith('png') for s in os.listdir(config.get_path("output"))].count(True) # get existing images
            for i, generation in enumerate(self.queue, start=1):
                generation['total_time'] = perf_counter() - self.image_start
                self.tc(self.image_start,"generation complete")
                generation['latents'] = generation['latents'].to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

                image = self.pipe.vae.decode(
                    generation['latents'] / self.pipe.vae.config.scaling_factor,
                    return_dict=False,
                )[0]

                image = self.pipe.image_processor.postprocess(image, output_type='pil')[0]

                self.tc(self.clock, f"saving")     
                counter += 1
                filename = f"{prefix}-{counter}-batch-{i}.png"

                image.save(os.path.join(config.get_path("output"), filename)) # optimize=True,

    def metrics(self):
        images_totals = ', '.join(map(lambda generation: str(round(generation['total_time'], 1)), self.queue))
        print('Image time:', images_totals, 'seconds')

        images_average = round(sum(generation['total_time'] for generation in self.queue) / len(self.queue), 1)
        print('Average image time:', images_average, 'seconds')


        if self.device == "cuda":
            max_memory = round(torch.cuda.max_memory_allocated(device='cuda') * 1e-9, 2)
            print('Max. memory used:', max_memory, 'GB')
        
        self.tc(self.clock, f" <-time total...")     


optimize = NodeTuner()
optimize.determine_tuning("ponyFaetality_v11.safetensors")
insta = T2IPipe()
prompt = "A slice of a rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles"
seed = int(soft_random())
insta.queue_manager(prompt,seed)
insta.set_device()
insta.declare_encoders()
insta.encode_prompt()
insta.cache_jettison(encoder=1)
opt = optimize._pipe_exp()
print(opt)
insta.construct_pipe(opt)
insta.build_scheduler()
insta.offload()
insta.add_lora()
insta.diffuse_latent()
insta.decode_latent()
insta.metrics()


