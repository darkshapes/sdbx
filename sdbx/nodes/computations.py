import gc
import os
import platform
import datetime
import time
from time import perf_counter
from typing import Tuple, List

from sdbx.config import config
from sdbx.nodes.helpers import seed_planter, soft_random
from diffusers import AutoPipelineForText2Image, AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers.schedulers import AysSchedules
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import torch
from llama_cpp import Llama


# AUTOCONFIG OPTIONS : INFERENCE
# [universal] lower vram use (and speed on pascal apparently!!)
sequential_offload = True
precision = '16'  # [universal], less memory for trivial quality decrease
# [universal] half cfg @ 50-75%. sdxl architecture only. playground, vega, ssd1b, pony. bad for pcm
dynamic_guidance = True
# [compatibility for alignyoursteps to match model type
model_ays = "StableDiffusionXLTimesteps"
# [compatibility] only for sdxl
pcm_default = "pcm_sdxl_normalcfg_8step_converted_fp16.safetensors"
pcm_default_dl = "Kijai/converted_pcm_loras_fp16/tree/main/sdxl/" + pcm_default
cpu_offload = False  # [compatibility] lower vram use by pushing to cpu
# [compatibility] certain types of models need this, it influences determinism as well
bf16 = False
timestep_spacing = "trailing"  # [compatibility] DDIM, PCM "trailing"
clip_sample = False  # [compatibility] PCM False
set_alpha_to_one = False,  # [compatibility]PCM False
rescale_betas_zero_snr = True  # [compatibility] DDIM True
disk_offload = False  # [compatibility] last resort, but things work
compile_unet = False #[performance] compile the model for speed, slows first gen only, doesnt work on my end

# AUTOCONFIG OPTIONS  : VAE
# pipe.upcast_vae()
vae_tile = True  # [compatibility] tile vae input to lower memory
vae_slice = False  # [compatibility] serialize vae to lower memory
# [compatibility] this should be detected by model type
vae_default = "madebyollin/sdxl-vae-fp16-fix.safetensors"
vae_config_file = "ssdxlvae.json"  # [compatibility] this too
vae_path = os.listdir(config.get_path("models.vae"))

# SYS IMPORT
device = ""
compile_unet = ""
queue = ""
clear_cache = ""
linux = ""

# Utility function to print the time
def tc() -> None:
    print(str(datetime.timedelta(seconds=time.process_time())), end="")

# Device and memory setup
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"

def clear_memory_cache(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


class Inference:
    def __init__(self):
        self.imports()
        self.device = get_device()
        self.pipe = None
        self.generator = torch.Generator(device=self.device)
        if queue not in globals(): queue = {}

    def push_prompt(self, prompt, seed):
        prompt = prompt
        seed = seed
        return queue.extend([{
            "prompt": prompt,
            "seed": seed,
        }])

    def encode_prompt(self, prompts, tokenizers, text_encoders):
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

    def  load_token_encoder(self,**token_encoder):
        tokenizer = []
        text_encoder = []
        for t in token_encoder:
            #determine model class here
            tokenizer[t] = token_encoder[t]
            text_encoder[t] = token_encoder[t]
            token_args = {
                'subfolder':'tokenizer',
            }
            encoder_args = {
                'subfolder':'text_encoder',
                'torch_dtype':torch.float16,
                'variant':'fp16',           
            }
            tokenizer[t] = CLIPTokenizer.from_pretrained(
                token_encoder[t],
                **token_args,
            )
            text_encoder[t] = CLIPTextModel.from_pretrained(
                token_encoder[t],
                **encoder_args,
            ).to(self.device)

        return tokenizer, text_encoder
    
    def start_encoding(self, queue, tokenizer, token_encoder):
        with torch.no_grad():
            for generation in queue:
                generation['embeddings'] = self.encode_prompt(
                    [generation['prompt'], generation['prompt']],
                    [*tokenizer],
                    [*token_encoder]
            )
        return queue

    def load_pipeline(self, model, precision=16, bfloat=False):
        #determine model class here
        self.pipe_args = {
            "tokenizer":None,
            "text_encoder":None,
            "tokenizer_2":None,
            "text_encoder_2":None,
        }
        if precision=='16':
            self.pipe_args["variant"]="fp16"
            self.pipe_args["torch_dtype"]=torch.bfloat16 if bf16 else torch.float16
        print(f"{tc()} precision set for: {precision}, bfloat: {bf16}, using {self.pipe_args["torch_dtype"]}") #debug

        self.pipe = AutoPipelineForText2Image.from_pretrained(model, **self.pipe_args).to(self.device)
        if compile_unet: self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        return self.pipe

    def run_inference(self, 
        queue, inference_steps=8, guidance_scale=5.0, dynamic_guidance=False,
        scheduler = EulerAncestralDiscreteScheduler, lora="pcm_sdxl_normalcfg_8step_converted_fp16.safetensors"):
        lora_path = config.get_path("models.loras")

        print(f"{tc()} set scheduler, lora = {os.path.join(lora_path, lora)}")  # lora2
        scheduler_args = {
        }

        if scheduler=="DDIMScheduler":
            scheduler_args[ "timestep_spacing"]=timestep_spacing
            scheduler_args["rescale_betas_zero_snr"]=rescale_betas_zero_snr #[compatibility] DDIM, v-pred?
            if lora:
                scheduler_args["clip_sample"]=clip_sample #[compatibility] PCM
                scheduler_args["set_alpha_to_one"]=set_alpha_to_one, #[compatibility]PCM
        # if scheduler=="DPMMultiStepScheduler":
            # scheduler_args["algorithm_type"]=

        self.pipe.load_lora_weights(lora_path, weight_name=lora)
        # if lora2: pipe.load_lora_weights(lora2, weight_name=weight_name)
        # load lora into u-net only : pipeline.unet.load_attn_procs("jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors")

        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config, **scheduler_args)
        # if text_inversion: pipe.load_textual_inversion(text_inversion)

        print(f"{tc()} set offload {cpu_offload} and sequential as {sequential_offload} for {device} device") #debug 
        if device=="cuda":
            if sequential_offload: self.pipe.enable_sequential_cpu_offload()
            if cpu_offload: self.pipe.enable_model_cpu_offload() 

        gen_args = {}
        if dynamic_guidance:
            print(f"{tc()} set dynamic cfg") #debug
            def dynamic_cfg(pipe, step_index, timestep, callback_key):
                if step_index == int(pipe.num_timesteps * 0.5):
                    callback_key['prompt_embeds'] = callback_key['prompt_embeds'].chunk(2)[-1]
                    callback_key['add_text_embeds'] = callback_key['add_text_embeds'].chunk(2)[-1]
                    callback_key['add_time_ids'] = callback_key['add_time_ids'].chunk(2)[-1]
                    pipe._guidance_scale = 0.0
                return callback_key
            
            gen_args["callback_on_step_end"]=dynamic_cfg
            gen_args["callback_on_step_end_tensor_inputs"]=['prompt_embeds', 'add_text_embeds','add_time_ids']

        if scheduler == "AysSchedules":
            timesteps = AysSchedules[model_ays] # should be autodetected
            gen_args["timesteps"]=timesteps

        print(f"{tc()} set generator") #debug
        generator = torch.Generator(device=device)

        for i, generation in enumerate(queue, start=1):
            self.image_start = perf_counter()                        #start the metric stopwatch
            print(f"{tc()} planting seed {generation['seed']}...") #debug
            seed_planter(generation['seed'])
            generator.manual_seed(generation['seed'])
            print(f"{tc()} inference device: {device}....") #debug

            generation['latents'] = self.pipe(
                prompt_embeds=generation['embeddings'][0],
                negative_prompt_embeds =generation['embeddings'][1],
                pooled_prompt_embeds=generation['embeddings'][2],
                negative_pooled_prompt_embeds=generation['embeddings'][3],
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type='latent',
                **gen_args,
            ).images 

        self.pipe.unload_lora_weights()
        return queue

    def cleanup(self):
        clear_memory_cache(self.device)

    def load_vae(self,file):
        vae = AutoencoderKL.from_single_file(file, torch_dtype=torch.float16, cache_dir="vae_").to("cuda")
        #vae = FromOriginalModelMixin.from_single_file(autoencoder, config=vae_config).to(device)
        self.pipe.vae=vae
        self.pipe.upcast_vae()

    def autodecode(self, vae, queue, file_prefix: str = "Shadowbox-"):
        print(f"{tc()} decoding using {autoencoder}...") #debug
        vae_find = "flat"
        file_prefix = "Shadowbox-"
        compress_level = 4 # optional png compression

        ### AUTOCONFIG OPTIONS  : VAE
        # pipe.upcast_vae()
        vae_tile = True #[compatibility] tile vae input to lower memory
        vae_slice = False #[compatibility] serialize vae to lower memory
        vae_default = "madebyollin/sdxl-vae-fp16-fix.safetensors" #[compatibility] this should be detected by model type
        vae_config_file ="ssdxlvae.json" #[compatibility] this too

        ### VAE SYSTEM
        vae_path = config.get_path("models.vae")
        autoencoder = os.path.join(vae_path,next(vae, vae_default)) #autoencoder wants full path and filename

        vae_config_path = os.path.join(config.get_path("models"),"metadata")
        symlnk = os.path.join(vae_config_path,"config.json") #autoencoder also wants specific filenames
        if os.path.isfile(symlnk): os.remove(symlnk)
        os.symlink(os.path.join(vae_config_path,vae_config_file),symlnk) #note: no 'i' in 'symlnk'

        with torch.no_grad():
            counter = [s.endswith('png') for s in os.listdir(config.get_path("output"))].count(True) # get existing images
            for i, generation in enumerate(queue, start=1):
                generation['total_time'] = perf_counter() - self.image_start
                generation['latents'] = generation['latents'].to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

                image = self.pipe.vae.decode(
                    generation['latents'] / self.pipe.vae.config.scaling_factor,
                    return_dict=False,
                )[0]

                image = self.pipe.image_processor.postprocess(image, output_type='pil')[0]

                print(f"{tc()} saving") #debug     
                counter += 1
                filename = f"{file_prefix}-{counter}-batch-{i}.png"
                

                image.save(os.path.join(config.get_path("output"), filename)) # optimize=True,
                metrics(generation, queue)
                return image, queue

        def metrics(generation, queue):# Metrics
            images_totals = ', '.join(map(lambda generation: str(round(generation['total_time'], 1)), queue))
            print('Image time:', images_totals, 'seconds')

            images_average = round(sum(generation['total_time'] for generation in queue) / len(queue), 1)
            print('Average image time:', images_average, 'seconds')

            if device == "cuda":
                if linux: torch.cuda.memory._dump_snapshot("mem")
                else: 
                    max_memory = round(torch.cuda.max_memory_allocated(device='cuda') / 1000000000, 2)
                    print('Max. memory used:', max_memory, 'GB')

def gguf_load(checkpoint, threads=8, max_context=8192, verbose=True):
    #determine model class here
    # print(f"loading:GGUF{os.path.join(config.get_path('models.llms'), checkpoint)}")
    return Llama(
        model_path=os.path.join(config.get_path("models.llms"), checkpoint),
        seed=soft_random(), #if one_time_seed == False else hard_random(),
        #n_gpu_layers=gpu_layers if cpu_only == False else 0,
        n_threads=threads,
        n_ctx=max_context,
        #n_batch=batch,
        #flash_attn=flash_attention,
        verbose=verbose,
    )

def llm_prompt(system_prompt, user_prompt, streaming=True):
    print("Encoding Prompt")
    return Llama.create_chat_completion(
        messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
        stream=streaming,
        #repeat_penalty=repeat_penalty,
        #temperature=temperature,
        #top_k=top_k,
        #top_p=top_p,
        #max_tokens=max_tokens,
    )
