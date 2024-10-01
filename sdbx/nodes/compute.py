"""
Credits:
Felixsans
"""

import gc
import os
from time import perf_counter
from sdbx.nodes.tuner import NodeTuner
from diffusers import DiffusionPipeline, AutoPipelineForText2Image , AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, TCDScheduler, HeunDiscreteScheduler, UniPCMultistepScheduler, DEISMultistepScheduler
from diffusers.schedulers import AysSchedules, FlowMatchEulerDiscreteScheduler, EDMDPMSolverMultistepScheduler, DPMSolverMultistepScheduler, LCMScheduler, LMSDiscreteScheduler
from sdbx.config import config
from sdbx.nodes.helpers import seed_planter, soft_random
import torch
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
import peft

# local_files_only
# force_download
# export HF_HUB_OFFLINE=True
# export DISABLE_TELEMETRY=YES linux macos
# set DISABLE_TELEMETRY=YES win

class Inference:
    #def __init__(self):
    config_path = config.get_path("models.metadata")
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
    queue = []
    
    def float_converter(self, old_index):
        self.old_index = old_index
        for key, val in self.float_chart.items():
            if self.old_index == key:
                self.new_index = val
                return self.new_index
            
    def algorithm_converter(self, non_constant):
        self.non_constant = non_constant
        for each in self.schedule_chart:
            if self.non_constant == each:
                self.non_constant = each
                return self.non_constant
    
    def symlinker(self, true_file, link_file, vae=False):
            self.link_file = link_file
            self.sym_lnk_split = os.path.split(self.link_file) # class_name split from path
            self.sym_lnk_file = os.path.join(self.config_path, self.link_file) #full path to file
            self.sym_lnk_folder = os.path.join(self.config_path,self.sym_lnk_split[0]) # path to file only
            if os.path.isfile(self.sym_lnk_file): os.remove(self.sym_lnk_file) # if already made, delete
            os.symlink(true_file ,self.sym_lnk_file) # create symlink note: no 'i' in 'symlnk'
            if vae: 
                return self.sym_lnk_file #return full path and file if vae
            else: 
                return self.sym_lnk_folder #return path only to others


    def declare_encoders(self, device, models):
        self.device = device
        self.tokenizer = {}
        self.text_encoder = {}
        self.models = models

        for each in self.models["file"].keys():
            self.class_name = each
            self.weights = self.models["file"][self.class_name]
            self.variant, self.tensor_dtype = self.float_converter(self.models["dtype"][each])
            self.symlnk_path = self.symlinker(self.weights,self.models["config"][each])
            self.tokenizer[self.class_name] = AutoTokenizer.from_pretrained(
                self.symlnk_path,
            )
            self.text_encoder[self.class_name] = AutoModel.from_pretrained(
                self.symlnk_path,
                torch_dtype=self.tensor_dtype,
                variant=self.variant,
            ).to(self.device)
        
        return self.tokenizer, self.text_encoder

    def _generate_embeddings(self, prompt, tokenizers, text_encoders):
        embeddings_list = []
        self.prompts = prompt
        self.tokenizers = tokenizers
        self.text_encoders = text_encoders
        #create instances of the models
        prompt_embeds = []
        i = 0
        for prompt, tokenizer, text_encoder in zip(self.prompts, self.tokenizers.values(), self.text_encoders.values()):
            # assuming tokeniZERS was a dict, but tokeniZER is now a list
            cond_input = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            prompt_embeds = text_encoder(cond_input.input_ids.to(self.device), output_hidden_states=True)

            pooled_prompt_embeds = prompt_embeds[i]
            embeddings_list.append(prompt_embeds.hidden_states[-2])

            prompt_embeds = torch.concat(embeddings_list, dim=-1)
            i += 1

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
            
    def encode_prompt(self, queue, tokenizer, text_encoder):
        self.queue=queue
        self.tokenizer=tokenizer
        self.text_encoder=text_encoder
        with torch.no_grad():
            for self.generation in self.queue:
                self.generation['embeddings'] = self._generate_embeddings(
                    [self.generation['prompt'], self.generation['prompt']],
                    self.tokenizer,
                    self.text_encoder
                )
            return self.generation

    def cache_jettison(self, device, encoder=False, lora=False, discard=True):
        self.device = device
        if encoder: del self.tokenizer, self.text_encoder
        if lora: 
            self.pipe.unload_lora_weights()
            del self.pipe.unet
        if discard:
            gc.collect()
            if self.device == "cuda": torch.cuda.empty_cache()
            if self.device == "mps": torch.mps.empty_cache()

    #not sure if this will work here yet
    # def dynamic_cfg(self, pipe, step_index, timestep, callback_val):
    #     self.pipe = pipe
    #     if step_index == int(self.pipe.num_timesteps * 0.5):
    #         self.callback_val['prompt_embeds'] = self.callback_val['prompt_embeds'].chunk(2)[-1]
    #         self.callback_val['add_text_embeds'] = self.callback_val['add_text_embeds'].chunk(2)[-1]
    #         self.callback_val['add_time_ids'] = self.callback_val['add_time_ids'].chunk(2)[-1]
    #         self.pipe._guidance_scale = 0.0
    #     return self.callback_val
    
    def debug_printer(self, locals):
        self.var_list = [
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
            ]
        self.variables = dict(locals)
        for each in self.variables:
            if each not in self.var_list:
                print(f"{each} = {self.variables[each]}")


    def latent_diffusion(self, device, embeddings, parameters):
        self.embeddings = embeddings
        self.device = device
        self.parameters=parameters

        self.model = self.parameters["model"]["file"]
        self.lora_path = config.get_path("models.lora")
        self.lora_file = os.path.basename(self.parameters["lora"]["file"])
        self.variant, self.tensor_dtype = self.float_converter(self.parameters["model"]["dtype"])
        self.use_low_cpu = self.parameters["pipe"]["low_cpu_mem_usage"]
        self.scheduler = self.algorithm_converter(self.parameters["pipe"]["algorithm"])
        self.model_config = self.parameters["pipe"]["config_path"]
        #self.model_config = self.parameters["model"]["yaml"]
        self.class_name = self.parameters["model"]["class"]
        
        self.keys, self.values = zip(*self.parameters["scheduler"].items())
        self.scheduler_args = {key: value for key, value in self.parameters["scheduler"].items()}     
        self.pipe_args = {
            "torch_dtype": self.tensor_dtype, #self.tensor_dtype,
            "variant": self.variant, #self.variant
            "tokenizer":None,
            "text_encoder":None,
            "tokenizer_2":None,
            "text_encoder_2":None,
            "config_name": self.model_config,
            "low_cpu_mem_usage":self.use_low_cpu,
            "device_map": None,
            #"vae": self.parameters["pipe"]["config_path"],
            #"unet":None,
            #"feature_extractor":None,
            #"image_encoder":None,
            #"scheduler": self.scheduler,
            #"local_files_only": True,
            #"local_files_only":True,
        }

        #symlink model
        self.symlnk_path = self.symlinker(self.model,self.parameters["model"]["config"])
        self.pipe = AutoPipelineForText2Image.from_pretrained(self.symlnk_path, **self.pipe_args).to(device)
        # , **self.pipe_args).to(device)
        #self.debug_printer(locals())
        print(self.lora_file, self.lora_path,self.model)
        #if self.parameters["lora"]["unet_only"]: self.pipe.unet.load_attn_procs(self.lora_path, weight_name=self.lora_file)
        #else: self.pipe.load_lora_weights(self.lora_path, weight_name=self.lora_file)
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
            #**self.scheduler_args)
        #if self.parameters["lora"]["fuse"]: self.pipe.fuse_lora(lora_scale=self.parameters["lora"]["scale"]) 
        # if lora 2: self.pipe.load_lora_weights(self.lora_path, weight_name=self.lora_name)
        # fuse lora 2
        # if text_inversion: pipe.load_textual_inversion(text_inversion)      

        if self.parameters["pipe"]["sequential_offload"]: self.pipe.enable_sequential_cpu_offload()
        if self.parameters["pipe"]["cpu_offload"]: self.pipe.enable_model_cpu_offload() 

        # if self.parameters["pipe"]["dynamic_cfg"]:
        #     def dynamic_cfg(pipe, step_index, timestep, callback_val):
        #         self.pipe = pipe
        #         if step_index == int(pipe.num_timesteps * 0.5):
        #             callback_val['prompt_embeds'] = callback_val['prompt_embeds'].chunk(2)[-1]
        #             callback_val['add_text_embeds'] = callback_val['add_text_embeds'].chunk(2)[-1]
        #             callback_val['add_time_ids'] = callback_val['add_time_ids'].chunk(2)[-1]
        #             self.pipe._guidance_scale = 0.0
        #         return callback_val
            
        #     self.gen_args["callback_on_step_end"]=self.dynamic_cfg
        #     self.gen_args["callback_on_step_end_tensor_inputs"]=['prompt_embeds', 'add_text_embeds','add_time_ids']

        self.generator = torch.Generator(device=self.device)
        self.gen_args = {}        
        self.gen_args["output_type"] = 'latent',

        if self.parameters["compile"]["unet"]: self.pipe.unet = torch.compile(
               self.pipe.unet, 
               mode=self.parameters["mode"], 
               fullgraph=self.parameters["fullgraph"]
           )    

        for i, generation in enumerate(self.embeddings):#, start=1):
            print(self.embeddings['embeddings'][0])
            print(self.embeddings['embeddings'][1])
            print(self.embeddings['embeddings'][2])
            print(self.embeddings['embeddings'][3])
            print(i)
            seed_planter(self.parameters["pipe"]["seed"])
            self.generator.manual_seed(self.parameters["pipe"]["seed"])

            generation['latents'] = self.pipe(
                prompt_embeds=self.embeddings['embeddings'][0],
                negative_prompt_embeds = self.embeddings['embeddings'][1],
                pooled_prompt_embeds=self.embeddings['embeddings'][2],
                negative_pooled_prompt_embeds=self.embeddings['embeddings'][3],
                num_inference_steps=self.parameters["pipe"]["num_inference_steps"],
                generator=self.generator,
                **self.gen_args,
            ).images 
        
        return generation


    def decode_latent(self, parameters):
        self.parameters = parameters
        if self.parameters["vae"]["upcast"]:self.pipe.upcast_vae()
        if self.parameters["vae"]["slice"]:self.pipe.upcast_vae() 
        self.autoencoder = self.parameters["vae"]["file"]
        self.dtype = self.float_converter(self.parameters["vae"]["dtype"])
        #self.symlnk_path = self.symlinker(self.class_name,self.mdel_config,self.config_path)

        #self.vae_config_path = self.parameters["vae"]["config"]
        self.output_path = config.get_path("output")


        self.vae = AutoencoderKL.from_single_file(self.autoencoder, torch_dtype=self.dtype[0], cache_dir="vae_").to(self.device)
        #vae = FromOriginalModelMixin.from_single_file(autoencoder, config=vae_config).to(device)
        self.pipe.vae=self.vae
        self.pipe.upcast_vae()

        with torch.no_grad():
            counter = [s.endswith('png') for s in os.listdir(config.get_path("output"))].count(True) # get existing images
            for i, generation in enumerate(self.queue, start=1):
                generation['latents'] = generation['latents'].to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

                self.image = self.pipe.vae.decode(
                    generation['latents'] / self.pipe.vae.config.scaling_factor,
                    return_dict=False,
                )[0]

                self.image = self.pipe.image_processor.postprocess(self.image, output_type='pil')[0]
                counter += 1
                self.filename = f"{self.parameters["file_prefix"]}-{counter}-batch-{i}.png"

                self.image.save(os.path.join(self.output_path, self.filename)) # optimize=True,


class QueueManager(Inference):
    def prompt(self, prompt, seed):
        self.queue= []
        self.queue.extend([{
            'prompt': prompt,
            'seed': int(seed),            
        }])
        for i, self.generation in enumerate(self.queue, start=1):
            self.embed_start = perf_counter()  # stopwatch begin
        return self.queue

    def metrics(self, queue):
            self.generation['time'] = perf_counter() - self.embed_start #stopwatch end
            self.embed_total = ', '.join(map(lambda generation: str(round(generation['time'], 1)), self.queue))
            print('time:', self.embed_total)

            embed_avg = round(sum(self.generation['time'] for self.generation in self.queue) / len(self.queue), 1)
            print('Average time:', embed_avg)
            return self.generation


def test_process():
    filename= "ponyFaetality_v11.safetensors"
    path = config.get_path("models.image")
    device = "cuda"
    print("beginning")
    print(os.path.join(path,filename))
    default = NodeTuner().determine_tuning(filename)
    #model_class = default["model"]["class"]
    if default != None:
        prompt=default["pipe"]["prompt"]
        seed=default["pipe"]["seed"]
        #neg=list(default["pipe"]["negative"])
        tokenizer = {}
        text_encoder = {}
        inference_instance = []
        inference_instance = Inference()
        queue = QueueManager()
        active = queue.prompt(prompt,seed)
        tokenizer,  text_encoder = inference_instance.declare_encoders(device, default["transformer"])
        embeddings = inference_instance.encode_prompt(active, tokenizer, text_encoder)
        null = inference_instance.cache_jettison(device, encoder=True, discard=default["pipe"]["cache_jettison"])
        gen = inference_instance.latent_diffusion(device, embeddings, default)
        null = inference_instance.cache_jettison(device, lora=True, discard=default["pipe"]["cache_jettison"])
        image = inference_instance.decode_latent(device, default)
        timer = queue.metrics(prompt)
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
            ]
        variables = dict(locals())
        for each in variables:
            if each not in var_list:
                print(f"{each} = {variables[each]}")

go = test_process()

    #transfer this block to nodes ----------->
    # <------------ end of block for nodes

    #transfer this block to system config .json ----------->
    # if torch.cuda.is_available(): 
    #     device = "cuda" # https://pytorch.org/docs/stable/torch_cuda_memory.html
    # else:  # https://pytorch.org/docs/master/notes/mps.html
    #    device = "mps" if (torch.backends.mps.is_available() & torch.backends.mps.is_built()) else "cpu"
    # <------------ end of block for system config .json
    # filename = 
    #     if not os.path.isdir(each):
    #         #full_path=os.path.join(path,each)
