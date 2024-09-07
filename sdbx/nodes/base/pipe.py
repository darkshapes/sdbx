from sdbx.nodes.types import *
from sdbx.nodes.helpers import seed_planter, get_gpus, cache_bin, get_schedulers, get_solvers, soft_random, hard_random, get_gpus, get_dir_files
from time import perf_counter
import torch
from sdbx.config import config
from sdbx.nodes.types import *
from diffusers import AutoPipelineForText2Image, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
import os
from llama_cpp import Llama


@node(name="GGUF Loader")
def gguf_loader(
    checkpoint: Literal[*get_dir_files("models.llms", ".gguf")] = next(iter(get_dir_files("models.llms", ".gguf")), None),
    cpu_only: bool = True,
        gpu_layers: A[int, Dependent(on="cpu_only", when=False), Slider(min=-1, max=35, step=1)] = -1,
    advanced_options: bool = False,
        threads: A[int, Dependent(on="advanced_options", when=True), Slider(min=0, max=64, step=1)] = 8,
        max_context: A[int,Dependent(on="advanced_options", when=True), Slider(min=0, max=32767, step=64)] = 0,
        one_time_seed: A[bool, Dependent(on="advanced_options", when=True)] = False,
        flash_attention: A[bool, Dependent(on="advanced_options", when=True)] = False,
        verbose: A[bool, Dependent(on="advanced_options", when=True)] = False,
        batch: A[int, Dependent(on="advanced_options", when=True), Numerical(min=0, max=512, step=1), ] = 1,  
) -> I[Any]:
    # print(f"loading:GGUF{os.path.join(config.get_path('models.llms'), checkpoint)}")
    generate = Llama(
        model_path=os.path.join(config.get_path("models.llms"), "codeninja-1.0-openchat-7b.Q5_K_M.gguf"),
        seed=soft_random(), #if one_time_seed == False else hard_random(),
        #n_gpu_layers=gpu_layers if cpu_only == False else 0,
        n_threads=threads,
        n_ctx=max_context,
        #n_batch=batch,
        #flash_attn=flash_attention,
        verbose=verbose,
    )
    return generate

@node(name="Safetensors Loader")
def safetensors_loader(
    checkpoint: Literal[*get_dir_files("models.checkpoints", ".safetensors")] = next(iter(get_dir_files("models.checkpoints", ".safetensors")), None),
    model_type: Literal["diffusion", "autoencoder" ,"super_resolution", "token_encoder"] = "diffusion",
    safety: A[bool, Dependent(on="model_type", when="diffusion")] = False,
    device: Literal[*get_gpus()] = next(iter(get_gpus()), None),
    float_32: A[bool, Dependent(on="device", when="cpu")] = False,
    bfloat: A[bool, Dependent(on="float_32", when="False")] = False,
) -> Any:
    print("loading:Safetensors:" + checkpoint)
    if model_type == "token_encoder":
        print("init tokenizer & text encoder")
        tokenizer = CLIPTokenizer.from_pretrained(
            checkpoint,
            subfolder='tokenizer',
        )
        text_encoder = CLIPTextModel.from_pretrained(
            checkpoint,
            subfolder='text_encoder',
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant='fp16',
        ).to(device)

        vectors = [{
            "tokenizer": tokenizer, 
            "text_encoder": text_encoder 
            }]
        return vectors

    elif model_type == "diffusion":
        print("create pipeline")
        pipe_args = {
                'use_safetensors': True,
                'tokenizer':None,
                'text_encoder':None,
                'tokenizer_2':None,
                'text_encoder_2':None,
            } 
        if float_32 != True:
            pipe_args.extend({
                'torch_btype': 'torch.float16' if bfloat == False else 'torch.bfloat16',
                'variant': 'fp16' if bfloat == False else 'bf16',
            })

        if safety is None: pipe_args.extend({'safety_checker': 'None',})

        print("apply pipeline")
        # Load the model on the graphics card
        pipe = AutoPipelineForText2Image.from_pretrained(checkpoint,**pipe_args).to(device)
        return pipe
        
    elif model_type == "vae":
        print("setup vae")
        checkpoint = AutoencoderKL.from_pretrained(
            'madebyollin/sdxl-vae-fp16-fix',
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(device)
        return autoencoder

@node(name="LLM Prompt")
def llm_prompt(
    generate: Any,
    system_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "You're a guru for revealing what you know, yet wiser for revealing what you do not.",
    user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    streaming: bool = True, #triggers generator in next node? 
    advanced_options: bool = False,
        top_k: A[int, Dependent(on="advanced_options", when=True),Slider(min=0, max=100)] = 40,
        top_p: A[float, Dependent(on="advanced_options", when=True), Slider(min=0, max=1, step=0.01, round=0.01)] = 0.95,
        repeat_penalty: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 1,
        temperature: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 0.2,
        max_tokens:  A[int, Dependent(on="advanced_options", when=True),  Numerical(min=0, max=2)] = 256,
) -> I[Any]:
    print("Encoding Prompt")
    z = generate.create_chat_completion(
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
    return z

@node(name="Diffusion Prompt")
def diffusion_prompt(
    pipe: torch.Tensor,
    text_encoder: torch.Tensor,
    text_encoder_2: Any,
    prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "A rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles",
    seed: A[int, Numerical(min=0, max=0xFFFFFFFFFFFFFF, step=1,randomizable=True)]= int(soft_random()),
    device: Literal[*get_gpus()] = get_gpus()[0], # type: ignore
) -> Tuple[torch.Tensor, dict]:
    print("token encode init")
    if queue not in globals(): queue = []
    queue.extend([{
        "prompt": prompt,
        "seed": seed
    }])
    if text_encoder is None: text_encoder = pipe
    tokenizer = text_encoder
    if text_encoder_2 is not None: tokenizer_2 = text_encoder_2

    print("encode prompt")

    def encode_prompt(prompts, tokenizers, text_encoders):
        embeddings_list = []

        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            cond_input = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

            prompt_embeds = text_encoder(cond_input.input_ids.to(device), output_hidden_states=True)

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

    with torch.no_grad():
        for generation in queue:
            generation['embeddings'] = encode_prompt(
            [generation['prompt'], generation['prompt']],
            [tokenizer, tokenizer_2],
            [text_encoder, text_encoder_2],
            )
    
    del tokenizer, text_encoder, tokenizer_2, text_encoder_2
    cache_bin()
    return generation, queue

@node
def diffusion(
    pipe: torch.Tensor,
    generation: torch.Tensor,
    queue: dict,
    device: Literal[*get_gpus()] = get_gpus()[0], # type: ignore
    inference_steps: A[int, Numerical(min=0, max=500, step=1)] = 25,
    cfg_scale: A[float,Slider(min=0.000, max=20.000, step=0.001, round=0.001)] = 5.00,
    height: int = 1024,
    width: int = 1024,
    scheduler: Literal[*get_schedulers()] = "EulerDiscreteScheduler", # type: ignore
    algorithm_type: A[Literal[*get_solvers()], Dependent(on="scheduler", when="DPMSolverMultistepScheduler")] = "dpmsolver++", # type: ignore
    use_karras_sigmas: A[bool, Dependent(on="scheduler", when=("LMSDiscreteScheduler" or "DPMSolverMultistepScheduler"),)] = True,
    solver_order: A[int, Dependent(on="scheduler", when="DPMSolverMultistepScheduler"), Slider(min=1, max=3, step=1)] = 2,
    v_pred: A[bool, Dependent(on="scheduler", when="DDIMScheduler")] = False, 
    timestep_spacing: A[Literal["trailing","linspace","leading"], Dependent(on="scheduler", when="DDIMScheduler")] = "trailing",    
) -> torch.Tensor:
    debug = True
    low_memory = False
    pipe.scheduler = getattr(get_schedulers(), scheduler).from_config( {
        "config": pipe.scheduler.config,
        "use_karras_sigmas": True if use_karras_sigmas==True else False, 
        "rescale_betas_zero_snr": True if v_pred==True else False, #vprediction
        "force_zeros_for_empty_prompt": False if v_pred==True else False, #vprediction
        "solver_order": solver_order if scheduler=="DPMSolverMultistepScheduler" else None, #dpm options
        "prediction_type": "v_prediction" if v_pred==True else "epsilon", #vprediction
        "timestep_spacing": timestep_spacing if timestep_spacing != None else "trailing" #pcm
        # "clip_sample": False, #pcm
        # "set_alpha_to_one": False #pcm
    })
    pipe = pipe.to(device)

    print("lower overhead, select generator")
    if device!="cpu": pipe.enable_model_cpu_offload()
    if low_memory==True: pipe.enable_sequential_cpu_offload()

    def dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
        if step_index == int(pipe.num_timesteps * 0.5):
            callback_kwargs['prompt_embeds'] = callback_kwargs['prompt_embeds'].chunk(2)[-1]
            callback_kwargs['add_text_embeds'] = callback_kwargs['add_text_embeds'].chunk(2)[-1]
            callback_kwargs['add_time_ids'] = callback_kwargs['add_time_ids'].chunk(2)[-1]
            pipe._guidance_scale = 0.0
        return callback_kwargs

    generator = torch.Generator(device=device)

    if debug==True: print("begin queue loop")
    # Start a loop to process prompts one by one

    for i, generation in enumerate(queue, start=1):
    # We start the counter
        generation.image_start = perf_counter()
        # Assign the seed to the generator
        print(generation['seed'])
        seed_planter(generation['seed'])
        generator.manual_seed(generation['seed'])

        generation['latents'] = pipe(
            prompt_embeds=generation['embeddings'][0],
            negative_prompt_embeds =generation['embeddings'][1],
            pooled_prompt_embeds=generation['embeddings'][2],
            negative_pooled_prompt_embeds=generation['embeddings'][3],
            num_inference_steps=inference_steps,
            generator=generator,
            output_type='latent',
            guidance_scale=cfg_scale,
            callback_on_step_end=dynamic_cfg,
            callback_on_step_end_tensor_inputs=['prompt_embeds', 'add_text_embeds', 'add_time_ids'],
        ).images
    del pipe.unet
    cache_bin()
    return generation

@node(name="Autoencode Reverse")
def autoencode(
    pipe: torch.Tensor,
    latent: torch.Tensor,
    queue: Any,
) -> Any:
    with torch.no_grad():
        for i, generation in enumerate(queue, start=1):
            generation['total_time'] = perf_counter() - generation.image_start
            generation['latents'] = generation['latents'].to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

        image = pipe.vae.decode(
        generation['latents'] / pipe.vae.config.scaling_factor,
        return_dict=False,
        )[0]
    # Print the generation time of each image
    images_totals = ', '.join(map(lambda generation: str(round(generation['total_time'], 1)), queue))
    print('Image time:', images_totals)

    # Print the average time
    images_average = round(sum(generation['total_time'] for generation in queue) / len(queue), 1)
    print('Average image time:', images_average)

    if torch.cuda.is_available():
        max_memory = round(torch.cuda.max_memory_allocated(device='cuda') / 1000000000, 2)
        print('Max. memory used:', max_memory, 'GB')
    return image

