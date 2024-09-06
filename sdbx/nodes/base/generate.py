from sdbx.nodes.types import *
from sdbx.config import config
from sdbx.nodes.helpers import softRandom, seedPlanter, getGPUs, cacheBin, getSchedulers, getSolvers

from time import perf_counter
import os

from torch import torch
from diffusers import AutoPipelineForText2Image, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
# import accelerate


@node(name="Diffusion")
def diffusion(
    pipe: torch.Tensor,
    vectors: torch.Tensor,
    queue: dict,
    device: Literal[*getGPUs()] = getGPUs()[0],
    inference_steps: A[int, Numerical(min=0, max=500, step=1)] = 25,
    cfg_scale: A[float,Slider(min=0.000, max=20.000, step=0.001, round=0.001)] = 5.00,
    height: int = 1024,
    width: int = 1024,
    scheduler: Literal[*getSchedulers()] = "EulerDiscreteScheduler",
    algorithm_type: A[Literal[*getSolvers()], Dependent(on="scheduler", when="DPMSolverMultistepScheduler")] = "dpmsolver++",
    use_karras_sigmas: A[bool, Dependent(on="scheduler", when=("LMSDiscreteScheduler" or "DPMSolverMultistepScheduler"),)] = True,
    solver_order: A[int, Dependent(on="scheduler", when="DPMSolverMultistepScheduler"), Slider(min=1, max=3, step=1)] = 2,
    v_pred: A[bool, Dependent(on="scheduler", when="DDIMScheduler")] = False, 
    timestep_spacing: A[Literal["trailing","linspace","leading"], Dependent(on="scheduler", when="DDIMScheduler")] = "trailing",    
) -> torch.Tensor:
    debug = True
    low_memory = False
    pipe.scheduler = getattr(schedulerdict, scheduler).from_config( {
        "config": pipe.scheduler.config,
        "use_karras_sigmas": True if dpm_karras_sigmas==True or lms_karras_sigmas==True else False, 
        "rescale_betas_zero_snr": True if v_pred==True else False, #vprediction
        "force_zeros_for_empty_prompt": False if v_pred==True else False, #vprediction
        "solver_order": solver_order if scheduler=="DPMSolverMultistepScheduler" else None, #dpm options
        "prediction_type": "v_prediction" if v_pred==True else "epsilon", #vprediction
        "timestep_spacing": timestep_spacing if timestep_spacing != None else "trailing" #pcm
        # "clip_sample": False, #pcm
        # "set_alpha_to_one": False #pcm
    })
    pipe = pipe.to(device)

    if debug==True: print("lower overhead, select generator")
    if device=="cpu": pipe.enable_model_cpu_offload()
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
        image_start = perf_counter()
        # Assign the seed to the generator
        print(generation['seed'])
        seedPlanter(generation['seed'])
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
    cacheBin()
    return generation

@node(name="Autoencode Reverse")
def autoencode(
    pipe: torch.Tensor,
    latent: torch.Tensor,
) -> Any:
    with torch.no_grad():
        for i, generation in enumerate(queue, start=1):
            generation['total_time'] = perf_counter() - image_start
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

