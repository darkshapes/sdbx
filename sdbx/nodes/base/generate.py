from sdbx.nodes.types import *
from sdbx.nodes.helpers import getSchedulers, getSolvers, softRandom
from torch import Generator, manual_seed
from diffusers import (
    DiffusionPipeline, UNet2DConditionModel, AutoencoderKL,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler,
    UniPCMultistepScheduler, HeunDiscreteScheduler, DPMSolverMultistepScheduler,
    LMSDiscreteScheduler, DEISMultistepScheduler )
from llama_cpp import Llama

@node(name="Diffusion")
def diffusion(
    checkpoint: Llama,
    embeds: str,
    unet: Llama,
    height: int = 1024,
    width: int = 1024,
    seed: A[int, Numerical(min=0, max=0xFFFFFFFFFFFFFFF, randomizable=True)] = softRandom,
    steps: A[int, Numerical(min=0, max=1000, step=1)] = 10,
    guidance: A[float, Slider(min=0.0, max=20.0, step=0.01)] = 5.0,
    scheduler: Literal[*getSchedulers()] = "EulerDiscreteScheduler",
    algorithm_type: A[Literal[*getSolvers()], Dependent(on="scheduler", when="DPMSolverMultistepScheduler")] = "dpmsolver++",
    use_karras_sigmas: A[bool, Dependent(on="scheduler", when=("LMSDiscreteScheduler" or "DPMSolverMultistepScheduler"),)] = True,
    solver_order: A[int, Dependent(on="scheduler", when="DPMSolverMultistepScheduler"), Slider(min=1, max=3, step=1)] = 2,
    v_pred: A[bool, Dependent(on="scheduler", when="DDIMScheduler")] = False, 
    timestep_spacing: A[Literal["trailing",], Dependent(on="scheduler", when="DDIMScheduler")] = "trailing",
) -> str: # placeholder for latent space denoised tensor
    print("Generating")
    pipe = checkpoint
    pipe = DiffusionPipeline.from_pretrained(use_safetensors=True)
    pipe.scheduler = getattr(schedulerdict, scheduler).from_config( {
        "config": pipe.scheduler.config,
        "use_karras_sigmas": True if dpm_karras_sigmas==True or lms_karras_sigmas==True else False, 
        "rescale_betas_zero_snr": True if v_pred==True else False, #vprediction
        "force_zeros_for_empty_prompt": False if v_pred==True else False, #vprediction
        "solver_order": solver_order if scheduler=="DPMSolverMultistepScheduler" else None, #dpm options
        "prediction_type": "v_prediction" if v_pred==True else "epsilon", #vprediction
        "timestep_spacing": "trailing" if timestep_spacing=="trailing" else None, #pcm
        # "clip_sample": False, #pcm
        # "set_alpha_to_one": False #pcm
    })
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=n_batch,
            generator= torch.Generator(pipe.device).manual_seed(softRandom())).images[0]
    return image

    