from sdbx.nodes.types import *
from sdbx.nodes.helpers import getSchedulers
from torch import Generator, manual_seed
from diffusers import StableDiffusionXLPipeline
from llama_cpp import Llama
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, DPMSolverMultistepScheduler

@node(name="Inference")
def inference(
    checkpoint: Llama,
    embeds: str,
    unet: Llama,
    scheduler: Literal[*getSchedulers()] = "EulerDiscreteScheduler",
) -> str: # placeholder for latent space denoised tensor
    print("⎆Generating:")
    pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler+"Scheduler",
            force_zeros_for_empty_prompt=False)
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
        num_images_per_prompt=1,
        generator= torch.Generator(pipe.device).manual_seed(66)).images[0]
    return image

    