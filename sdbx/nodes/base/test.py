from sdbx.nodes.types import *
from sdbx.nodes.helpers import getSchedulers, getSolvers, softRandom, seedPlanter, getDirFiles, getDirFilesCount


import torch
import transformers
import diffusers

from transformers import AutoTokenizer, AutoModel
import diffusers

from PIL import Image, ImageOps, ImageSequence, ImageFile
from torch import Generator, manual_seed
from diffusers import (
    StableDiffusionXLPipeline, DiffusionPipeline, UNet2DConditionModel, AutoencoderKL,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler,
    UniPCMultistepScheduler, HeunDiscreteScheduler, DPMSolverMultistepScheduler,
    LMSDiscreteScheduler, DEISMultistepScheduler )
from llama_cpp import Llama

@node
def prints_number(
    number: int
):
    print("prints_number prints:", number)

@node
def prints_string(
    string: str
):
    print("prints_string prints:", string)

@node
def outputs_number(
    number: A[int, Numerical(min=0, max=10)] = None
) -> int:
    return number

@node
def outputs_string(
    string: A[str, Text()] = None
) -> str:
    return string

@node(path="name test", name="name test node")
def name_test(
    string: A[str, Text()] = None
) -> str:
    return string

@node
def basic_generator(
    n: A[int, Numerical(min=1, max=20)] = 10
) -> I[int]:
    basic_generator.steps = n
    
    for i in range(n):
        yield i

@node(name="Kolors Diffusion")
def kolors_diffusion(
    checkpoint: Any,
    embedding: Any,
    unet: Any,
    height: int = 1024,
    width: int = 1024,
    seed: A[int, Numerical(min=0, max=0xFFFFFFFFFFFFFFF, randomizable=True)] = softRandom(),
    steps: A[int, Numerical(min=0, max=1000, step=1)] = 10,
    guidance: A[float, Slider(min=0.0, max=20.0, step=0.01)] = 5.0,
    scheduler: Literal[*getSchedulers()] = "EulerDiscreteScheduler",
    algorithm_type: A[Literal[*getSolvers()], Dependent(on="scheduler", when="DPMSolverMultistepScheduler")] = "dpmsolver++",
    use_karras_sigmas: A[bool, Dependent(on="scheduler", when=("LMSDiscreteScheduler" or "DPMSolverMultistepScheduler"),)] = True,
    solver_order: A[int, Dependent(on="scheduler", when="DPMSolverMultistepScheduler"), Slider(min=1, max=3, step=1)] = 2,
    v_pred: A[bool, Dependent(on="scheduler", when="DDIMScheduler")] = False, 
    timestep_spacing: A[Literal["trailing","linspace","leading"], Dependent(on="scheduler", when="DDIMScheduler")] = "trailing",
) -> Any: # placeholder for latent space denoised tensor
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
        "timestep_spacing": timestep_spacing if timestep_spacing != None else "trailing" #pcm
        # "clip_sample": False, #pcm
        # "set_alpha_to_one": False #pcm
    })
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    seedPlanter(seed)

    image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=n_batch,
            generator= torch.Generator(pipe.device).manual_seed(seed)).images[0]
    return image

@node(name="Diffusion")
def diffusion(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    seed: A[int, Numerical(min=0, max=0xFFFFFFFFFFFFFFF, randomizable=True)] = softRandom,
    steps: A[int, Numerical(min=0, max=1000, step=1)] = 10,
    scheduler: Literal[*getSchedulers()] = "EulerDiscreteScheduler",
    algorithm_type: A[Literal[*getSolvers()], Dependent(on="scheduler", when="DPMSolverMultistepScheduler")] = "dpmsolver++",
    use_karras_sigmas: A[bool, Dependent(on="scheduler", when=("LMSDiscreteScheduler" or "DPMSolverMultistepScheduler"),)] = True,
    solver_order: A[int, Dependent(on="scheduler", when="DPMSolverMultistepScheduler"), Slider(min=1, max=3, step=1)] = 2,
    timestep_spacing: A[Literal["trailing","linspace","leading"], Dependent(on="scheduler", when="DDIMScheduler")] = "trailing",
) -> Any: # placeholder for latent space denoised tensor
    print("Generating")
    print(getattr(schedulerdict, scheduler))
    pipe.scheduler = DPMSolverMultistepScheduler.from_config( {
        "config": pipe.scheduler.config,
        "use_karras_sigmas": True if dpm_karras_sigmas==True or lms_karras_sigmas==True else False, 
        "solver_order": solver_order if scheduler=="DPMSolverMultistepScheduler" else None, #dpm options
        "timestep_spacing": timestep_spacing if timestep_spacing != None else "trailing" #pcm
    })
    pipe.device="cpu"
    seedPlanter(seed)
    
    generator = torch.Generator(pipe.device).manual_seed(seed)
    pipe.enable_model_cpu_offload()

    image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            generator=generator,
    ).images[0]
    return image

@node(name="Save/Preview Image")
def save_image(
    image: str, # placeholder for Image type
    metadata: str, # placeholder for JSON type
    filename_prefix: Annotated[str, Text()] = "Shadowbox-",
) -> None:
    if temp == True:
        type = "temp"
        prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        compress_level = 1
    else:
        type = "output"
        prefix_append = ""
        compress_level = 4
    counter = getDirFiles("output", ".png")
    counter = counter + getDirFiles("output", ".jpg")
    counter = format(len(counter))

    results = list()
    for (batch_number, image) in enumerate(images):
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_prefix + prefix_append}-{filename_with_batch_num}_{counter:05}_.png"
        img.save(os.path.join(config.get_path("output"), file), pnginfo=metadata, compress_level=self.compress_level)
        results.append({
            "abs_path": os.path.abspath(abs_path),
            "filename": file,
        })
        counter += 1

    return results


@node(name="Kolors Diffusion Prompt")
def kolors_diffusion_prompt(
    checkpoint: Any,
    encoder: Llama,
    encoder_2: Any = None, # when input is attached, needs to make other _2 options show
    lora_scale: float = 0.00,
    prompt : A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    negative_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    clip_skip: A[int, Slider(min=0, max=3)] = 0,
    prompt_2: A[str, Dependent(on="encoder_2", when=(not None)), Text(multiline=True, dynamic_prompts=True)] = None,
    negative_prompt_2: A[str, Dependent(on="encoder_2", when=(not None)), Text(multiline=True, dynamic_prompts=True)] = None,
) -> Any: # placeholder for latent space embeddings
    print("Encoding Prompt")
    embedding = encoder.diffusion_prompt(prompt)  # returns embeds for 1)prompt, 2)negative, 3)pooled prompt, 4)negative pooled
    return embedding


@node(name="SDXL Loader")
def safetensors_loader(
    checkpoint: Literal[*getDirFiles("models.checkpoints", ".safetensors")] = getDirFiles("models.checkpoints", ".safetensors")[0],
) -> Any:
    checkpoint = "" + os.path.join(config.get_path("models.checkpoints"), checkpoint)
    print(f"loading:Safetensors '/Users/Shared/ouerve/recent/darkshapes/models/checkpoints/ponyFaetality_v11.safetensors'" )
    return {"pipe" : AutoPipelineForText2Image.from_single_file(
        '/Users/Shared/ouerve/recent/darkshapes/models/checkpoints/ponyFaetality_v11.safetensors',
        torch_dtype=torch.float16,
        variant="fp16",
    ) }

    