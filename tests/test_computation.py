import gc
import os
import platform
import datetime
import time
from time import perf_counter
from typing import Tuple, List

import torch
from diffusers import AutoPipelineForText2Image, AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers.schedulers import AysSchedules
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from sdbx.config import config
from sdbx.nodes.helpers import seed_planter, soft_random

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


# Prompt encoder
def encode_prompt(
    prompts: List[str], tokenizers: List[CLIPTokenizer], text_encoders: List[torch.nn.Module], device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings_list = []
    pooled_prompt_embeds = None

    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        cond_input = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        prompt_embeds = text_encoder(cond_input.input_ids.to(device), output_hidden_states=True)
        embeddings_list.append(prompt_embeds.hidden_states[-2])

        pooled_prompt_embeds = prompt_embeds[0]

        prompt_embeds = torch.concat(embeddings_list, dim=-1)

    # Creating negative prompts
    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, 1, 1).view(bs_embed * 1, seq_len, -1)
    negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1).view(1 * 1, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


# Text generation class
class Text2ImageGenerator:
    def __init__(self, prompt: str, seed: int):
        self.device = get_device()
        self.prompt = prompt
        self.seed = seed
        self.token_encoder = "stabilityai/stable-diffusion-xl-base-1.0"

        self.tokenizer = CLIPTokenizer.from_pretrained(self.token_encoder, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.token_encoder, subfolder="text_encoder", use_safetensors=True, torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)

        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.token_encoder, subfolder="tokenizer_2")
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.token_encoder, subfolder="text_encoder_2", use_safetensors=True, torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)

        self.scheduler = EulerAncestralDiscreteScheduler
        self.pipe = None
        self.generator = torch.Generator(device=self.device)

    def configure_pipeline(self):
        model = next(iter(config.get_path_contents("models.diffusers", extension="safetensors")), self.token_encoder)
        pipe_args = {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
        }
        self.pipe = AutoPipelineForText2Image.from_pretrained(model, **pipe_args).to(self.device)

    def prepare_prompt(self):
        queue = [{"prompt": self.prompt, "seed": self.seed}]
        embeddings = encode_prompt(
            [self.prompt, self.prompt], [self.tokenizer, self.tokenizer_2], [self.text_encoder, self.text_encoder_2], self.device
        )
        queue[0]["embeddings"] = embeddings
        return queue

    def run_inference(self, queue, num_inference_steps=8, guidance_scale=5.0, dynamic_guidance=True):
        results = []

        for i, generation in enumerate(queue, start=1):
            image_start = perf_counter()
            seed_planter(generation["seed"])
            self.generator.manual_seed(generation["seed"])

            generation["latents"] = self.pipe(
                prompt_embeds=generation["embeddings"][0],
                negative_prompt_embeds=generation["embeddings"][1],
                pooled_prompt_embeds=generation["embeddings"][2],
                negative_pooled_prompt_embeds=generation["embeddings"][3],
                num_inference_steps=num_inference_steps,
                generator=self.generator,
                output_type="latent",
            ).images

            generation["total_time"] = perf_counter() - image_start
            results.append(generation)

        return results

    def cleanup(self):
        clear_memory_cache(self.device)

    def save_images(self, queue, file_prefix: str = "Shadowbox-"):
        vae_find = "flat"
        compress_level = 4

        # vae_default = "madebyollin/sdxl-vae-fp16-fix.safetensors"
        vae_default = "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl.vae.safetensors"
        vae_path = config.get_path("models.vae")
        autoencoder = next(iter([c for c in config.get_path_contents("models.vae") if vae_find in c]), vae_default)

        vae = AutoencoderKL.from_single_file(autoencoder, torch_dtype=torch.float16, cache_dir="vae_").to(self.device)
        self.pipe.vae = vae
        self.pipe.upcast_vae()

        counter = len(config.get_path_contents("output", extension="png"))

        for i, generation in enumerate(queue, start=1):
            generation["latents"] = generation["latents"].to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

            image = self.pipe.vae.decode(
                generation["latents"] / self.pipe.vae.config.scaling_factor, return_dict=False
            )[0]
            image = self.pipe.image_processor.postprocess(image.detach(), output_type="pil")[0]

            counter += 1
            filename = f"{file_prefix}{counter}-batch-{i}.png"
            image.save(os.path.join(config.get_path("output"), filename), optimize=True)


def test_generation():
    # Setup
    prompt = "A slice of a rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles"
    seed = int(soft_random())

    # Initialize generator
    generator = Text2ImageGenerator(prompt, seed)
    generator.configure_pipeline()

    # Encode and prepare prompt
    queue = generator.prepare_prompt()

    # Run inference
    results = generator.run_inference(queue)

    # Save results
    generator.save_images(results)

    # Clean up resources
    generator.cleanup()

    # Metrics
    total_times = [str(round(generation["total_time"], 1)) for generation in results]
    print("Image time:", ", ".join(total_times), "seconds")

    avg_time = round(sum(generation["total_time"] for generation in results) / len(results), 1)
    print("Average image time:", avg_time, "seconds")

    if generator.device == "cuda":
        max_memory = round(torch.cuda.max_memory_allocated(device="cuda") / 1e9, 2)
        print("Max memory used:", max_memory, "GB")