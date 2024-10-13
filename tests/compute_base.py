import os
from sdbx.nodes.helpers import soft_random, seed_planter
from sdbx.config import config

import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, AutoencoderKL, UNet2DConditionModel #EulerAncestralDiscreteScheduler StableDiffusionXLPipeline
from diffusers.schedulers import AysSchedules
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

queue = []    
prompt = "A slice of a rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles"
seed = soft_random()
queue.extend([{
"prompt": prompt,
"seed": seed,
}])

device = "cuda"
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

        prompt_embeds = text_encoder(cond_input.input_ids.to("cuda"), output_hidden_states=True)

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
model = "C:\\Users\\Public\\models\\metadata\\STA-XL"

tokenizer = CLIPTokenizer.from_pretrained(
    model,
    subfolder='tokenizer',
    local_files_only=True,
)

text_encoder = CLIPTextModel.from_pretrained(
    model,
    subfolder='text_encoder',
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant='fp16',
    local_files_only=True,
).to(device)

tokenizer_2 = CLIPTokenizer.from_pretrained(
    model,
    subfolder='tokenizer_2',
    local_files_only=True,
)

text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    model,
    subfolder='text_encoder_2',
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant='fp16',
    local_files_only=True,
).to(device)

with torch.no_grad():
    for generation in queue:
        generation['embeddings'] = encode_prompt(
            [generation['prompt'], generation['prompt']],
            [tokenizer, tokenizer_2],
            [text_encoder, text_encoder_2],
    )

del tokenizer, text_encoder, tokenizer_2, text_encoder_2

torch.cuda.empty_cache()
max_memory = round(torch.cuda.max_memory_allocated(device=device) / 1e9, 2)
print('Max. memory used:', max_memory, 'GB')

vae_file = "C:\\Users\\Public\\models\\image\\flatpiecexlVAE_baseonA1579.safetensors"
config_file = "C:\\Users\\Public\\models\\metadata\\STA-XL\\vae\\config.json"
vae = AutoencoderKL.from_single_file(vae_file, config=config_file, local_files_only=True,  torch_dtype=torch.float16, variant="fp16").to("cuda")

pipe = AutoPipelineForText2Image.from_pretrained(
    model,
    torch_dtype=torch.float16, 
    variant="fp16",
    tokenizer = None,
    text_encoder = None,
    tokenizer_2 = None,
    text_encoder_2 = None,
    local_files_only=True,
    vae = vae #"C:\\Users\\Public\\models\\image\\flatpiecexlVAE_baseonA1579.safetensors"
    ).to("cuda")

ays = AysSchedules["StableDiffusionXLTimesteps"]

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")
pipe.enable_sequential_cpu_offload()

generator = torch.Generator(device=device)

for i, generation in enumerate(queue, start=1):
    seed_planter(generation['seed'])
    generator.manual_seed(generation['seed'])

    generation['latents'] = pipe(
        prompt_embeds=generation['embeddings'][0],
        negative_prompt_embeds =generation['embeddings'][1],
        pooled_prompt_embeds=generation['embeddings'][2],
        negative_pooled_prompt_embeds=generation['embeddings'][3],
        num_inference_steps=10,
        timesteps=ays,
        guidance_scale=5,
        generator=generator,
        output_type='latent',
    ).images 

torch.cuda.empty_cache()

pipe.upcast_vae()
output_dir = config.get_path("output")
with torch.no_grad():
    counter = [s.endswith('png') for s in output_dir].count(True) # get existing images
    for i, generation in enumerate(queue, start=1):
        generation['latents'] = generation['latents'].to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

        image = pipe.vae.decode(
            generation['latents'] / pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]

        image = pipe.image_processor.postprocess(image, output_type='pil')[0]

        counter += 1
        filename = f"Shadowbox-{counter}-batch-{i}.png"

        image.save(os.path.join(config.get_path("output"), filename)) # optimize=True,


torch.cuda.empty_cache()
max_memory = round(torch.cuda.max_memory_allocated(device='cuda') / 1e9, 2)
print('Max. memory used:', max_memory, 'GB')