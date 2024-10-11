import gc
import os
from time import perf_counter_ns
import datetime

from diffusers import AutoPipelineForText2Image, AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, FromOriginalModelMixin
from diffusers.schedulers import AysSchedules
from sdbx.config import config
from sdbx.nodes.helpers import seed_planter, soft_random
import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

def tc(clock, string, debug=False): 
    if not debug: 
        return print(f"[ {str(datetime.timedelta(milliseconds=(((perf_counter_ns()-clock)*1e-6))))[:-2]} ] {string}") 

clock = perf_counter_ns() # 00:00:00
tc(clock, " ")

queue = []    
prompt = "A slice of a rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles"
seed = int(soft_random())
queue.extend([{
"prompt": prompt,
"seed": seed,
}])

device = "cuda"
tc(clock,f"encoding prompt with device: {device}")
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

model_path = "C:\\Users\\Public\\models\\metadata\\STA-XL"
# token_encoder = "C:\\Users\\Public\\models\\metadata\\CLI-VL"
# token_encoder_2 = "C:\\Users\\Public\\models\\metadata\\CLI-VG"

tokenizer = CLIPTokenizer.from_pretrained(
    model_path,
    subfolder='tokenizer',
)

text_encoder = CLIPTextModel.from_pretrained(
    model_path,
    subfolder='text_encoder',
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant='fp16',
).to(device)

tokenizer_2 = CLIPTokenizer.from_pretrained(
    model_path,
    subfolder='tokenizer_2',
)

text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    model_path,
    subfolder='text_encoder_2',
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant='fp16',
).to(device)

with torch.no_grad():
    for generation in queue:
        generation['embeddings'] = encode_prompt(
            [generation['prompt'], generation['prompt']],
            [tokenizer, tokenizer_2],
            [text_encoder, text_encoder_2],
    )

    del tokenizer, text_encoder, tokenizer_2, text_encoder_2

    if device == "cuda": torch.cuda.empty_cache()

model_file = "C:\\Users\\Public\\models\\image\\ponyFaetality_v11.safetensors",

pipe = AutoPipelineForText2Image.from_pretrained(
    model_path, torch_dtype=torch.float16, variant="fp16"                
).to(device)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_sequential_cpu_offload()

tc(clock, "set generator")
generator = torch.Generator(device=device)

tc(clock, "begin queue loop...")

for i, generation in enumerate(queue, start=1):
    image_start = perf_counter_ns()                        #start the metric stopwatch
    tc(clock, f"planting seed {generation['seed']}...")
    seed_planter(generation['seed'])
    generator.manual_seed(generation['seed'])
    tc(clock, f"inference device: {device}....")

    generation['latents'] = pipe(
        prompt_embeds=generation['embeddings'][0],
        negative_prompt_embeds =generation['embeddings'][1],
        pooled_prompt_embeds=generation['embeddings'][2],
        negative_pooled_prompt_embeds=generation['embeddings'][3],
        num_inference_steps=20,
        guidance_scale=5,
        generator=generator,
        output_type='latent',
    ).images 

    if device == "cuda": torch.cuda.empty_cache()

tc(clock, "decoding using {vae_path}...")
vae_file = "c:\\Users\\Public\\models\\image\\flatpiecexlVAE_baseonA1579.safetensors",
vae_path = "C:\\Users\\Public\\models\\metadata\\STA-XL\\vae"
vae_config = "C:\\Users\\Public\\models\\metadata\\STA-XL\\vae\\config.json"
#vae = AutoencoderKL.from_single_file(vae_file, config=vae_path, torch_dtype=torch.float16, cache_dir="vae_").to("cuda")
vae = FromOriginalModelMixin.from_single_file(vae_file, config=vae_config).to(device)
pipe.vae=vae
pipe.upcast_vae()

with torch.no_grad():
    counter = [s.endswith('png') for s in os.listdir(config.get_path("output"))].count(True) # get existing images
    for i, generation in enumerate(queue, start=1):
        generation['total_time'] = perf_counter_ns() - image_start
        generation['latents'] = generation['latents'].to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

        image = pipe.vae.decode(
            generation['latents'] / pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]

        image = pipe.image_processor.postprocess(image, output_type='pil')[0]

        tc(clock, "saving")     
        counter += 1
        filename = f"Shadowbox-{counter}-batch-{i}.png"

        image.save(os.path.join(config.get_path("output"), filename)) # optimize=True,


if device == "cuda": torch.cuda.empty_cache()

### METRICS
images_totals = ', '.join(map(lambda generation: str(round(generation['total_time'], 1)), queue))
print('Image time:', images_totals, 'seconds')

images_average = round(sum(generation['total_time'] for generation in queue) / len(queue), 1)
print('Average image time:', images_average, 'seconds')

if device == "cuda":
        max_memory = round(torch.cuda.max_memory_allocated(device='cuda') / 1000000000, 2)
        print('Max. memory used:', max_memory, 'GB')