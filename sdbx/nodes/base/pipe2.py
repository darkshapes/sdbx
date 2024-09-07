from sdbx.config import config
from sdbx.nodes.helpers import softRandom, seedPlanter

from time import perf_counter
import gc
import os

from torch import torch
from diffusers import AutoPipelineForText2Image, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
# import accelerate

if torch.cuda.is_available(): device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else: device = "cpu"

queue = []

queue.extend([{
  "prompt": "A rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles",
  "seed": int(softRandom()),
}])

if debug==True: print("init")
model =  "stabilityai/stable-diffusion-xl-base-1.0"
token_encoder = "stabilityai/stable-diffusion-xl-base-1.0"
# sd_repo = "SG161222/Realistic_Vision_V4.0_noVAE"
#sd_repo = "digiplay/Noosphere_v4.2"


prompt = "woman lying amidst a vast and snowy wilderness, prismatic sunlight, tree branches dressed in white"

if debug==True: print("encode prompt")

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

# ...

tokenizer = CLIPTokenizer.from_pretrained(
  token_encoder,
  subfolder='tokenizer',
)

text_encoder = CLIPTextModel.from_pretrained(
  token_encoder,
  subfolder='text_encoder',
  use_safetensors=True,
  torch_dtype=torch.float16,
  variant='fp16',
).to(device)

tokenizer_2 = CLIPTokenizer.from_pretrained(
  token_encoder,
  subfolder='tokenizer_2',
)

text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
  token_encoder,
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
gc.collect()
if device == "cuda": torch.cuda.empty_cache()
if device == "mps": torch.mps.empty_cache()

low_memory = False
inference_steps=30
fp16 = True
bf16 = False
if fp16==True: set_float = [{
 "torch_dtype": "torch.float16",
  "variant" : "fp16",
  }]
if bf16==True: set_float = [{
 "torch_dtype": "torch.bfloat16",
  "variant" : "bf16",
  }]

if debug==True: print("create pipeline")
# Load the model on the graphics card
pipe = AutoPipelineForText2Image.from_pretrained(
  model,
  use_safetensors=True,
  torch_dtype=set_float["torch_dtype"],
  variant=set_float["variant"],
  tokenizer=None,
  text_encoder=None,
  tokenizer_2=None,
  text_encoder_2=None,
).to(device)

if debug==True: print("lower overhead, select generator")
if gen_device!="cpu" or "mps": pipe.enable_model_cpu_offload()
if low_memory==True: pipe.enable_sequential_cpu_offload()

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
  ).images 
  
del pipe.unet
gc.collect()
if device == "cuda": torch.cuda.empty_cache()
if device == "mps": torch.mps.empty_cache()

vae = AutoencoderKL.from_pretrained(
  'madebyollin/sdxl-vae-fp16-fix',
  use_safetensors=True,
  torch_dtype=torch.float16,
).to(device)

with torch.no_grad():
  for i, generation in enumerate(queue, start=1):
    generation['total_time'] = perf_counter() - image_start
    generation['latents'] = generation['latents'].to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

    image = pipe.vae.decode(
      generation['latents'] / pipe.vae.config.scaling_factor,
      return_dict=False,
    )[0]

    compress_level = 4 # png compression

    file_prefix = "Shadowbox-"
    image = pipe.image_processor.postprocess(image, output_type='pil')[0]
    # Save the image
    if debug==True: print("save")
    counter = format(len(os.listdir(config.get_path("output")))) #file count
    image.save(f'{file_prefix + counter}-batch-{i}.png')

# Print the generation time of each image
images_totals = ', '.join(map(lambda generation: str(round(generation['total_time'], 1)), queue))
print('Image time:', images_totals)

# Print the average time
images_average = round(sum(generation['total_time'] for generation in queue) / len(queue), 1)
print('Average image time:', images_average)

if device == "cuda":
  max_memory = round(torch.cuda.max_memory_allocated(device='cuda') / 1000000000, 2)
  print('Max. memory used:', max_memory, 'GB')
