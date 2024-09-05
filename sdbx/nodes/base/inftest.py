from diffusers import AutoPipelineForText2Image
import transformers
import accelerate
import torch
from sdbx.nodes.helpers import softRandom
from time import perf_counter
import config

model = "ponyFaetality_v11"
seed = int(softRandom())
filename_prefix = "Shadowbox-"
print(seed)
queue = []

queue.extend([{
    'prompt': 'lots of balloons',
    'seed' : seed,
}])

from time import perf_counter
# Import libraries
# import ...

# Define prompts
# queue = []
# queue.extend ...

for i, generation in enumerate(queue, start=1):
  # We start the counter
  image_start = perf_counter()

  # Generate and save image
  # ...

  # We stop the counter and save the result
  generation['total_time'] = perf_counter() - image_start

# Print the generation time of each image
images_totals = ', '.join(map(lambda generation: str(round(generation['total_time'], 1)), queue))
print('Image time:', images_totals)

# Print the average time
images_average = round(sum(generation['total_time'] for generation in queue) / len(queue), 1)
print('Average image time:', images_average)

max_memory = round(torch.cuda.device.max_memory_allocated(device='cpu') / 1000000000, 2)
print('Max. memory used:', max_memory, 'GB')

# Load the model on the graphics card
pipe = AutoPipelineForText2Image.from_pretrained(
    "" + config.get_path("models.checkpoints") + model,
    use_safetensors=True,
    torch_dtype=torch.float16,
    #variant='fp16',
)
# Create a generator
generator = torch.Generator(device='cuda')

# Start a loop to process prompts one by one
for i, generation in enumerate(queue, start=1):
  # Assign the seed to the generator
  generator.manual_seed(generation['seed'])

  # Create the image
  image = pipe(
    prompt=generation['prompt'],
    generator=generator,
  ).images[0]
  # Save the image
  image.save(os.path.join(config.get_path("output"), prefix + i +'.png'))