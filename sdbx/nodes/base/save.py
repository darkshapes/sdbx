from sdbx.nodes.types import *
from sdbx.config import config
import os
import PIL

@node(name="LLM Print")
def llm_print(
    response: str
) -> I[str]:
    print("Calculating Resposnse")
    for chunk in range(response):
        delta = chunk['choices'][0]['delta']
        # if 'role' in delta:               # this prints assistant: user: etc
            # print(delta['role'], end=': ')
            #yield (delta['role'], ': ')
        if 'content' in delta:              # the response itself
            print(delta['content'], end='')
            yield delta['content'], ''

@node(name="Save / Preview Image", display=True)
def save_preview_img(
    pipe: Any,
    file_prefix: A[str, Text(multiline=False)]= "Shadowbox-",
    compress_level: A[int, Slider(min=0, max=4, step=1)]= 4,
    temp: bool = False,
) -> I[Any]:
        image = pipe.image_processor.postprocess(image, output_type='pil')[0]
        # Save the image
        counter = format(len(os.listdir(config.get_path("output")))) #file count
        file_prefix = os.path.join(config.get_path("output"), file_prefix)
        image.save(f'{file_prefix + counter}.png')
        print("Complete.")
        yield image