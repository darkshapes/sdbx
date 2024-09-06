from sdbx.nodes.types import *
from sdbx.config import config
import os
import PIL

@node(name="LLM Print")
def llm_print(
    response: A[str, Text()]
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

@node(name="Save / Preview Image")
def save_preview_img(
    image: Any,
    file_prefix: A[Text(multiline=False)]= "Shadowbox-",
    compress_level: A[int, Slider(min=0,max=4, step=1)]= 4,
) -> None:
        image = pipe.image_processor.postprocess(image, output_type='pil')[0]
        # Save the image
        if debug==True: print("save")
        counter = format(len(os.listdir(config.get_path("output")))) #file count
        image.save(f'{file_prefix + counter}.png')

