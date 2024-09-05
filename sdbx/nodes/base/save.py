from sdbx.nodes.types import *
from sdbx.nodes.helpers import getDirFiles, getDirFilesCount
from PIL import Image, ImageOps, ImageSequence, ImageFile

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
