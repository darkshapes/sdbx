from sdbx.nodes.types import *
from sdbx.nodes.helpers import getDirFiles, getDirFilesCount

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

@node(name="LLM Print")
def llm_print(
    response: A[str, Text()]
) -> None:
    print("Calculating Resposnse")
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'role' in delta:
            print(delta['role'], end=': ')
        elif 'content' in delta:
            print(delta['content'], end='')

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