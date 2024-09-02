from sdbx.nodes.types import *
from sdbx.nodes.helpers import getDir

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
    print("Printing Resposnse")
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'role' in delta:
            print(delta['role'], end=': ')
        elif 'content' in delta:
            print(delta['content'], end='')

@node(name="Save/Preview Image")
def save_image(
    image: Image,
    filename_prefix: Annotated[str, Text()] = "shadowbox-",

) -> None:
    if temp == True:
        type = "temp"
        prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        compress_level = 1
    else:
        type = "output"
        prefix_append = ""
        compress_level = 4
    counter = getDirFilesCount("output", (".png", ".jpg"))

    results = list()
    for (batch_number, image) in enumerate(images):
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_prefix + prefix_append}-{getDirFiles("output")}_{counter:05}_.png"
        img.save(os.path.join(config.get_path("output"), file), pnginfo=metadata, compress_level=self.compress_level)
        results.append({
            "abs_path": os.path.abspath(abs_path),
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        counter += 1

    return { "ui": { "images": results } }