from PIL import Image

from sdbx.nodes.types import *

@node
def outputs_number(
    number: A[int, Numerical(min=0, max=10)] = None
) -> int:
    return number

@node
def outputs_string(
    string: str = None
) -> str:
    return string

@node(display=True)
def displays_number(
    number: int
):
    print("prints_number prints:", number)

@node(display=True)
def displays_string(
    string: str
):
    print("prints_string prints:", string)

@node(display=True)
def displays_test_image(
    color: str = "red"
) -> Image:
    return Image.new('RGB', (512, 512), color=color)

@node(path="test", name="Name Test Node")
def name_test(
    string: A[str, Text()] = None
) -> str:
    return string

@node(display=True)
def basic_generator(
    n: A[int, Numerical(min=1, max=20)] = 10
) -> I[int]:
    import time
    basic_generator.steps = n
    
    for i in range(n):
        yield i
        time.sleep(1)