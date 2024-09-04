from sdbx.nodes.types import *

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

@node
def multi_io_test(
    string: A[str, Text()] = None,
    number: A[int, Numerical(min=0, max=10)] = None
) -> Tuple[str, int]:
    return string, number

@node
def basic_generator(
    n: A[int, Numerical(min=1, max=20)] = 10
) -> I[int]:
    basic_generator.steps = n
    
    for i in range(n):
        print(i)
        yield i

@node(name="LLM Print")
def llm_print(
    response: A[str, Text()]
) -> None:
    print("printing response")
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'role' in delta:
            print(delta['role'], end=': ')
        elif 'content' in delta:
            print(delta['content'], end='')