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