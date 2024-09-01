from sdbx.nodes.types import *
from llama_cpp import Llama
from sdbx.nodes.trndup import softRndmc

"""
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
"""

@node
def llm_loader(
    threads: A[int, Slider(min=0, max=10, step=1)] = 8,
    max_context: A[int, Numerical(min=0, max=32767, step=64)] = 2048,
    repeat_penalty: A[float, Numerical(min=0, max=2, step=0.01)] = 1,
    temperature: A[float, Numerical(min=0, max=2, step=0.01)] = 0.2
) -> str:
    rndmc = softRndmc.softRandom()
    print(rndmc)
    Llama(
        model_path="/Users/Shared/ouerve/recent/darkshapes/models/llms/codeninja-1.0-openchat-7b.Q5_K_M.gguf",
        # n_gpu_layers=-1, # Uncomment to use GPU acceleration
        n_threads=threads,   # The number of CPU threads to use, tailor to your system and the resulting performance
        seed=rndmc, # Uncomment to set a specific seed
        n_ctx=max_context, # Uncomment to increase the context window
        chat_format="openchat",
        repeat_penalty=repeat_penalty,
        temperature=temperature,
    )

@node
def llm_prompt(
    system_prompt: Text(multiline=True, dynamic_prompts=True) = "You are a senior level programmer who gives an accurate and concise examples within the scope of your knowledge, while disclosing when a request goes beyond it.",
    user_prompt: Text(multiline=True, dynamic_prompts=True) = "Return python code to access a webcam with as few dependencies as possible.",
    streaming: bool = True
) -> str:
    response = llm.create_chat_completion(
    messages=[
        { "role": "system", "content": system_prompt },
        {
            "role": "user",
            "content": user_prompt
        }
    ],
    stream=streaming
    )

@node
def llm_output(
    str: Text()
) -> None:
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'role' in delta:
            stream = print(delta['role'], end=': ')
            print(stream)
            return stream
        elif 'content' in delta:
            stream = delta['content'], end=''
            print(stream)
            return stream