from sdbx.nodes.types import *

from llama_cpp import Llama

@node(name="LLM Prompt")
def llm_prompt(
    llama: Llama,
    streaming: bool = True,
    top_k: A[int, Slider(min=0, max=100)] = 40,
    top_p: A[float, Slider(min=0, max=1, step=0.01)] = 0.95,
    repeat_penalty: A[float, Numerical(min=0, max=2, step=0.01)] = 1,
    temperature: A[float, Numerical(min=0.0, max=2.0, step=0.01)] = 0.2,
    max_tokens:  A[int, Numerical(min=0, max=2, step=1)] = 256,
    system_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "You are a senior level programmer who gives an accurate and concise examples within the scope of your knowledge, while disclosing when a request goes beyond it.",
    user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "Return python code to access a webcam with as few dependencies as possible.",
) -> str:
    print("Prompting:")
    return llama.create_chat_completion(
        messages=[
                { "role": "system", "content": system_prompt },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
        stream=streaming,
        repeat_penalty=repeat_penalty,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
    )