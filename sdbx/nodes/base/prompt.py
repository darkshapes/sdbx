from sdbx.nodes.types import *

from llama_cpp import Llama

@node
def llm_prompt(
    system_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "You are a senior level programmer who gives an accurate and concise examples within the scope of your knowledge, while disclosing when a request goes beyond it.",
    user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "Return python code to access a webcam with as few dependencies as possible.",
    streaming: bool = True
) -> str:
    return llm.create_chat_completion(
        messages=[
                { "role": "system", "content": system_prompt },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
        stream=streaming
    )