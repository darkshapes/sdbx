from sdbx.nodes.types import *
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModel
from diffusers import encode_prompt

@node(name="LLM Prompt")
def llm_prompt(
    llama: Llama,
    streaming: bool = True,
    top_k: A[int, Slider(min=0, max=100)] = 40,
    top_p: A[float, Slider(min=0, max=1, step=0.01)] = 0.95,
    repeat_penalty: A[float, Numerical(min=0.0, max=2.0, step=0.01)] = 1,
    temperature: A[float, Numerical(min=0.0, max=2.0, step=0.01)] = 0.2,
    max_tokens:  A[int, Numerical(min=0, max=2)] = 256,
    system_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "You're a guru for revealing what you know, yet wise for revealing what you do not.",
    user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
) -> str:
    print("⎆Adding Prompt:")
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

def text_encode(
    checkpoint: Llama,
    encoder: Llama,
    lora_scale: A[float],
    prompt : A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    negative_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    clip_skip: A[int, Slider(min=0, max=3)] = 0,
    second_encoder: bool = False,
    encoder_2: A[Llama, Dependent(on="second_encoder", when=True)] = None,
    prompt_2: A[str, Dependent(on="second_encoder", when=True), Text(multiline=True, dynamic_prompts=True)] = None,
    negative_prompt_2: A[str, Dependent(on="second_encoder", when=True), Text(multiline=True, dynamic_prompts=True)] = None,
) -> embeddings:
    print("⎆Encoding Prompt")
    encode = encoder.encode_prompt(prompt)  # return for encode_prompt= prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    return encode

