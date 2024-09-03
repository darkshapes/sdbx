from sdbx.nodes.types import *
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModel
import diffusers

@node(name="LLM Prompt")
def llm_prompt(
    llama: Llama,
    system_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "You're a guru for revealing what you know, yet wiser for revealing what you do not.",
    user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    streaming: bool = True, #triggers generator in next node?
    advanced_options: bool = False,
        top_k: A[int, Dependent(on="advanced_options", when=True),Slider(min=0, max=100)] = 40,
        top_p: A[float, Dependent(on="advanced_options", when=True), Slider(min=0, max=1, step=0.01)] = 0.95,
        repeat_penalty: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01)] = 1,
        temperature: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01)] = 0.2,
        max_tokens:  A[int, Dependent(on="advanced_options", when=True),  Numerical(min=0, max=2)] = 256,
) -> str:
    print("Encoding Prompt")
    return llama.create_chat_completion(
        messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
        stream=streaming,
        repeat_penalty=repeat_penalty,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
    )

@node
def text_encode(
    checkpoint: Llama,
    encoder: Llama,
    # encoder_2: when input is attached, needs to make other _2 options show
    lora_scale: float = 0.00,
    prompt : A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    negative_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    clip_skip: A[int, Slider(min=0, max=3)] = 0,
    prompt_2: A[str, Dependent(on="encoder_2", when=True), Text(multiline=True, dynamic_prompts=True)] = None,
    negative_prompt_2: A[str, Dependent(on="encoder_2", when=True), Text(multiline=True, dynamic_prompts=True)] = None,
) -> str: # placeholder for latent space embeddings
    print("Encoding Prompt")
    encode = encoder.encode_prompt(prompt)  # returns embeds for 1)prompt, 2)negative, 3)pooled prompt, 4)negative pooled
    return encode

