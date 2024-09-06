from sdbx.nodes.types import *
from sdbx.nodes.helpers import softRandom, getGPUs

from torch import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from llama_cpp import Llama

@node(name="LLM Prompt")
def llm_prompt(
    llama: Llama,
    system_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "You're a guru for revealing what you know, yet wiser for revealing what you do not.",
    user_prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "",
    streaming: bool = True, #triggers generator in next node? 
    advanced_options: bool = False,
        top_k: A[int, Dependent(on="advanced_options", when=True),Slider(min=0, max=100)] = 40,
        top_p: A[float, Dependent(on="advanced_options", when=True), Slider(min=0, max=1, step=0.01, round=0.01)] = 0.95,
        repeat_penalty: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 1,
        temperature: A[float, Dependent(on="advanced_options", when=True), Numerical(min=0.0, max=2.0, step=0.01, round=0.01)] = 0.2,
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

@node(name="Diffusion Prompt")
def diffusion_prompt(
    pipe: torch.Tensor or Llama,
    text_encoder: torch.Tensor or Llama = None,
    text_encoder_2: torch.Tensor or Llama = None,
    prompt: A[str, Text(multiline=True, dynamic_prompts=True)] = "A rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles",
    seed: A[int, Numerical(min=0, max=0xFFFFFFFFFFFFFF, step=1,randomizable=True)]= int(softRandom()),
    device: Literal[*getGPUs()] = getGPUs()[0],
) -> Tuple[torch.Tensor, dict]:
    if debug==True: print("token encode init")
    if queue not in globals(): queue = []
    queue.extend([{
        "prompt": prompt,
        "seed": seed
    }])
    if text_encoder is None: text_encoder = pipe
    tokenizer = text_encoder
    if text_encoder_2 is not None: tokenizer_2 = text_encoder_2

    if debug==True: print("encode prompt")

    def encode_prompt(prompts, tokenizers, text_encoders):
        embeddings_list = []

        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            cond_input = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            )

            prompt_embeds = text_encoder(cond_input.input_ids.to(device), output_hidden_states=True)

            pooled_prompt_embeds = prompt_embeds[0]
            embeddings_list.append(prompt_embeds.hidden_states[-2])

        prompt_embeds = torch.concat(embeddings_list, dim=-1)

        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(1 * 1, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    with torch.no_grad():
        for vectors in queue:
            vectors['embeddings'] = encode_prompt(
            [vectors['prompt'], vectors['prompt']],
            [tokenizer, tokenizer_2],
            [text_encoder, text_encoder_2],
            )
    
    del tokenizer, text_encoder, tokenizer_2, text_encoder_2
    cacheBin()
    return vectors, queue