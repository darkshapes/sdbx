from torch import Generator, manual_seed
from diffusers import StableDiffusionXLPipeline, enable_model_cpu_offload, pipe

@node(Name="Inference")
def inference(
    checkpoint: Llama,
    embeds: Llama,
    scheduler: None,
    unet: Llama,
) -> Image:
    print("⎆Generating:")
    pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=False)
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
        num_images_per_prompt=1,
        generator= torch.Generator(pipe.device).manual_seed(66)).images[0]
    return image
    

if __name__ == '__main__':
    import fire
    fire.Fire(infer)