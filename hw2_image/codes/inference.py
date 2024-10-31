from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("./ckpts", use_safetensors=True)

# pipeline.to("cuda")

image = pipeline().images[0]

image.show()