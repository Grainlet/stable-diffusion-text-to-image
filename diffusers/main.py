import torch
from diffusers import StableDiffusionPipeline

# Optional: Check for GPU
if torch.cuda.is_available():
    !nvidia-smi

# Install dependencies
!pip install diffusers==0.11.1
!pip install transformers scipy ftfy accelerate
!pip install --upgrade huggingface-hub==0.26.2 transformers==4.46.1 tokenizers==0.20.1 diffusers==0.31.0

# Load pipeline (first version)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float16
)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

pipe = pipe.to(device)

# Load another pipeline variant
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")  # Change to "cpu" if needed

# Prompt and image generation
prompt = "huge dinosaur eating a watermelon"
image = pipe(prompt).images[0]
image.save("astronaut_rides_horse.png")
image

# Seeded image generation (reproducible)
generator = torch.Generator(device).manual_seed(1024)
image = pipe(prompt, generator=generator).images[0]
image

# Custom inference steps
image = pipe(prompt, num_inference_steps=15, generator=generator).images[0]
image
