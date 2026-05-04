import torch
from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from pathlib import Path
from PIL import Image
import yaml


with open("options.yaml", "r") as f:
    config = yaml.safe_load(f)

input_dir = Path(config["input_path"])
output_dir = Path(config["out_path"])
prompt = config["prompt"]
flux_model = config["flux_model"]
hf_token = config["hf_token"]
image_height = config.get("image_height")
image_width = config.get("image_width")

device = "cuda"
dtype = torch.bfloat16


if not input_dir.exists():
    raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

output_dir.mkdir(parents=True, exist_ok=True)

pipe = Flux2KleinPipeline.from_pretrained(
    flux_model,
    token=hf_token if hf_token else None,
    torch_dtype=dtype
)

pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

for img_path in input_dir.glob("*.*"):
    output_path = output_dir / img_path.name

    #if output_path.exists():
        #print(f"Skipping {img_path.name}, already processed.")
        #continue

    print(f"Processing {img_path.name}...")

    # Load image
    init_image = load_image(str(img_path)).convert("RGB")

    # Run model
    result = pipe(
        prompt=prompt,
        image=init_image,
        guidance_scale=1.0,
        num_inference_steps=2,
        generator=torch.Generator(device=device).manual_seed(0)
    )

    image = result.images[0]

    if image_height is not None and image_width is not None:
        image = image.resize((image_width, image_height), Image.LANCZOS)

    # Save
    image.save(output_path)
    print(f"Saved {output_path}")
