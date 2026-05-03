import torch
import cv2
import numpy as np
import os
from glob import glob
import yaml
from generator import define_G


with open("options.yaml", "r") as f:
    config = yaml.safe_load(f)

input_dir = config["out_path"]
output_dir = config["out_path"]
checkpoint_path = config["regen_model_path"]

image_height = config.get("image_height")
image_width = config.get("image_width")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(output_dir, exist_ok=True)


def make_generator():
    return define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="global",
        norm="instance",
        n_downsample_global=4,
        n_blocks_global=9,
        n_local_enhancers=0
    ).to(device)

generator_ema = make_generator()


checkpoint = torch.load(checkpoint_path, map_location=device)
generator_ema.load_state_dict(checkpoint)
generator_ema.eval()


frame_paths = sorted(glob(os.path.join(input_dir, "*")))
print(f"Found {len(frame_paths)} frames")


with torch.no_grad():
    for idx, frame_path in enumerate(frame_paths):

        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Skipping {frame_path}")
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Inference
        out = generator_ema(img_tensor)

        # Tensor -> image
        out_img = out[0].cpu().permute(1, 2, 0).numpy()
        out_img = ((out_img + 1) * 127.5).clip(0, 255).astype(np.uint8)

        if image_height is not None and image_width is not None:
            out_img = cv2.resize(out_img, (image_width, image_height), interpolation=cv2.INTER_LANCZOS4)

        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        filename = os.path.basename(frame_path)
        save_path = os.path.join(output_dir, filename)

        cv2.imwrite(save_path, out_img)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(frame_paths)} frames")

print(f"Saved enhanced frames to {output_dir}")
