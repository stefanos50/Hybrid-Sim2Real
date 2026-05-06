import torch
import cv2
import numpy as np
import os
from glob import glob
import yaml
import torch.nn as nn


# -------------------------------
# Load config
# -------------------------------
with open("options.yaml", "r") as f:
    config = yaml.safe_load(f)

input_dir = config["out_path"]
output_dir = config["out_path"]
checkpoint_path = config["regen_model_path"]

image_height = config.get("image_height")
image_width = config.get("image_width")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(output_dir, exist_ok=True)


# -------------------------------
# NEW MODEL (UNetGenerator)
# -------------------------------
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c),
            nn.ReLU(True),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c)
        )

    def forward(self, x):
        return x + self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.ReLU(True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.InstanceNorm2d(base_ch * 4),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(*[ResBlock(base_ch * 4) for _ in range(4)])

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch, 4, 2, 1),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, in_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        m = self.middle(e3)

        d3 = self.dec3(m)
        d2 = self.dec2(torch.cat([d3, e2], 1))
        out = self.dec1(torch.cat([d2, e1], 1))

        return out


# -------------------------------
# Load model
# -------------------------------
generator_ema = UNetGenerator().to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
generator_ema.load_state_dict(checkpoint)
generator_ema.eval()


# -------------------------------
# Frame loading
# -------------------------------
frame_paths = sorted(glob(os.path.join(input_dir, "*")))
print(f"Found {len(frame_paths)} frames")


# -------------------------------
# Inference loop
# -------------------------------
with torch.no_grad():
    for idx, frame_path in enumerate(frame_paths):

        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Skipping {frame_path}")
            continue

        # BGR -> RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ NEW: match training preprocessing
        img = cv2.resize(img, (960, 544), interpolation=cv2.INTER_LINEAR)

        # Normalize [-1, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Inference
        out = generator_ema(img_tensor)

        # Tensor -> image
        out_img = out[0].cpu().permute(1, 2, 0).numpy()
        out_img = ((out_img * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

        # Optional resize (same as your original)
        if image_height is not None and image_width is not None:
            out_img = cv2.resize(
                out_img,
                (image_width, image_height),
                interpolation=cv2.INTER_LANCZOS4
            )

        # RGB -> BGR
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        filename = os.path.basename(frame_path)
        save_path = os.path.join(output_dir, filename)

        cv2.imwrite(save_path, out_img)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(frame_paths)} frames")

print(f"Saved enhanced frames to {output_dir}")