# encoder-decoder/infer.py
import os
import csv
import yaml
from pathlib import Path

import torch
from PIL import Image
import numpy as np

# ====== CONSTANTS (no CLI) ======
EXP_NAME        = "encdec_sr_v1"   # must match training
IMAGE_SIZE      = 512              # expected LR/HR size
SAVE_ERROR_TILE = False            # if True and HR exists, append |abs(SR-HR)| as 4th tile
# =================================

from models.nafnet import NAFNetU

# --- tiny helpers ---
def load_paths_yaml(path="configs/paths.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_rgb(path: str, size: int = IMAGE_SIZE) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.array(img, dtype=np.uint8)              # H,W,3 (0..255)
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x.clamp_(0.0, 1.0)

def to_pil(x: torch.Tensor) -> Image.Image:
    x = x.clamp(0.0, 1.0).detach().cpu()
    x = (x * 255.0).round().byte().permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(x, mode="RGB")

def save_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))

def make_row(images, tile_w=IMAGE_SIZE, tile_h=IMAGE_SIZE, pad=6, bgcolor=(20, 20, 20)) -> Image.Image:
    n = len(images)
    W = n * tile_w + (n - 1) * pad
    H = tile_h
    canvas = Image.new("RGB", (W, H), bgcolor)
    x = 0
    for im in images:
        if im.size != (tile_w, tile_h):
            im = im.resize((tile_w, tile_h), Image.BICUBIC)
        canvas.paste(im, (x, 0))
        x += tile_w + pad
    return canvas

@torch.no_grad()
def psnr(x, y, eps=1e-8):
    # x,y in [0,1], tensors
    mse = torch.mean((x - y) ** 2).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths & output dirs ---
    paths = load_paths_yaml()  # expects: manifest, (optional) out_dir
    manifest_path = paths["manifest"]
    out_root = Path(f"outputs/{EXP_NAME}")
    out_final = out_root / "final"
    out_panels = out_root / "panels"
    out_final.mkdir(parents=True, exist_ok=True)
    out_panels.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    ckpt_path = Path(f"checkpoints/{EXP_NAME}/ckpt.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)

    # Hard-code architecture to match training (ignore ckpt['arch'] if present)
    model = NAFNetU(in_ch=3, base=64, out_ch=3, residual=True, use_attn=True).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # --- Iterate manifest ---
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hr_path = row.get("hr_path")
            lr_path = row.get("lr_path")
            stem = Path(hr_path or lr_path).stem

            # Load LR (and HR if available)
            lr = load_rgb(lr_path, IMAGE_SIZE).to(device)
            hr = None
            if hr_path and os.path.exists(hr_path):
                hr = load_rgb(hr_path, IMAGE_SIZE).to(device)

            # Inference
            sr = model(lr.unsqueeze(0)).clamp(0.0, 1.0)[0]  # [3,H,W]

            # Prepare panel tiles
            lr_img = to_pil(lr)
            sr_img = to_pil(sr)
            tiles = [lr_img, sr_img]

            if hr is not None:
                hr_img = to_pil(hr)
                tiles.append(hr_img)
                if SAVE_ERROR_TILE:
                    err = (sr - hr).abs().mean(0, keepdim=True).repeat(3, 1, 1)  # simple mean-abs error heat (grayscale)
                    # normalize to [0,1] for display
                    e = err / (err.max().clamp(min=1e-6))
                    tiles.append(to_pil(e))

                # Print PSNR
                p = psnr(sr, hr).item()
                print(f"{stem}: PSNR={p:.2f} dB")

            panel = make_row(tiles)

            # Save
            save_image(sr_img, out_final / f"{stem}.png")
            save_image(panel,  out_panels / f"{stem}_panel.png")

            print(f"Saved → final: {out_final / (stem + '.png')} | panel: {out_panels / (stem + '_panel.png')}")

if __name__ == "__main__":
    main()
