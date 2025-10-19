# minimal-satSR/infer.py
import os
import csv
import yaml
import math
from pathlib import Path

import torch
from PIL import Image

# ==== CONSTANTS (no CLI) ====
EXP_NAME   = "minimal_satSR_v1"  # must match what you trained/saved
STEPS      = 25                  # number of reverse denoising steps
PANELS     = 5                   # number of snapshots in the top row (incl. final)
IMAGE_SIZE = 512                 # must match training (dataset.py)
SEED       = 123                 # fixed seed for reproducible noise visualization (optional)

# ============================

from models.unet import UNetSmall
from diffusion.schedule import make_linear_schedule, extract
from diffusion.sampler import make_timesteps, predict_x0_from_eps  # reuse helpers

# --- tiny helpers (keep local to avoid extra modules) ---
def load_paths_yaml(path="configs/paths.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_rgb(path: str, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Open image -> RGB -> resize -> tensor [0,1], shape (3,H,W)."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(torch.ByteTensor(bytearray(img.tobytes())).float().numpy())
    x = x.view(img.size[1], img.size[0], 3).permute(2, 0, 1).contiguous().float() / 255.0
    return x.clamp_(0.0, 1.0)

def to_pil(x: torch.Tensor) -> Image.Image:
    """Tensor [3,H,W] in [0,1] -> PIL Image."""
    x = x.clamp(0.0, 1.0).detach().cpu()
    x = (x * 255.0).round().byte()
    x = x.permute(1, 2, 0).contiguous()  # H,W,3
    return Image.fromarray(x.numpy(), mode="RGB")

def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def make_panel_row(images, tile_w=256, tile_h=256, pad=4, bgcolor=(20, 20, 20)):
    """Horizontally concatenate PIL images with padding."""
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

def make_panel_grid(top_images, bottom_images=None, pad_vertical=8):
    """2-row grid: top shows denoising snapshots; bottom shows LR (+ HR if provided)."""
    top = make_panel_row(top_images)
    if bottom_images:
        bottom = make_panel_row(bottom_images)
        W = max(top.width, bottom.width)
        H = top.height + pad_vertical + bottom.height
        canvas = Image.new("RGB", (W, H), (20, 20, 20))
        canvas.paste(top, (0, 0))
        canvas.paste(bottom, (0, top.height + pad_vertical))
        return canvas
    return top

def noise_to_vis(noise: torch.Tensor) -> torch.Tensor:
    """
    Map N(0,1) noise to [0,1] for display.
    Simple symmetric clamp to [-3,3] then scale.
    """
    n = noise.clamp(-3.0, 3.0)
    return (n + 3.0) / 6.0

# --- deterministic ε-sampling (inline; uses helpers from sampler.py) ---
@torch.no_grad()
def sample_sequence(model, lr: torch.Tensor, sched: dict, steps: int, panels: int, device: torch.device):
    """
    Runs deterministic DDIM (eta=0) with epsilon-prediction.
    Returns:
        final_x0: [3,H,W]
        panel_images: list of [3,H,W] tensors for visualization
    """
    model.eval()
    B, C, H, W = 1, 3, lr.shape[-2], lr.shape[-1]
    T = sched["T"]

    # Start noise (save a visual copy for the first panel)
    torch.manual_seed(SEED)
    x_t = torch.randn(1, 3, H, W, device=device)

    # Build decreasing timesteps
    ts = make_timesteps(T, steps, device=device)  # [S,] ints
    # Decide which step indices to snapshot as x0 panels (evenly spaced)
    snap_ids = torch.linspace(0, len(ts) - 1, steps=min(panels - 1, len(ts)), device=device).round().long().tolist()

    # First panel = pure noise visualization
    panel_imgs = [noise_to_vis(x_t[0]).clamp(0, 1)]

    x0_pred = None
    for i in range(len(ts) - 1):
        t = ts[i].expand(1)          # (1,)
        t_next = ts[i + 1].expand(1) # (1,)

        eps_pred = model(torch.cat([x_t, lr.unsqueeze(0)], dim=1))  # [1,3,H,W]
        x0_pred = predict_x0_from_eps(x_t, eps_pred, t, sched)

        # DDIM deterministic update: x_{t_next} = sqrt(ab_next)*x0 + sqrt(1-ab_next)*eps_pred
        sqrt_ab_next = extract(sched["sqrt_ab"], t_next, x_t.shape)
        sqrt_1mab_next = extract(sched["sqrt_one_minus_ab"], t_next, x_t.shape)
        x_t = sqrt_ab_next * x0_pred + sqrt_1mab_next * eps_pred

        # Save evenly spaced snapshots of x0 (clean reconstructions)
        if i in snap_ids:
            panel_imgs.append(x0_pred[0].clamp(0, 1))

    # Ensure final x0 is the last panel
    if len(panel_imgs) < PANELS:
        panel_imgs.append(x0_pred[0].clamp(0, 1))

    final_x0 = x0_pred[0].clamp(0, 1)
    return final_x0, panel_imgs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load paths + manifest ---
    paths = load_paths_yaml()
    manifest_path = paths["manifest"]
    out_dir = paths.get("out_dir", "outputs/minimal")
    out_final = Path(out_dir) / "final"
    out_panels = Path(out_dir) / "panels"
    out_final.mkdir(parents=True, exist_ok=True)
    out_panels.mkdir(parents=True, exist_ok=True)

    # --- Load model + checkpoint ---
    ckpt_path = f"checkpoints/{EXP_NAME}/ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNetSmall(in_ch=6, base=64, out_ch=3).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    # Rebuild schedule (use values saved in ckpt to avoid mismatch)
    T = ckpt["schedule"]["T"]
    beta_start = ckpt["schedule"]["beta_start"]
    beta_end = ckpt["schedule"]["beta_end"]
    sched = make_linear_schedule(T=T, beta_start=beta_start, beta_end=beta_end, device=device)

    # --- Iterate manifest rows ---
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hr_path = row.get("hr_path")
            lr_path = row.get("lr_path")
            stem = Path(hr_path or lr_path).stem  # pick something stable for naming

            # Load LR (and HR if available for bottom row)
            lr = load_rgb(lr_path, IMAGE_SIZE).to(device)
            hr_im = None
            if hr_path and os.path.exists(hr_path):
                hr_im = Image.open(hr_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)

            # Run deterministic sampling
            final_x0, snaps = sample_sequence(model, lr, sched, STEPS, PANELS, device=device)

            # Build panel: top = [noise | mid... | final]; bottom = [LR | (HR optional)]
            # Convert tensors to PIL
            top_images = [to_pil(t) for t in snaps]
            bottom_images = [to_pil(lr)]
            if hr_im is not None:
                bottom_images.append(hr_im)

            panel_img = make_panel_grid(top_images, bottom_images)
            final_img = to_pil(final_x0)

            # Save
            save_image(final_img, str(out_final / f"{stem}.png"))
            save_image(panel_img, str(out_panels / f"{stem}_panel.png"))

            print(f"Saved: final={out_final / (stem + '.png')}  panel={out_panels / (stem + '_panel.png')}")

if __name__ == "__main__":
    main()
