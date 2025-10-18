"""
infer/run.py

YAML-driven super-resolution inference for the diffusion model.

Config (default: configs/infer.yaml; override with SATSR_CONFIG env var):
  checkpoint: "checkpoints/sr_diffusion_v1/last.pt"  # or model_step_XXXX.pt
  sampler:
    type: "ddim"
    steps: 50
    cfg_weight: 1.0
  io:
    lr_dir: "data/lr_whu"
    out_dir: "outputs/sr"
    image_size: 512

Behavior:
- Loads UNet architecture from the checkpoint's saved training config (to avoid mismatches)
- Uses EMA weights if present, falling back to raw model weights
- Processes all images in io.lr_dir (png/jpg/tif/webp), saves *_sr.png to io.out_dir
"""

import os
from pathlib import Path
import yaml
import torch
import numpy as np
from PIL import Image

from src.models.unet import UNetModel
from src.models.diffusion import DiffusionEngine


# -----------------------------
# Helpers
# -----------------------------

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_cfg():
    cfg_path = os.environ.get("SATSR_CONFIG", "configs/infer.yaml")
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return load_yaml(cfg_path)

def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()])

def pil_to_tensor(img: Image.Image, size: int):
    img = img.convert("RGB").resize((size, size), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,3
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return t

def tensor_to_pil(t: torch.Tensor):
    # t in [0,1], shape 1x3xHxW or 3xHxW
    if t.dim() == 4:
        t = t[0]
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# -----------------------------
# Inference
# -----------------------------

def main():
    cfg = get_cfg()
    ckpt_path = Path(cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    io_cfg = cfg["io"]
    lr_dir = Path(io_cfg["lr_dir"])
    out_dir = Path(io_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    image_size = int(io_cfg.get("image_size", 512))

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Restore model config from checkpoint to avoid mismatches
    train_cfg = ckpt.get("cfg", {})
    model_cfg = train_cfg.get("model", {
        "in_channels": 6,
        "out_channels": 3,
        "base_channels": 64,
        "channel_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16, 8],
        "dropout": 0.0,
    })

    # Build model + diffusion engine with the *training* config
    model = UNetModel(
        in_channels=model_cfg.get("in_channels", 6),
        out_channels=model_cfg.get("out_channels", 3),
        base_channels=model_cfg.get("base_channels", 64),
        channel_mult=tuple(model_cfg.get("channel_mult", [1, 2, 4, 4])),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        attn_resolutions=tuple(model_cfg.get("attn_resolutions", [16, 8])),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    engine = DiffusionEngine(model, train_cfg if train_cfg else {
        "model": model_cfg,
        "diffusion": {"objective": "v", "timesteps": 1000, "cosine_s": 0.008},
        "loss": {"lowfreq_l1_weight": 0.0},
        "sampler": {"type": "ddim", "steps": 50, "cfg_weight": 1.0},
    }).to(device)

    # Load EMA weights if available, else raw model weights
    if "ema" in ckpt:
        print("Loading EMA weights")
        engine.model.load_state_dict(ckpt["ema"])
    else:
        print("Loading model weights")
        engine.model.load_state_dict(ckpt["model"])

    engine.eval()

    # Sampler settings (can override steps from infer.yaml)
    sampler_cfg = cfg.get("sampler", {})
    steps = int(sampler_cfg.get("steps", 50))
    cfg_weight = float(sampler_cfg.get("cfg_weight", 1.0))

    # Process all LR images
    lr_paths = list_images(lr_dir)
    if not lr_paths:
        print(f"No images found in {lr_dir}")
        return

    print(f"Inferencing {len(lr_paths)} images from {lr_dir} → {out_dir}")
    for p in lr_paths:
        # Load LR, to device
        lr = pil_to_tensor(Image.open(p), image_size).to(device)

        # Sample SR
        with torch.no_grad():
            sr = engine.ddim_sample(lr, steps=steps, cfg_weight=cfg_weight)  # 1x3xHxW

        # Save
        sr_pil = tensor_to_pil(sr)
        out_path = out_dir / f"{p.stem}_sr.png"
        sr_pil.save(out_path, "PNG")
        print(f"wrote {out_path}")

    print("✅ Inference complete!")


if __name__ == "__main__":
    main()
