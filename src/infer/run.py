"""
infer/run.py

YAML-driven super-resolution inference with a denoising panel.
- Saves final SR PNG.
- Also saves a composite grid:
    Row 1 (5 cols):  noise | mid1 | mid2 | mid3 | final SR
    Row 2 (5 cols):  LR    |  (blank) (blank) (blank) | HR (if provided; else blank)

Config (default: configs/infer.yaml; override with SATSR_CONFIG):
  checkpoint: "checkpoints/sr_v1/model_step_1000.pt"
  sampler:
    type: "ddim"
    steps: 50
    cfg_weight: 1.0
  io:
    lr_dir: "data/lr_whu"
    out_dir: "outputs/sr"
    image_size: 512
    hr_dir: null                  # optional; if set, will load HR by stem match
  panel:
    save: true
    mids: 3                       # number of mid frames to show (3 → 5 total with noise+final)
"""

import os
from pathlib import Path
import yaml
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.models.unet import UNetModel
from src.models.diffusion import DiffusionEngine, extract_at


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

def normalize_vis(t: torch.Tensor):
    """Normalize arbitrary tensor image to 0..1 for visualization (e.g., pure noise)."""
    if t.dim() == 4:
        t = t[0]
    # Per-channel min/max helps show pattern while keeping color-ish appearance
    c, h, w = t.shape
    t = t.clone()
    for ch in range(c):
        v = t[ch]
        vmin, vmax = v.min(), v.max()
        if float(vmax - vmin) < 1e-6:
            v[:] = 0.5
        else:
            v[:] = (v - vmin) / (vmax - vmin)
        t[ch] = v
    return t.clamp(0, 1)

def blank_panel(w, h):
    return Image.new("RGB", (w, h), (240, 240, 240))

def label_image(im: Image.Image, text: str):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([(0,0),(im.width, 20)], fill=(0,0,0))
    draw.text((6, 3), text, fill=(255,255,255), font=font)
    return im

def make_two_row_panel(top_imgs, bottom_imgs, cell_size):
    """Assemble a 2x5 grid. top_imgs and bottom_imgs are lists of len<=5 PIL Images."""
    cols = 5
    rows = 2
    w, h = cell_size
    canvas = Image.new("RGB", (cols*w, rows*h), (255,255,255))
    for r in range(rows):
        for c in range(cols):
            idx = c
            src = None
            if r == 0:
                if idx < len(top_imgs):
                    src = top_imgs[idx]
            else:
                if idx < len(bottom_imgs):
                    src = bottom_imgs[idx]
            if src is None:
                src = blank_panel(w, h)
            canvas.paste(src.resize((w, h), Image.BICUBIC), (c*w, r*h))
    return canvas


# -----------------------------
# Inference with panel capture
# -----------------------------

@torch.no_grad()
def sample_with_panels(engine: DiffusionEngine, lr: torch.Tensor, steps: int, cfg_weight: float):
    """
    Run a DDIM-like loop but capture:
      - pure noise at start
      - x0_pred at several mid steps
      - final SR
    Returns final SR tensor in [0,1], plus list of PIL images for the panel.
    """
    device = lr.device
    B, C, H, W = lr.shape
    assert B == 1, "This panel capture path expects batch size 1 for now."

    T = engine.alpha_bar.size(0)
    ts = torch.linspace(T - 1, 0, steps, device=device).long()

    # Initialize
    x_t = torch.randn(B, 3, H, W, device=device)

    # Panels to capture
    panels = []
    # Panel 0: pure noise (normalized for vis)
    panels.append(tensor_to_pil(normalize_vis(x_t)))

    # Choose 3 mid indices (evenly spaced in 1..steps-1)
    mids = [steps // 4, steps // 2, (3 * steps) // 4]
    mid_set = set(mids)

    final_sr = None

    for i in range(steps):
        t = ts[i].repeat(B)

        x_in = torch.cat([x_t, lr], dim=1)
        # Timestep conditioning (internal to engine)
        t_emb = engine.sinusoidal_embedding(t, engine.t_embed_dim)
        t_emb = engine.t_mlp(t_emb)
        t_bias = engine.t_to_bias(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_in = x_in + t_bias

        pred = engine.model(x_in)  # [B,3,H,W]
        # Get eps & x0_pred
        eps_pred, x0_pred = engine.model_pred_to_eps_x0(x_t, pred, t)

        # Capture mids using x0_pred (clean reconstruction estimate)
        if (i + 1) in mid_set:
            panels.append(label_image(tensor_to_pil(x0_pred), f"t={int(ts[i].item())}"))

        # DDIM deterministic update
        ab_t   = extract_at(engine.alpha_bar, t, x_t.shape)
        t_next = torch.clamp(t - 1, min=0)
        ab_next = extract_at(engine.alpha_bar, t_next, x_t.shape)
        x_prev = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * eps_pred
        x_t = x_prev

    # Final SR (x0 approx from last)
    final_sr = x0_pred.clamp(0, 1)
    panels.append(label_image(tensor_to_pil(final_sr), "final"))

    return final_sr, panels


def main():
    cfg = get_cfg()
    ckpt_path = Path(cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    io_cfg = cfg.get("io", {})
    lr_dir = Path(io_cfg.get("lr_dir", "data/lr_whu"))
    out_dir = Path(io_cfg.get("out_dir", "outputs/sr"))
    out_dir.mkdir(parents=True, exist_ok=True)
    image_size = int(io_cfg.get("image_size", 512))
    hr_dir = io_cfg.get("hr_dir", None)
    hr_dir = Path(hr_dir) if hr_dir else None

    panel_cfg = cfg.get("panel", {})
    save_panel = bool(panel_cfg.get("save", True))
    mids_count = int(panel_cfg.get("mids", 3))  # currently fixed at 3 in code above

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Restore model config
    train_cfg = ckpt.get("cfg", {})
    model_cfg = train_cfg.get("model", {
        "in_channels": 6,
        "out_channels": 3,
        "base_channels": 64,
        "channel_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
    })

    # Build model + diffusion engine
    model = UNetModel(
        in_channels=model_cfg.get("in_channels", 6),
        out_channels=model_cfg.get("out_channels", 3),
        base_channels=model_cfg.get("base_channels", 64),
        channel_mult=tuple(model_cfg.get("channel_mult", [1, 2, 4, 4])),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        attn_resolutions=tuple(model_cfg.get("attn_resolutions", [])),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    engine = DiffusionEngine(model, train_cfg if train_cfg else {
        "model": model_cfg,
        "diffusion": {"objective": "v", "timesteps": 1000, "cosine_s": 0.008},
        "loss": {"lowfreq_l1_weight": 0.0},
        "sampler": {"type": "ddim", "steps": 50, "cfg_weight": 1.0},
    }).to(device)

    # Load EMA if present, else raw
    if "ema" in ckpt:
        print("Loading EMA weights")
        engine.model.load_state_dict(ckpt["ema"])
    else:
        print("Loading model weights")
        engine.model.load_state_dict(ckpt["model"])
    engine.eval()

    # Sampler settings
    sampler_cfg = cfg.get("sampler", {})
    steps = int(sampler_cfg.get("steps", 50))
    cfg_weight = float(sampler_cfg.get("cfg_weight", 1.0))

    # List LR images
    lr_paths = list_images(lr_dir)
    if not lr_paths:
        print(f"No images found in {lr_dir}")
        return

    # Optional HR mapping by stem
    hr_map = {}
    if hr_dir and hr_dir.exists():
        for hp in list_images(hr_dir):
            hr_map[hp.stem] = hp

    print(f"Inferencing {len(lr_paths)} images from {lr_dir} → {out_dir}")
    for p in lr_paths:
        # Load LR
        lr = pil_to_tensor(Image.open(p), image_size).to(device)

        # Sample with panel capture
        sr, top_panels = sample_with_panels(engine, lr, steps=steps, cfg_weight=cfg_weight)

        # Save final SR
        sr_pil = Image.fromarray((sr[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        out_img = out_dir / f"{p.stem}_sr.png"
        sr_pil.save(out_img, "PNG")
        print(f"wrote {out_img}")

        # Build 2x5 panel if requested
        if save_panel:
            # Bottom row: LR | (blank) (blank) (blank) | HR
            lr_pil = Image.open(p).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
            lr_pil = label_image(lr_pil, "LR")
            if hr_dir:
                # Try to find HR by stripping _lr... suffix if present
                stem = p.stem
                # simple heuristic: take text before "_lr"
                base = stem.split("_lr")[0] if "_lr" in stem else stem
                hr_path = hr_map.get(base, None)
                hr_pil = Image.open(hr_path).convert("RGB").resize((image_size, image_size), Image.BICUBIC) if hr_path else blank_panel(image_size, image_size)
                if hr_path:
                    hr_pil = label_image(hr_pil, "HR")
            else:
                hr_pil = blank_panel(image_size, image_size)

            # Ensure we have exactly 5 top panels: noise + 3 mids + final
            # (sample_with_panels returns exactly that order)
            top5 = top_panels
            if len(top5) < 5:
                # pad to 5 if steps were too few
                while len(top5) < 5:
                    top5.insert(-1, blank_panel(image_size, image_size))

            # Build bottom row with 5 slots: LR, blank, blank, blank, HR
            bottom5 = [lr_pil, blank_panel(image_size, image_size), blank_panel(image_size, image_size), blank_panel(image_size, image_size), hr_pil]

            grid = make_two_row_panel(top5[:5], bottom5, (image_size, image_size))
            panel_path = out_dir / f"{p.stem}_panel.png"
            grid.save(panel_path, "PNG")
            print(f"wrote {panel_path}")

    print("✅ Inference complete!")


if __name__ == "__main__":
    main()
