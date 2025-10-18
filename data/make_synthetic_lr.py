#!/usr/bin/env python3
"""
make_synthetic_lr.py

Create synthetic LR images from HR images while KEEPING the original pixel size.
Pipeline per image:
  HR (HxW) -> blur -> downsample by s -> add noise -> JPEG-like compression
  -> upsample back to (H x W) with bicubic  ==> save as *_lr.png

Also writes a manifest.csv with degradation parameters and sizes.

Usage:
  pip install -U pillow numpy scipy
  python make_synthetic_lr.py --hr_dir data/hr_whu --lr_dir data/lr_whu --scale 4

Notes:
  - Outputs PNGs (lossless container) for both HR copy (optional flag) and LR.
  - Uses PIL (Pillow) for resizing (BOX for down, BICUBIC for up) to avoid extra deps.
"""

import argparse, csv, math, random
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hr_dir", required=True, help="Folder with HR images (png/jpg/tif, etc.)")
    p.add_argument("--lr_dir", required=True, help="Folder to write LR images")
    p.add_argument("--scale", type=int, default=4, help="Downsample factor (e.g., 2/3/4)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--copy_hr", action="store_true", help="Also save HR as PNG next to LR for convenience")
    return p.parse_args()


def aniso_gauss_kernel(k=21, sigx=1.2, sigy=2.0, theta=0.0):
    ax = np.arange(-k//2 + 1., k//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xr =  np.cos(theta)*xx + np.sin(theta)*yy
    yr = -np.sin(theta)*xx + np.cos(theta)*yy
    ker = np.exp(-0.5*((xr**2)/(sigx**2) + (yr**2)/(sigy**2)))
    ker /= np.sum(ker)
    return ker


def jpeg_roundtrip_u8(u8_rgb, q=80):
    """u8_rgb: (H,W,3) uint8"""
    buf = BytesIO()
    Image.fromarray(u8_rgb, mode="RGB").save(buf, format="JPEG", quality=int(q))
    return np.asarray(Image.open(BytesIO(buf.getvalue())).convert("RGB"))


def to_chw(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)  # H,W,3
    return arr.transpose(2, 0, 1)  # 3,H,W


def to_pil(chw: np.ndarray) -> Image.Image:
    arr = np.clip(chw.transpose(1, 2, 0), 0, 255).astype(np.uint8)  # H,W,3
    return Image.fromarray(arr, mode="RGB")


def main():
    a = parse_args()
    rng = random.Random(a.seed)
    np.random.seed(a.seed)

    hr_dir = Path(a.hr_dir)
    lr_dir = Path(a.lr_dir)
    lr_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = lr_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "basename","hr_path","lr_path","scale",
            "H","W","h_down","w_down",
            "sigx","sigy","theta_deg","gain","sigma_read","jpeg_q"
        ])

        # Accept common image extensions
        hr_files = sorted([p for p in hr_dir.iterdir() if p.suffix.lower() in
                           {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"}])

        if not hr_files:
            print("No HR images found in", hr_dir)
            return

        for hr_path in hr_files:
            # --- Load HR ---
            hr_pil = Image.open(hr_path).convert("RGB")
            W, H = hr_pil.size  # PIL: (width, height)

            # --- Blur (per-channel convolution) ---
            sigx = rng.uniform(0.8, 2.0)
            sigy = rng.uniform(0.8, 2.2)
            theta = rng.uniform(0, math.pi)
            ker = aniso_gauss_kernel(21, sigx, sigy, theta)

            hr_chw = to_chw(hr_pil)  # 3,H,W float32 in [0..255]
            blurred = np.stack([fftconvolve(hr_chw[c], ker, mode="same") for c in range(3)], axis=0)

            # --- Downsample (×s) with proper prefilter already applied (the blur) ---
            s = int(a.scale)
            h_down, w_down = H // s, W // s
            # Use BOX (area) for downsampling to avoid aliasing
            down_pil = to_pil(blurred).resize((w_down, h_down), resample=Image.BOX)

            # --- Noise model (Poisson-Gaussian in DN-ish domain 0..255) ---
            base = np.asarray(down_pil, dtype=np.float32)  # H',W',3
            gain = rng.uniform(4.0, 12.0)       # electrons per DN
            sigma_r = rng.uniform(0.5, 3.0)     # read noise (DN)
            signal = base * gain
            noisy = np.random.poisson(np.clip(signal, 0, None)) / gain
            noisy += np.random.normal(0, sigma_r, size=noisy.shape)

            # --- JPEG-like compression (introduce blocking/ringing artifacts) ---
            u8 = np.clip(noisy, 0, 255).astype(np.uint8)
            jpeg_q = rng.randint(65, 95)
            comp = jpeg_roundtrip_u8(u8, jpeg_q)  # H',W',3 (uint8)

            # --- Upsample back to original size with bicubic (same pixel dims as HR) ---
            lr_up_pil = Image.fromarray(comp, mode="RGB").resize((W, H), resample=Image.BICUBIC)

            # --- Save outputs ---
            base = hr_path.stem
            lr_path = lr_dir / f"{base}_lr_x{s}.png"
            lr_up_pil.save(lr_path, "PNG")

            if a.copy_hr:
                hr_copy_path = lr_dir / f"{base}_hr.png"
                hr_pil.save(hr_copy_path, "PNG")
                hr_ref_for_manifest = hr_copy_path
            else:
                hr_ref_for_manifest = hr_path

            # --- Manifest row ---
            w.writerow([
                base, str(hr_ref_for_manifest), str(lr_path), s,
                H, W, h_down, w_down,
                f"{sigx:.4f}", f"{sigy:.4f}", f"{math.degrees(theta):.2f}",
                f"{gain:.3f}", f"{sigma_r:.3f}", jpeg_q
            ])

            print(f"wrote {lr_path}  (HR size preserved: {H}x{W})")

    print("done. manifest:", manifest_path)


if __name__ == "__main__":
    main()
