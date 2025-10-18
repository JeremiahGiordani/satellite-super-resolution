#!/usr/bin/env python3
"""
hf_fetch_whu_minraw.py

Download N images from:
  deprem-ml/deprem_satellite_semantic_whu_dataset
and save them with **no processing/decoding** — just raw bytes to files.

- Auto-detects the Image column
- Forces decode=False so we get {bytes, path, format} instead of PIL
- Writes bytes directly; if bytes missing, copies from path
- Preserves original format via `format` or path suffix (fallback to .img)

Usage:
  pip install -U datasets
  # if needed: huggingface-cli login
  python hf_fetch_whu_minraw.py --outdir data/hr_whu --count 20 --split train
"""

import argparse
from pathlib import Path
import os
import shutil
from datasets import load_dataset, Image as HFImage

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--count", type=int, default=20, help="How many samples to save (takes first N)")
    p.add_argument("--split", default="train", help="Dataset split ('train', 'validation', 'test', or 'default')")
    return p.parse_args()

def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("deprem-ml/deprem_satellite_semantic_whu_dataset", split=args.split)

    # Find the first Image column
    img_col = None
    print(ds.features)
    for name, feat in ds.features.items():
        if isinstance(feat, HFImage):
            img_col = name
            break
    if img_col is None:
        raise RuntimeError("No Image column found in dataset features.")

    # Force decode=False so we get a dict with raw bytes/path/format, not PIL
    ds = ds.cast_column(img_col, HFImage(decode=False))

    n = min(args.count, len(ds))
    for i in range(n):
        row = ds[i]
        val = row[img_col]  # dict-like: {'bytes':..., 'path':..., 'format':...}
        b = val.get("bytes", None)
        pth = val.get("path", None)
        fmt = val.get("format", None)

        print(pth)

        # decide extension
        if fmt:
            ext = f".{fmt.lower()}"
        elif pth:
            ext = Path(pth).suffix or ".img"
        else:
            ext = ".img"

        out_path = out / f"whu_{i:04d}{ext}"

        if b is not None:
            with open(out_path, "wb") as f:
                f.write(b)
        elif pth and os.path.exists(pth):
            shutil.copy2(pth, out_path)
        else:
            # nothing to write for this item
            print(f"skip {i}: no bytes/path")
            continue

        print("wrote", out_path)

    print(f"done. saved {n} images to {out}")

if __name__ == "__main__":
    main()
