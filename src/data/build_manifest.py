#!/usr/bin/env python3
"""
build_manifest.py (YAML-driven, no CLI)

Reads a YAML config (default: configs/data.yaml) and builds a CSV manifest
pairing HR and LR images for training.

Expected YAML keys (minimal):
  paths:
    hr_dir: "data/hr_whu"
    lr_dir: "data/lr_whu"
    manifest: "data/manifests/pairs.csv"
  naming:
    lr_suffix: "_lr"     # substring before extension in LR files (e.g., foo_lr_x4.png)

Notes
- No command-line args; override config path with env var SATSR_CONFIG if needed.
- Pairs by stripping `lr_suffix` and everything after it from the LR stem.
  Example: "tile_0123_lr_x4.png" → base "tile_0123" → matches "tile_0123.png".
"""

import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


# ---------- Helpers ----------

def load_config() -> dict:
    cfg_path = os.environ.get("SATSR_CONFIG", "configs/data.yaml")
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        sys.exit(f"[build_manifest] Config file not found: {cfg_file}")
    with cfg_file.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    if not folder.exists():
        sys.exit(f"[build_manifest] Folder does not exist: {folder}")
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def hr_index_by_stem(hr_paths: List[Path]) -> Dict[str, Path]:
    # Map pure stem -> path (e.g., "tile_0123" → ".../tile_0123.png")
    return {p.stem: p for p in hr_paths}


def derive_hr_key_from_lr_stem(lr_stem: str, lr_suffix: str) -> str:
    """
    Given an LR filename stem (e.g., 'tile_0123_lr_x4'), strip the configured
    lr_suffix and everything after it, returning the base key ('tile_0123').
    If suffix is not present, return empty string to indicate "no match".
    """
    if lr_suffix and lr_suffix in lr_stem:
        return lr_stem.split(lr_suffix, 1)[0]
    return ""


# ---------- Main ----------

def main():
    cfg = load_config()

    # Required paths in YAML
    try:
        hr_dir = Path(cfg["paths"]["hr_dir"])
        lr_dir = Path(cfg["paths"]["lr_dir"])
        manifest_path = Path(cfg["paths"]["manifest"])
    except KeyError as e:
        sys.exit(f"[build_manifest] Missing config key: {e}. "
                 f"Ensure 'paths.hr_dir', 'paths.lr_dir', and 'paths.manifest' are set.")

    # Optional naming config
    lr_suffix = cfg.get("naming", {}).get("lr_suffix", "_lr")

    # Prepare output dir
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Scan files
    hr_files = list_images(hr_dir)
    lr_files = list_images(lr_dir)

    print(f"[build_manifest] HR files: {len(hr_files)}  |  LR files: {len(lr_files)}")
    if not hr_files or not lr_files:
        sys.exit("[build_manifest] No files found—check your config paths.")

    hr_map = hr_index_by_stem(hr_files)

    # Build pairs
    pairs: List[Tuple[Path, Path]] = []
    unmatched_lr: List[Path] = []

    for lr_path in lr_files:
        hr_key = derive_hr_key_from_lr_stem(lr_path.stem, lr_suffix)
        if not hr_key:
            unmatched_lr.append(lr_path)
            continue
        hr_path = hr_map.get(hr_key)
        if hr_path is None:
            unmatched_lr.append(lr_path)
        else:
            pairs.append((hr_path, lr_path))

    # Write manifest
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hr_path", "lr_path"])
        for hr, lr in pairs:
            writer.writerow([str(hr), str(lr)])

    # Logging
    print(f"[build_manifest] Pairs written: {len(pairs)} → {manifest_path}")
    if unmatched_lr:
        print(f"[build_manifest] WARNING: {len(unmatched_lr)} LR files had no HR match "
              f"(check 'naming.lr_suffix' and filenames). Example:")
        for p in unmatched_lr[:5]:
            print("   -", p.name)
        if len(unmatched_lr) > 5:
            print(f"   ... {len(unmatched_lr) - 5} more")


if __name__ == "__main__":
    main()
