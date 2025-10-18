"""
dataset.py

YAML-driven PyTorch dataset for paired LR/HR images.
Reads pairs.csv manifest declared in configs/data.yaml.

Expected YAML:
paths:
  manifest: "data/manifests/pairs.csv"
io:
  image_size: 512
  channels: 3
"""

import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def load_config():
    """Load YAML config from default location or SATSR_CONFIG env var."""
    cfg_path = os.environ.get("SATSR_CONFIG", "configs/data.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class PairedImageDataset(Dataset):
    """
    Loads paired LR/HR images using the manifest CSV defined in configs/data.yaml.

    Each __getitem__ returns:
        {
          "lr": Tensor [C,H,W] float32 in [0,1]
          "hr": Tensor [C,H,W] float32 in [0,1]
          "path": hr_path (for tracking)
        }
    """

    def __init__(self, split="train"):
        """
        Args:
            split (str): 'train' or 'val' — filters manifest by split if available.
                         Otherwise it shuffles & splits using train_frac / val_frac from config.
        """
        self.cfg = load_config()
        manifest_path = Path(self.cfg["paths"]["manifest"])
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        # Read manifest CSV
        lines = manifest_path.read_text().strip().splitlines()
        self.pairs = [tuple(line.split(",")) for line in lines[1:]]  # skip header
        # Example: [("data/hr/foo.png", "data/lr/foo_lr_x4.png"), ...]

        # Optional split logic
        train_frac = self.cfg.get("splits", {}).get("train_frac", 0.9)
        idx = int(len(self.pairs) * train_frac)
        if split == "train":
            self.pairs = self.pairs[:idx]
        else:
            self.pairs = self.pairs[idx:]

        self.image_size = self.cfg["io"]["image_size"]
        self.channels = self.cfg["io"]["channels"]

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        """Load image and convert to CHW float tensor in [0,1]."""
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size),
            resample=Image.BICUBIC
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,C
        return torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W

    def __getitem__(self, idx):
        hr_path, lr_path = self.pairs[idx]
        hr = self._load_image(hr_path)
        lr = self._load_image(lr_path)
        return {"lr": lr, "hr": hr, "path": hr_path}
