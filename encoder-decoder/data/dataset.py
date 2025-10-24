# minimal-satSR/data/dataset.py
import csv
import os
from typing import List, Tuple

from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

IMAGE_SIZE = 512  # hard-coded: keep it simple

def _load_rgb(path: str, size: int=IMAGE_SIZE) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.array(img, dtype=np.uint8)              # H,W,3 in 0..255
    x = torch.from_numpy(arr).permute(2,0,1).float() # 3,H,W
    return (x / 255.0).clamp_(0.0, 1.0)

class PairDataset(Dataset):
    """
    Minimal paired dataset: reads a CSV with headers 'hr_path,lr_path'.
    Paths can be absolute or relative; no transforms, no augmentations.
    Returns dict with 'hr', 'lr', and 'path' (hr path for reference).
    """
    def __init__(self, manifest_csv: str):
        self.pairs: List[Tuple[str, str]] = []
        with open(manifest_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hr, lr = row["hr_path"], row["lr_path"]
                self.pairs.append((hr, lr))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        hr_path, lr_path = self.pairs[idx]
        hr = _load_rgb(hr_path)
        lr = _load_rgb(lr_path)
        return {"hr": hr, "lr": lr, "path": hr_path}

def make_loaders(
    manifest_csv: str,
    batch_size: int = 8,
    num_workers: int = 4,
    train_frac: float = 0.9,
    seed: int = 1337,
):
    """
    Create train/val DataLoaders with a single deterministic split.
    """
    ds = PairDataset(manifest_csv)
    n_train = int(len(ds) * train_frac)
    n_val = len(ds) - n_train
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
