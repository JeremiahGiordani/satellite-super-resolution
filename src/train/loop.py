"""
train/loop.py

YAML-driven training loop for SR diffusion.
Loads:
  - data config from configs/data.yaml
  - model/training config from configs/train.yaml

Run:
    SATSR_CONFIG=configs/train.yaml python src/train/loop.py
"""

import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.data.dataset import PairedImageDataset
from src.models.unet import UNetModel
from src.models.diffusion import DiffusionEngine


# -----------------------------
# Helpers
# -----------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ema_update(target_model, source_model, decay=0.9999):
    with torch.no_grad():
        for p_t, p_s in zip(target_model.parameters(), source_model.parameters()):
            p_t.data.mul_(decay).add_(p_s.data, alpha=(1 - decay))


# -----------------------------
# Main Training
# -----------------------------
def main():
    # -------- Load YAML configs --------
    train_cfg = load_yaml(os.environ.get("SATSR_CONFIG", "configs/train.yaml"))
    data_cfg = load_yaml("configs/data.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Data --------
    dataset = PairedImageDataset(split="train")
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["loader"]["batch_size"],
        shuffle=True,
        num_workers=train_cfg["loader"].get("num_workers", 4),
        pin_memory=True
    )

    # -------- Model + Diffusion --------
    model = UNetModel(
        in_channels=train_cfg["model"]["in_channels"],
        out_channels=train_cfg["model"]["out_channels"],
        base_channels=train_cfg["model"]["base_channels"],
        channel_mult=train_cfg["model"].get("channel_mult", [1, 2, 4, 4]),
        num_res_blocks=train_cfg["model"].get("num_res_blocks", 2),
        attn_resolutions=train_cfg["model"].get("attn_resolutions", [16, 8]),
    ).to(device)

    diffusion = DiffusionEngine(model, train_cfg).to(device)

    # -------- Optimizer + EMA --------
    optim = AdamW(
        model.parameters(),
        lr=float(train_cfg["optim"]["lr"]),
        betas=tuple(train_cfg["optim"].get("betas", [0.9, 0.999])),
        weight_decay=float(train_cfg["optim"].get("weight_decay", 0.0)),
    )

    ema_model = UNetModel(**train_cfg["model"]).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = train_cfg["optim"].get("ema_decay", 0.9999)

    # -------- Training Loop --------
    epochs = train_cfg["train"]["epochs"]
    ckpt_dir = Path(train_cfg["ckpt"]["out_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(epochs):
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            optim.zero_grad()
            loss_dict = diffusion.forward_loss(lr, hr)
            loss_dict["loss"].backward()
            optim.step()

            # EMA
            ema_update(ema_model, model, decay=ema_decay)

            # Logging
            if step % 50 == 0:
                print(
                    f"[step {step}] loss={loss_dict['loss'].item():.4f} "
                    f"diff={loss_dict['diff_loss']:.4f} "
                    f"lf={loss_dict['lf_loss']:.4f} "
                )

            # Save checkpoint
            if step % 1000 == 0 and step > 0:
                ckpt_path = ckpt_dir / f"model_step_{step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema_model.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                        "cfg": train_cfg,
                    },
                    ckpt_path,
                )
                print(f"✅ Saved checkpoint: {ckpt_path}")

            step += 1

    print("✅ Training complete!")


if __name__ == "__main__":
    main()
