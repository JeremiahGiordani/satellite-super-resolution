# encoder-decoder/train.py
import os
import time
import yaml
import torch
import torch.nn as nn

# ====== CONSTANTS (no CLI) ======
EXP_NAME            = "encdec_sr_v1"
BATCH_SIZE          = 4
NUM_WORKERS         = 4
EPOCHS              = 2
LEARNING_RATE       = 2e-4
SEED                = 1337

LOG_EVERY           = 1            # optimizer steps
SAVE_EVERY_STEPS    = 500           # checkpoint cadence (by step)
SAVE_LATEST_COPY    = True          # keep checkpoints/EXP_NAME/ckpt.pt up to date
# =================================

from data.dataset import make_loaders
from models.srunet import SRUNet

def set_seed(seed: int = 1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

def load_paths_yaml(path="configs/paths.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _ckpt_payload(model):
    return {
        "exp_name": EXP_NAME,
        "model": model.state_dict(),
        "arch": {"in_ch": 3, "base": 64, "out_ch": 3, "residual": True, "use_attn": True},
    }

def _save_ckpt(model, out_dir, step=None, refresh_latest=True):
    os.makedirs(out_dir, exist_ok=True)
    if step is None:
        path = os.path.join(out_dir, "ckpt_final.pt")
    else:
        path = os.path.join(out_dir, f"ckpt_step_{step:06d}.pt")
    torch.save(_ckpt_payload(model), path)
    if refresh_latest:
        latest = os.path.join(out_dir, "ckpt.pt")
        torch.save(_ckpt_payload(model), latest)
    print(f"Saved checkpoint → {path}")

@torch.no_grad()
def psnr(x, y, eps=1e-8):
    # x,y in [0,1]
    mse = torch.mean((x - y) ** 2).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = f"checkpoints/{EXP_NAME}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Data ---
    paths = load_paths_yaml()  # expects keys: manifest (and optionally out_dir)
    train_loader, val_loader = make_loaders(
        manifest_csv=paths["manifest"],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_frac=0.9,
        seed=SEED,
    )

    # --- Model / Optim ---
    model = SRUNet(in_ch=3, base=64, out_ch=3, residual=True, use_attn=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    l1 = nn.L1Loss()

    global_step = 0
    start = time.time()
    model.train()

    for epoch in range(1, EPOCHS + 1):
        for batch in train_loader:
            lr = batch["lr"].to(device, non_blocking=True)  # [B,3,512,512], [0,1]
            hr = batch["hr"].to(device, non_blocking=True)

            sr = model(lr)                                 # residual inside: SR = LR + Δ
            sr = sr.clamp(0.0, 1.0)
            loss = l1(sr, hr)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            if (global_step % LOG_EVERY) == 0:
                with torch.no_grad():
                    p = psnr(sr, hr).item()
                elapsed = time.time() - start
                print(f"[epoch {epoch}] step {global_step:06d}  loss={loss.item():.5f}  psnr={p:.2f}dB  ({elapsed:.1f}s)")

            if (global_step % SAVE_EVERY_STEPS) == 0:
                _save_ckpt(model, ckpt_dir, step=global_step, refresh_latest=SAVE_LATEST_COPY)

        # --- quick val at epoch end ---
        with torch.no_grad():
            model.eval()
            val_loss_sum, val_psnr_sum, val_batches = 0.0, 0.0, 0
            for vb in val_loader:
                vlr = vb["lr"].to(device)
                vhr = vb["hr"].to(device)
                vsr = model(vlr).clamp(0.0, 1.0)
                val_loss_sum += l1(vsr, vhr).item()
                val_psnr_sum += psnr(vsr, vhr).item()
                val_batches += 1
                if val_batches >= 10:  # cap time; this is just a quick pulse
                    break
            if val_batches > 0:
                print(f"  ↳ val_loss={val_loss_sum/val_batches:.5f}  val_psnr={val_psnr_sum/val_batches:.2f}dB")
            model.train()

    # Final save
    _save_ckpt(model, ckpt_dir, step=None, refresh_latest=SAVE_LATEST_COPY)
    print(f"Done. Checkpoints in: {ckpt_dir}")

if __name__ == "__main__":
    main()
