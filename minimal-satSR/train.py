# minimal-satSR/train.py
import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Experiment constants (no CLI) ---
EXP_NAME             = "minimal_satSR_v1"
BATCH_SIZE           = 8
NUM_WORKERS          = 4
EPOCHS               = 2
LEARNING_RATE        = 2e-4
SEED                 = 1337

# Diffusion schedule (fixed)
T                    = 200          # total training timesteps
BETA_START           = 1e-4
BETA_END             = 2e-2

# Logging / Checkpoints
LOG_EVERY            = 50           # steps
SAVE_EVERY_STEPS     = 500          # <-- save numbered checkpoint every N steps
SAVE_LATEST_SYMLINK  = True         # also write/refresh checkpoints/EXP/ckpt.pt

# -------------------------

from data.dataset import make_loaders
from models.unet import UNetSmall
from diffusion.schedule import make_linear_schedule, extract
from diffusion.sampler import q_sample_eps

def set_seed(seed: int = 1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

def load_paths_yaml(path="configs/paths.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def _ckpt_payload(model):
    return {
        "exp_name": EXP_NAME,
        "model": model.state_dict(),
        "schedule": {"T": T, "beta_start": BETA_START, "beta_end": BETA_END},
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
        # write/refresh a "latest" copy (avoid symlink portability issues; copy instead)
        torch.save(_ckpt_payload(model), latest)
    print(f"Saved checkpoint → {path}")

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = f"checkpoints/{EXP_NAME}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Data ---
    paths = load_paths_yaml()
    train_loader, val_loader = make_loaders(
        manifest_csv=paths["manifest"],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_frac=0.9,
        seed=SEED,
    )

    # --- Model ---
    model = UNetSmall(in_ch=7, base=64, out_ch=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse = nn.MSELoss()

    # --- Schedule (on device when used) ---
    sched = make_linear_schedule(T=T, beta_start=BETA_START, beta_end=BETA_END, device=device)

    global_step = 0
    model.train()
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        for batch in train_loader:
            hr = batch["hr"].to(device, non_blocking=True)  # [B,3,H,W]
            lr = batch["lr"].to(device, non_blocking=True)  # [B,3,H,W]

            B = hr.shape[0]
            t = torch.randint(low=0, high=T, size=(B,), device=device)
            x_t, eps = q_sample_eps(hr, t, sched)

            # NEW: per-sample scalar s_t = sqrt(1 - alpha_bar[t])
            s_t = extract(sched["sqrt_one_minus_ab"], t, x_t.shape)[:, :, :1, :1]  # (B,1,1,1)
            s_map = s_t.expand(-1, 1, x_t.shape[-2], x_t.shape[-1])                # (B,1,H,W)

            inp = torch.cat([x_t, lr, s_map], dim=1)  # (B,7,H,W)
            eps_hat = model(inp)

            loss = mse(eps_hat, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1

            if (global_step % LOG_EVERY) == 0:
                elapsed = time.time() - start_time
                print(f"[epoch {epoch}] step {global_step:06d}  loss={loss.item():.5f}  ({elapsed:.1f}s)")

            if (global_step % SAVE_EVERY_STEPS) == 0:
                _save_ckpt(model, ckpt_dir, step=global_step, refresh_latest=SAVE_LATEST_SYMLINK)

        # (Optional) tiny val sanity check
        with torch.no_grad():
            model.eval()
            try:
                vb = next(iter(val_loader))
                hr_v = vb["hr"].to(device)
                lr_v = vb["lr"].to(device)
                t_v = torch.randint(0, T, (hr_v.shape[0],), device=device)
                x_t_v, eps_v = q_sample_eps(hr_v, t_v, sched)
                eps_hat_v = model(torch.cat([x_t_v, lr_v], dim=1))
                val_loss = mse(eps_hat_v, eps_v).item()
                print(f"  ↳ val_loss={val_loss:.5f}")
            except StopIteration:
                pass
            model.train()

    # --- Save final + refresh latest ---
    _save_ckpt(model, ckpt_dir, step=None, refresh_latest=SAVE_LATEST_SYMLINK)

if __name__ == "__main__":
    main()
