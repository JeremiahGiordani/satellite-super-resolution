# minimal-satSR/train.py
import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Experiment constants (no CLI) ---
EXP_NAME       = "minimal_satSR_v1"
BATCH_SIZE     = 2
NUM_WORKERS    = 4
EPOCHS         = 2
LEARNING_RATE  = 2e-4
SEED           = 1337

# Diffusion schedule (fixed)
T              = 200          # total training timesteps
BETA_START     = 1e-4
BETA_END       = 2e-2

# Logging cadence
LOG_EVERY      = 50           # steps

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
    # required keys: hr_dir, lr_dir, manifest, out_dir (out_dir is unused here)
    return cfg

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"checkpoints/{EXP_NAME}", exist_ok=True)

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
    model = UNetSmall(in_ch=6, base=64, out_ch=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse = nn.MSELoss()

    # --- Schedule (on device when used) ---
    sched = make_linear_schedule(T=T, beta_start=BETA_START, beta_end=BETA_END, device=device)

    global_step = 0
    model.train()
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        for batch in train_loader:
            hr = batch["hr"].to(device, non_blocking=True)  # [B,3,H,W] in [0,1]
            lr = batch["lr"].to(device, non_blocking=True)  # [B,3,H,W] in [0,1]

            B = hr.shape[0]
            # Sample per-sample timestep t ∈ [0..T-1]
            t = torch.randint(low=0, high=T, size=(B,), device=device)

            # Forward noising: x_t and ground-truth eps
            x_t, eps = q_sample_eps(hr, t, sched)  # both [B,3,H,W]

            # Predict eps from concat([x_t, lr])
            inp = torch.cat([x_t, lr], dim=1)      # [B,6,H,W]
            eps_hat = model(inp)

            loss = mse(eps_hat, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            if (global_step % LOG_EVERY) == 0:
                elapsed = time.time() - start_time
                print(f"[epoch {epoch}] step {global_step:06d}  loss={loss.item():.5f}  ({elapsed:.1f}s)")

        # (Optional) tiny val pass to make sure nothing exploded — no metrics stored
        with torch.no_grad():
            model.eval()
            try:
                vb = next(iter(val_loader))
                hr_v = vb["hr"].to(device)
                lr_v = vb["lr"].to(device)
                Bv = hr_v.shape[0]
                t_v = torch.randint(0, T, (Bv,), device=device)
                x_t_v, eps_v = q_sample_eps(hr_v, t_v, sched)
                inp_v = torch.cat([x_t_v, lr_v], dim=1)
                eps_hat_v = model(inp_v)
                val_loss = mse(eps_hat_v, eps_v).item()
                print(f"  ↳ val_loss={val_loss:.5f}")
            except StopIteration:
                pass
            model.train()

    # --- Save single checkpoint ---
    ckpt_path = f"checkpoints/{EXP_NAME}/ckpt.pt"
    torch.save(
        {
            "exp_name": EXP_NAME,
            "model": model.state_dict(),
            "schedule": {
                "T": T,
                "beta_start": BETA_START,
                "beta_end": BETA_END,
            },
        },
        ckpt_path,
    )
    print(f"Saved checkpoint → {ckpt_path}")

if __name__ == "__main__":
    main()
