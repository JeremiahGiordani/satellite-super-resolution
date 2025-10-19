# minimal-satSR/diffusion/sampler.py
import torch
import torch.nn.functional as F
from .schedule import extract

@torch.no_grad()
def q_sample_eps(x0: torch.Tensor, t: torch.Tensor, sched: dict, noise: torch.Tensor = None):
    """
    Forward noising: x_t = sqrt(ab[t]) * x0 + sqrt(1-ab[t]) * eps
    t is int tensor (B,) with values in [0..T-1].
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab_t = extract(sched["sqrt_ab"], t, x0.shape)
    sqrt_1mab_t = extract(sched["sqrt_one_minus_ab"], t, x0.shape)
    return sqrt_ab_t * x0 + sqrt_1mab_t * noise, noise

@torch.no_grad()
def predict_x0_from_eps(x_t: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor, sched: dict):
    """
    Reconstruct x0: x0 = (x_t - sqrt(1-ab[t]) * eps_pred) / sqrt(ab[t])
    """
    sqrt_ab_t = extract(sched["sqrt_ab"], t, x_t.shape)
    sqrt_1mab_t = extract(sched["sqrt_one_minus_ab"], t, x_t.shape)
    x0 = (x_t - sqrt_1mab_t * eps_pred) / (sqrt_ab_t + 1e-8)
    return x0

@torch.no_grad()
def ddim_step_eps(x_t: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, sched: dict):
    """
    Deterministic DDIM update (eta=0) in epsilon parameterization:
      x_{t_next} = sqrt(ab[t_next]) * x0 + sqrt(1 - ab[t_next]) * eps_pred
    """
    x0 = predict_x0_from_eps(x_t, eps_pred, t, sched)
    sqrt_ab_next = extract(sched["sqrt_ab"], t_next, x_t.shape)
    sqrt_1mab_next = extract(sched["sqrt_one_minus_ab"], t_next, x_t.shape)
    x_next = sqrt_ab_next * x0 + sqrt_1mab_next * eps_pred
    return x_next, x0

@torch.no_grad()
def make_timesteps(T: int, steps: int, device: torch.device):
    """
    Build a decreasing integer sequence from T-1 down to 0 with 'steps' points.
    Ensures unique, strictly non-increasing ints.
    """
    ts = torch.linspace(T - 1, 0, steps, device=device)
    ts = ts.long()
    ts = torch.unique_consecutive(ts)  # in case rounding duplicates
    if ts[-1] != 0:
        ts = torch.cat([ts, torch.zeros(1, dtype=torch.long, device=device)], dim=0)
    return ts  # (S,)

@torch.no_grad()
def sample_eps(
    model,
    lr: torch.Tensor,           # [B,3,H,W] in [0,1]
    sched: dict,
    steps: int = 25,
    panels: int = 5,            # number of x0 snapshots to return (including final)
):
    """
    Deterministic ε-prediction sampling with LR conditioning.
    Returns: final_x0 [B,3,H,W], list_of_x0_snapshots (length <= panels)
    """
    device = lr.device
    B, C, H, W = lr.shape
    T = sched["T"]

    # Start from pure noise
    x_t = torch.randn(B, 3, H, W, device=device)

    # Build decreasing timestep schedule
    ts = make_timesteps(T, steps, device=device)  # e.g., [199, 191, ..., 0]
    snaps = []
    snap_ids = torch.linspace(0, len(ts) - 1, steps=min(panels, len(ts)), device=device).round().long().tolist()

    for i in range(len(ts) - 1):
        t = ts[i].expand(B)            # (B,)
        t_next = ts[i + 1].expand(B)   # (B,)

        # Predict eps from concat([x_t, lr])
        inp = torch.cat([x_t, lr], dim=1)  # [B,6,H,W]
        eps_pred = model(inp)

        # DDIM step and x0 reconstruction
        x_t, x0_pred = ddim_step_eps(x_t, eps_pred, t, t_next, sched)

        # Save evenly spaced x0 snapshots
        if i in snap_ids:
            snaps.append(x0_pred.clamp(0.0, 1.0).detach())

    # Ensure final snapshot is included
    if (len(ts) - 2) not in snap_ids:
        snaps.append(x0_pred.clamp(0.0, 1.0).detach())

    final_x0 = x0_pred.clamp(0.0, 1.0).detach()
    return final_x0, snaps
