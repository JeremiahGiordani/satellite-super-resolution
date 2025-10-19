"""
diffusion.py

Core diffusion utilities for SR:
- Cosine schedule
- ε- or v-prediction
- Training loss with LR conditioning (concat)
- DDIM sampling (deterministic, eta=0)

Config is loaded from SATSR_CONFIG (default: configs/train.yaml).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Literal, Dict, Any

import yaml
import math
import torch
import torch.nn.functional as F
from torch import nn


# ------------------------------
# Config loader (YAML-driven)
# ------------------------------

def load_config() -> Dict[str, Any]:
    cfg_path = os.environ.get("SATSR_CONFIG", "configs/train.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------
# Cosine schedule (Nichol+Dhariwal)
# ------------------------------

def _cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Continuous alpha_bar(t) in [0, 1], cosine schedule."""
    # t in [0,1]
    return torch.cos((t + s) / (1 + s) * math.pi * 0.5).pow(2)


def build_discrete_schedule(T: int, device: torch.device, s: float = 0.008):
    """
    Precompute betas/alphas/alpha_bar for T discrete timesteps.
    Returns dict of tensors on `device`.
    """
    steps = torch.linspace(0, 1, T + 1, device=device)  # T+1 edges
    abar = _cosine_alpha_bar(steps, s=s)
    # Normalize so alpha_bar(0)=1
    abar = abar / abar[0]

    # alpha_bar[t] for t=0..T
    alpha_bar = abar[1:]  # len T
    alpha_bar_prev = abar[:-1]  # len T

    # betas derived from alpha_bar
    betas = 1 - (alpha_bar / alpha_bar_prev)
    betas = betas.clamp(1e-8, 0.999)

    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)

    sigma2 = (1 - alphas)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_cumprod,   # == alpha_bar
        "sigma2": sigma2,
    }


# ------------------------------
# Utilities for objectives
# ------------------------------

@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bar: torch.Tensor
    sigma2: torch.Tensor


def extract_at(idx_tensor: torch.Tensor, t: torch.Tensor, x_shape):
    """
    Gather per-sample coefficients at timestep t (B,) and reshape to [B,1,1,1].
    """
    out = idx_tensor.gather(0, t).float()
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out


def x0_from_xt_eps(x_t, eps, alpha_bar_t):
    """x0 = (x_t - sqrt(1-ab)*eps) / sqrt(ab)"""
    return (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)


def eps_from_xt_x0(x_t, x0, alpha_bar_t):
    """eps = (x_t - sqrt(ab)*x0) / sqrt(1-ab)"""
    return (x_t - torch.sqrt(alpha_bar_t) * x0) / torch.sqrt(1 - alpha_bar_t)


def v_from_x0_eps(x0, eps, alpha_bar_t):
    """v = sqrt(ab)*eps - sqrt(1-ab)*x0 (Imagen)"""
    return torch.sqrt(alpha_bar_t) * eps - torch.sqrt(1 - alpha_bar_t) * x0


def eps_x0_from_v(x_t, v, alpha_bar_t):
    """
    Recover (eps, x0) from v and x_t.
    From: v = sqrt(ab)*eps - sqrt(1-ab)*x0
    and x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps
    Solve:
      eps = sqrt(ab)*x_t + sqrt(1-ab)*v
      x0  = sqrt(ab)*x_t - sqrt(1-ab)*v
    then divide eps by ab? Careful—derive properly:

    Derivation:
      Let a = sqrt(ab), b = sqrt(1-ab)
      x_t = a x0 + b eps
      v   = a eps - b x0
      Solve linear system:

      [ b  a ] [eps] = [ x_t ]
      [-b  a ] [x0 ]   [ v   ]

      We can derive closed form:
      eps = ( a * x_t + b * v )
      x0  = ( a * x_t - b * v )

    where a = sqrt(ab), b = sqrt(1-ab).
    """
    a = torch.sqrt(alpha_bar_t)
    b = torch.sqrt(1 - alpha_bar_t)
    eps = a * v + b * x_t
    x0  = a * x_t - b * v
    # To keep same scale as eps/x0 definitions above (no extra division)
    return eps, x0


# ------------------------------
# Diffusion Engine
# ------------------------------

class DiffusionEngine(nn.Module):
    """
    Wraps schedule + losses + sampling for SR diffusion.
    Model is a U-Net: f(x_t || lr, t_emb) → pred in {eps, v}.
    We encode timestep via a simple sinusoidal embedding added as a channel bias.
    (To keep v1 light, we inject t via FiLM-like scale/shift at the input conv.)
    """

    def __init__(self, model: nn.Module, cfg: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.cfg = cfg
        device = next(model.parameters()).device

        T = int(cfg["diffusion"].get("timesteps", 1000))
        sched = build_discrete_schedule(T, device=device, s=cfg["diffusion"].get("cosine_s", 0.008))
        self.register_buffer("betas", sched["betas"])
        self.register_buffer("alphas", sched["alphas"])
        self.register_buffer("alpha_bar", sched["alpha_bar"])
        self.register_buffer("sigma2", sched["sigma2"])

        self.objective: Literal["eps", "v"] = cfg["diffusion"].get("objective", "v")
        self.lowfreq_l1_weight = float(cfg.get("loss", {}).get("lowfreq_l1_weight", 0.0))

        # Simple sinusoidal timestep embedding (dim = base_channels)
        self.t_embed_dim = int(cfg["model"].get("base_channels", 64))
        self.t_mlp = nn.Sequential(
            nn.Linear(self.t_embed_dim, self.t_embed_dim),
            nn.SiLU(),
            nn.Linear(self.t_embed_dim, self.t_embed_dim),
        )

        # Project t embedding to a (per-channel) bias for input concat
        in_ch = int(cfg["model"].get("in_channels", 6))
        self.t_to_bias = nn.Linear(self.t_embed_dim, in_ch)

    # ---- timestep embedding ----
    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        t: [B] integer timesteps in [0..T-1]
        Returns [B, dim] embedding.
        """
        device = t.device
        half = dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), half, device=device)
        )
        args = t.float().unsqueeze(1) / freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")
        return emb

    # ---- training step utilities ----
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """
        Sample x_t ~ q(x_t | x0)
        x_t = sqrt(ab_t) * x0 + sqrt(1-ab_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        ab_t = extract_at(self.alpha_bar, t, x0.shape)
        mean = torch.sqrt(ab_t) * x0
        std = torch.sqrt(1 - ab_t)
        return mean + std * noise, noise  # return eps used

    def model_pred_to_eps_x0(self, x_t: torch.Tensor, pred: torch.Tensor, t: torch.Tensor):
        ab_t = extract_at(self.alpha_bar, t, x_t.shape)
        if self.objective == "eps":
            eps = pred
            x0 = x0_from_xt_eps(x_t, eps, ab_t)
            return eps, x0
        elif self.objective == "v":
            # Convert v-pred to (eps, x0)
            eps, x0 = eps_x0_from_v(x_t, pred, ab_t)
            return eps, x0
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def forward_loss(self, lr: torch.Tensor, hr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        One training step (loss computation only).
        lr, hr: [B,3,H,W] in [0,1]
        Returns dict with 'loss' and logging items.
        """
        device = hr.device
        B = hr.size(0)
        T = self.alpha_bar.size(0)

        t = torch.randint(0, T, (B,), device=device)  # uniform
        x_t, eps = self.q_sample(hr, t)

        # Build timestep-conditioned input: concat(noisy_hr, lr) and add t-bias
        x_in = torch.cat([x_t, lr], dim=1)  # [B,6,H,W]

        t_emb = self.sinusoidal_embedding(t, self.t_embed_dim)
        t_emb = self.t_mlp(t_emb)                       # [B, C]
        t_bias = self.t_to_bias(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, in_ch,1,1]
        x_in = x_in + t_bias

        pred = self.model(x_in)  # [B,3,H,W] -> pred in {eps, v}

        # Supervised target
        if self.objective == "eps":
            target = eps
        else:  # v
            ab_t = extract_at(self.alpha_bar, t, hr.shape)
            v = v_from_x0_eps(hr, eps, ab_t)
            target = v

        diff_loss = F.mse_loss(pred, target)

        # Optional low-frequency L1 anchor between x0_pred and HR
        lf_loss = torch.tensor(0.0, device=device)
        if self.lowfreq_l1_weight > 0:
            eps_pred, x0_pred = self.model_pred_to_eps_x0(x_t, pred, t)
            # blur with a small Gaussian (depthwise conv as a quick low-pass)
            kernel = _gaussian_kernel(device, k=9, sigma=1.5)
            lf_hr = F.conv2d(hr, kernel, padding=4, groups=3)
            lf_pred = F.conv2d(x0_pred, kernel, padding=4, groups=3)
            lf_loss = F.l1_loss(lf_pred, lf_hr)

        loss = diff_loss + self.lowfreq_l1_weight * lf_loss

        return {
            "loss": loss,
            "diff_loss": diff_loss.detach(),
            "lf_loss": lf_loss.detach(),
            "t_mean": t.float().mean().detach(),
        }

    # ---- DDIM sampling (eta=0) ----
    @torch.no_grad()
    def ddim_sample(self, lr: torch.Tensor, steps: int = 50, cfg_weight: float = 1.0):
        """
        Deterministic DDIM sampling from pure noise, conditioned on LR.
        Returns SR in [0,1].

        cfg_weight is kept for future classifier-free guidance; for v1 keep =1.0.
        """
        device = lr.device
        B, C, H, W = lr.shape
        T = self.alpha_bar.size(0)

        # time indices we will traverse (e.g., 1000 -> 0 in ~50 steps)
        ts = torch.linspace(T - 1, 0, steps, device=device).long()

        x_t = torch.randn(B, 3, H, W, device=device)

        for i in range(steps):
            t = ts[i].repeat(B)

            x_in = torch.cat([x_t, lr], dim=1)
            t_emb = self.sinusoidal_embedding(t, self.t_embed_dim)
            t_emb = self.t_mlp(t_emb)
            t_bias = self.t_to_bias(t_emb).unsqueeze(-1).unsqueeze(-1)
            x_in = x_in + t_bias

            pred = self.model(x_in)  # [B,3,H,W]

            # Convert to eps/x0
            eps_pred, x0_pred = self.model_pred_to_eps_x0(x_t, pred, t)

            # DDIM update
            ab_t = extract_at(self.alpha_bar, t, x_t.shape)                 # ᾱ_t
            a_t = extract_at(self.alphas, t, x_t.shape)                      # α_t
            # Compute ᾱ_{t-1}
            t_next = torch.clamp(t - 1, min=0)
            ab_next = extract_at(self.alpha_bar, t_next, x_t.shape)

            # Deterministic eta=0:
            # x_{t-1} = sqrt(ᾱ_{t-1}) x0_pred + sqrt(1-ᾱ_{t-1}) * eps_pred
            x_prev = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * eps_pred
            x_t = x_prev

        # Final prediction: x0 from x_t at t=0 (ab=1 -> x0_pred ~ x_t)
        x0 = x_t.clamp(0.0, 1.0)
        return x0


# ------------------------------
# Small helper: Gaussian kernel
# ------------------------------

def _gaussian_kernel(device, k=9, sigma=1.5):
    """Depthwise 3x Gaussian conv kernel for low-frequency L1."""
    ax = torch.arange(k, device=device) - (k - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    ker = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    ker = ker / ker.sum()
    ker = ker.view(1, 1, k, k)
    ker = ker.repeat(3, 1, 1, 1)  # depthwise for 3 channels
    return ker
