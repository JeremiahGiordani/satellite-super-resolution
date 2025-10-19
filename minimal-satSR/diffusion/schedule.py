# minimal-satSR/diffusion/schedule.py
import torch

@torch.no_grad()
def make_linear_schedule(
    T: int = 200,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: str = "cpu",
):
    """
    Classic linear-beta DDPM schedule.
    Returns a dict of 1D tensors of length T on the requested device.
    """
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=device)  # (T,)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)  # (T,)

    sqrt_ab = torch.sqrt(alpha_bar)
    sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar)
    inv_sqrt_alpha = torch.sqrt(1.0 / alphas)

    return {
        "T": T,
        "betas": betas,                    # (T,)
        "alphas": alphas,                  # (T,)
        "alpha_bar": alpha_bar,            # (T,)
        "sqrt_ab": sqrt_ab,                # (T,)
        "sqrt_one_minus_ab": sqrt_one_minus_ab,  # (T,)
        "inv_sqrt_alpha": inv_sqrt_alpha,  # (T,)
    }

def extract(coeff: torch.Tensor, t: torch.Tensor, x_shape):
    """
    Gather per-sample coefficients at integer timesteps t (0..T-1),
    reshape to broadcast over x with shape (B,1,1,1).
    """
    b = t.shape[0]
    out = coeff.gather(0, t.clamp_min(0).clamp_max(coeff.shape[0]-1))
    return out.view(b, 1, 1, 1).expand(b, 1, x_shape[-2], x_shape[-1])
