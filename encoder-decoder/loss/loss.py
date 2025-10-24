# encoder-decoder/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== CONSTANTS (no CLI) ======
PIX_EPS        = 1e-3     # epsilon for Charbonnier
PERCEP_WEIGHT  = 0.05     # weight for VGG19 relu3_3 feature loss
EDGE_WEIGHT    = 0.01     # weight for Sobel gradient loss
FREEZE_VGG     = True     # VGG is inference-only
# =================================

# --- Charbonnier (smooth L1) ---
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = PIX_EPS):
        super().__init__()
        self.eps2 = eps * eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps2))

# --- Perceptual loss (VGG19 relu3_3) ---
class VGGPerceptual(nn.Module):
    """
    Computes L1 distance between VGG19 relu3_3 feature maps.
    Expects inputs in [0,1]. Handles ImageNet normalization internally.
    """
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            weights = VGG19_Weights.IMAGENET1K_FEATURES
            vgg = vgg19(weights=weights).features
        except Exception:
            # Fallback for older torchvision: load pretrained=True
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features
        # relu3_3 is features[16] (0-indexed) in standard VGG19
        self.backbone = nn.Sequential(*list(vgg.children())[:17])
        if FREEZE_VGG:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.l1 = nn.L1Loss()

    @torch.no_grad()
    def _normalize(self, x):
        # x in [0,1]
        return (x - self.mean) / self.std

    def forward(self, x, y):
        # Detach ImageNet stats buffers' device & dtype alignment
        x_n = self._normalize(x.clamp(0,1))
        y_n = self._normalize(y.clamp(0,1))
        if FREEZE_VGG:
            with torch.no_grad():
                fx = self.backbone(x_n)
                fy = self.backbone(y_n)
        else:
            fx = self.backbone(x_n)
            fy = self.backbone(y_n)
        return self.l1(fx, fy)

# --- Edge loss via Sobel gradients ---
class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        # 3-channel depthwise Sobel
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1,1,3,3))
        self.register_buffer("ky", ky.view(1,1,3,3))

    def forward(self, x):
        # x: [B,3,H,W], return gradient magnitude per channel
        gx = F.conv2d(x, self.kx.expand(3,1,3,3), padding=1, groups=3)
        gy = F.conv2d(x, self.ky.expand(3,1,3,3), padding=1, groups=3)
        return torch.abs(gx) + torch.abs(gy)

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = SobelGrad()
        self.l1 = nn.L1Loss()
    def forward(self, x, y):
        gx = self.sobel(x)
        gy = self.sobel(y)
        return self.l1(gx, gy)

# --- Combined loss ---
class CombinedLoss(nn.Module):
    """
    Total loss = Charbonnier + PERCEP_WEIGHT * VGGPerceptual + EDGE_WEIGHT * EdgeLoss
    Returns a scalar; component terms are also exposed via .last dict for logging.
    """
    def __init__(self,
                 use_perceptual: bool = True,
                 use_edge: bool = True,
                 pix_eps: float = PIX_EPS,
                 percep_w: float = PERCEP_WEIGHT,
                 edge_w: float = EDGE_WEIGHT):
        super().__init__()
        self.pix = CharbonnierLoss(eps=pix_eps)
        self.percep = VGGPerceptual() if use_perceptual else None
        self.edge = EdgeLoss() if use_edge else None
        self.percep_w = percep_w
        self.edge_w = edge_w
        self.last = {}

    def forward(self, sr, hr):
        # Expect sr, hr in [0,1]; DO NOT clamp before loss to keep gradients
        L_pix = self.pix(sr, hr)
        total = L_pix
        self.last = {"pix": float(L_pix.detach().cpu())}

        if self.percep is not None and self.percep_w > 0:
            L_p = self.percep(sr, hr)
            total = total + self.percep_w * L_p
            self.last["percep"] = float(L_p.detach().cpu())

        if self.edge is not None and self.edge_w > 0:
            L_e = self.edge(sr, hr)
            total = total + self.edge_w * L_e
            self.last["edge"] = float(L_e.detach().cpu())

        self.last["total"] = float(total.detach().cpu())
        return total
