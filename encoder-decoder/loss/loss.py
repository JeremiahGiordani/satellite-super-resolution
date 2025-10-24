# encoder-decoder/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== LIGHTWEIGHT DEFAULTS ======
USE_PERCEPTUAL     = False   # default OFF to avoid OOM
PERCEP_TARGET_SIDE = 224     # perceptual computed on downsampled 224x224
PERCEP_WEIGHT      = 0.05
EDGE_WEIGHT        = 0.01
PIX_EPS            = 1e-3    # Charbonnier epsilon
FREEZE_VGG         = True
USE_AMP_PERCEP     = True    # autocast for VGG forward
# ===================================

# ---------- Charbonnier ----------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = PIX_EPS):
        super().__init__()
        self.eps2 = eps * eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps2))

# ---------- Edge (Sobel) ----------
class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1,1,3,3))
        self.register_buffer("ky", ky.view(1,1,3,3))
    def forward(self, x):
        gx = F.conv2d(x, self.kx.expand(3,1,3,3), padding=1, groups=3)
        gy = F.conv2d(x, self.ky.expand(3,1,3,3), padding=1, groups=3)
        return torch.abs(gx) + torch.abs(gy)

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = SobelGrad()
        self.l1 = nn.L1Loss()
    def forward(self, x, y):
        return self.l1(self.sobel(x), self.sobel(y))

# ---------- Perceptual (downsampled & AMP) ----------
class VGGPerceptualSmall(nn.Module):
    """
    VGG19 relu3_3 on downsampled inputs to save memory.
    """
    def __init__(self, target_side: int = PERCEP_TARGET_SIDE):
        super().__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            weights = VGG19_Weights.IMAGENET1K_FEATURES
            vgg = vgg19(weights=weights).features
        except Exception:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features
        self.backbone = nn.Sequential(*list(vgg.children())[:17])  # relu3_3
        if FREEZE_VGG:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
        self.target_side = target_side
        self.l1 = nn.L1Loss()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample to target_side with antialias to cut memory
        B,C,H,W = x.shape
        if max(H,W) != self.target_side:
            x = F.interpolate(x, size=(self.target_side, self.target_side),
                              mode="bilinear", align_corners=False, antialias=True)
        x = x.clamp(0,1)
        x = (x - self.mean) / self.std
        return x

    def forward(self, x, y):
        x = self._prep(x)
        y = self._prep(y)
        # AMP just for VGG forward to halve activations
        if USE_AMP_PERCEP and x.is_cuda:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                fx = self.backbone(x)
                fy = self.backbone(y)
        else:
            fx = self.backbone(x)
            fy = self.backbone(y)
        return self.l1(fx, fy)

# ---------- Combined ----------
class CombinedLoss(nn.Module):
    """
    total = Charbonnier + EDGE_WEIGHT*Edge + PERCEP_WEIGHT*PerceptualSmall (optional)
    """
    def __init__(self,
                 use_perceptual: bool = USE_PERCEPTUAL,
                 percep_w: float = PERCEP_WEIGHT,
                 edge_w: float = EDGE_WEIGHT):
        super().__init__()
        self.pix = CharbonnierLoss()
        self.edge = EdgeLoss() if edge_w > 0 else None
        self.percep = VGGPerceptualSmall() if use_perceptual and percep_w > 0 else None
        self.percep_w = percep_w
        self.edge_w = edge_w
        self.last = {}

    def forward(self, sr, hr):
        # compute on unclamped tensors
        L_pix = self.pix(sr, hr)
        total = L_pix
        self.last = {"pix": float(L_pix.detach().cpu())}

        if self.edge is not None:
            L_e = self.edge(sr, hr)
            total = total + self.edge_w * L_e
            self.last["edge"] = float(L_e.detach().cpu())

        if self.percep is not None:
            # run VGG in eval; it’s frozen
            self.percep.eval()
            with torch.set_grad_enabled(False):
                L_p = self.percep(sr, hr)
            total = total + self.percep_w * L_p
            self.last["percep"] = float(L_p.detach().cpu())

        self.last["total"] = float(total.detach().cpu())
        return total
