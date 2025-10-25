# encoder-decoder/models/nafnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Core NAFNet components
# ----------------------------

class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm over HxW, like NAFNet's LN2d.
    y = (x - mean) / sqrt(var + eps) * weight + bias
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var  = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    """
    Split channels in half, elementwise product: x = x1 * x2
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCA(nn.Module):
    """
    Simplified Channel Attention: global average pool -> 1x1 conv -> scale
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)          # B,C,1,1
        w = self.conv(s).sigmoid()                    # B,C,1,1
        return x * w


class NAFBlock(nn.Module):
    """
    NAF block (simplified, following NAFNet spirit):
      x -> LN -> 1x1 (C->2C) -> DWConv(3x3) -> SimpleGate -> SCA -> 1x1 (C)
         -> scale by beta1 and add to x
         -> LN -> 1x1 (C->2C) -> SimpleGate -> 1x1 (C)
         -> scale by beta2 and add
    """
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.pw1   = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.dw    = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2)
        self.sg    = SimpleGate()
        self.sca   = SCA(channels)
        self.pw2   = nn.Conv2d(channels, channels, kernel_size=1)
        self.drop1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.pw3   = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.sg2   = SimpleGate()
        self.pw4   = nn.Conv2d(channels, channels, kernel_size=1)
        self.drop2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Learnable residual scales (initialized to zero)
        self.beta1 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta2 = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.sg(y)                    # B, C, H, W   (after gate halves channels)
        y = self.sca(y)
        y = self.pw2(y)
        y = self.drop1(y)
        x = x + self.beta1 * y

        y2 = self.norm2(x)
        y2 = self.pw3(y2)
        y2 = self.sg2(y2)
        y2 = self.pw4(y2)
        y2 = self.drop2(y2)
        x = x + self.beta2 * y2
        return x


class Downsample(nn.Module):
    """
    Strided conv downsample (H,W)/2, keep channels the same before a following 1x1 if needed.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Bilinear upsample (x2) + 1x1 projection to desired channels (lightweight & stable).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.proj(x)


# ----------------------------
# NAFNet-U encoder–decoder
# ----------------------------

class NAFNetU(nn.Module):
    """
    NAFNet-U: encoder–decoder with NAF blocks and UNet-style skips.

    Args:
        in_ch:    input channels (3 for RGB LR)
        base:     base channel width (64 is a good default)
        out_ch:   output channels (3 for RGB delta or RGB direct)
        residual: if True, returns x + head(...); else returns head(...)

    Structure:
        Stem -> [Enc1 -> Down1] -> [Enc2 -> Down2] -> [Enc3 -> Down3] -> Mid
             -> [Up3 -> Dec3] -> [Up2 -> Dec2] -> [Up1 -> Dec1] -> Head
    Resolutions:
        1.0x (H,W) -> 0.5x -> 0.25x -> 0.125x -> back to 1.0x
    Channels:
        base -> 2*base -> 4*base -> 4*base  (cap deep width)
    """
    def __init__(self, in_ch: int = 3, base: int = 64, out_ch: int = 3, residual: bool = True):
        super().__init__()
        self.residual = residual

        c1 = base
        c2 = base * 2
        c3 = base * 4
        c4 = base * 4  # cap

        # Stem
        self.stem = nn.Conv2d(in_ch, c1, kernel_size=3, padding=1)

        # Encoder stages (two NAF blocks per stage)
        self.enc1 = nn.Sequential(NAFBlock(c1), NAFBlock(c1))
        self.down1 = Downsample(c1, c2)

        self.enc2 = nn.Sequential(NAFBlock(c2), NAFBlock(c2))
        self.down2 = Downsample(c2, c3)

        self.enc3 = nn.Sequential(NAFBlock(c3), NAFBlock(c3))
        self.down3 = Downsample(c3, c4)

        # Bottleneck
        self.mid = nn.Sequential(NAFBlock(c4), NAFBlock(c4))

        # Decoder stages
        self.up3 = Upsample(c4, c3)
        self.dec3 = nn.Sequential(NAFBlock(c3 + c3), NAFBlock(c3 + c3))
        self.proj3 = nn.Conv2d(c3 + c3, c3, kernel_size=1)

        self.up2 = Upsample(c3, c2)
        self.dec2 = nn.Sequential(NAFBlock(c2 + c2), NAFBlock(c2 + c2))
        self.proj2 = nn.Conv2d(c2 + c2, c2, kernel_size=1)

        self.up1 = Upsample(c2, c1)
        self.dec1 = nn.Sequential(NAFBlock(c1 + c1), NAFBlock(c1 + c1))
        self.proj1 = nn.Conv2d(c1 + c1, c1, kernel_size=1)

        # Head (predict detail / direct RGB)
        self.head = nn.Conv2d(c1, out_ch, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.stem(x)          # B, c1, H, W
        e1 = self.enc1(x0)         # B, c1, H, W
        d1 = self.down1(e1)        # B, c2, H/2, W/2

        e2 = self.enc2(d1)         # B, c2, H/2, W/2
        d2 = self.down2(e2)        # B, c3, H/4, W/4

        e3 = self.enc3(d2)         # B, c3, H/4, W/4
        d3 = self.down3(e3)        # B, c4, H/8, W/8

        # Mid
        m  = self.mid(d3)          # B, c4, H/8, W/8

        # Decoder
        u3 = self.up3(m)           # -> c3, H/4, W/4
        u3 = torch.cat([u3, e3], dim=1)      # c3 + c3
        u3 = self.dec3(u3)
        u3 = self.proj3(u3)        # back to c3

        u2 = self.up2(u3)          # -> c2, H/2, W/2
        u2 = torch.cat([u2, e2], dim=1)      # c2 + c2
        u2 = self.dec2(u2)
        u2 = self.proj2(u2)        # back to c2

        u1 = self.up1(u2)          # -> c1, H, W
        u1 = torch.cat([u1, e1], dim=1)      # c1 + c1
        u1 = self.dec1(u1)
        u1 = self.proj1(u1)        # back to c1

        delta = self.head(u1)      # B, out_ch, H, W
        return x + delta if self.residual else delta
