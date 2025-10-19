# minimal-satSR/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Small, stable conv block: Conv -> GroupNorm -> SiLU
def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )

class UNetSmall(nn.Module):
    """
    Minimal U-Net for epsilon prediction.
    Input:  [B, 6, H, W]  (concat of x_t and LR, both in [0,1])
    Output: [B, 3, H, W]  (epsilon prediction; unbounded)
    """
    def __init__(self, in_ch: int = 7, base: int = 64, out_ch: int = 3):
        super().__init__()
        # Encoder
        self.enc1 = conv_block(in_ch, base)        # 6 -> 64
        self.down1 = nn.Conv2d(base, base, 3, stride=2, padding=1)  # 64 -> 64 @ /2

        self.enc2 = conv_block(base, base * 2)     # 64 -> 128
        self.down2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)  # /4

        self.enc3 = conv_block(base * 2, base * 4) # 128 -> 256  (bottleneck)

        # Decoder
        self.up2 = nn.Conv2d(base * 4, base * 2, 1)   # channel reduce before concat
        self.dec2 = conv_block(base * 2 + base * 2, base * 2)  # skip with enc2

        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = conv_block(base + base, base)      # skip with enc1

        # Head (linear; epsilon can be any real value)
        self.head = nn.Conv2d(base, out_ch, kernel_size=3, padding=1)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)  # ← remove nonlinearity="silu"
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 6, H, W]
        # Encoder
        e1 = self.enc1(x)                 # [B, 64, H,   W]
        d1 = self.down1(e1)               # [B, 64, H/2, W/2]

        e2 = self.enc2(d1)                # [B, 128, H/2, W/2]
        d2 = self.down2(e2)               # [B, 128, H/4, W/4]

        e3 = self.enc3(d2)                # [B, 256, H/4, W/4]

        # Decoder
        u2 = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)  # H/2, W/2
        u2 = self.up2(u2)                 # [B, 128, H/2, W/2]
        c2 = torch.cat([u2, e2], dim=1)   # [B, 256, H/2, W/2]
        d2 = self.dec2(c2)                # [B, 128, H/2, W/2]

        u1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)  # H, W
        u1 = self.up1(u1)                 # [B, 64, H, W]
        c1 = torch.cat([u1, e1], dim=1)   # [B, 128, H, W]
        d1 = self.dec1(c1)                # [B, 64, H, W]

        out = self.head(d1)               # [B, 3, H, W]  (epsilon̂)
        return out
