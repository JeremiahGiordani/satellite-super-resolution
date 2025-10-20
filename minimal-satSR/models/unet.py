# minimal-satSR/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Lightweight building blocks
# ----------------------------

def norm(ch: int) -> nn.GroupNorm:
    # GroupNorm is stable for small batch sizes
    return nn.GroupNorm(num_groups=min(16, ch), num_channels=ch)

class SepConv(nn.Module):
    """
    Depthwise-separable conv: DW 3x3 + PW 1x1
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pw(self.dw(x))

class ResBlockLite(nn.Module):
    """
    Residual block using SepConv to reduce params/activations.
    GN -> SiLU -> SepConv -> GN -> SiLU -> SepConv
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1 = norm(in_ch)
        self.act1  = nn.SiLU(inplace=True)
        self.conv1 = SepConv(in_ch, out_ch)
        self.norm2 = norm(out_ch)
        self.act2  = nn.SiLU(inplace=True)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = SepConv(out_ch, out_ch)
        self.skip  = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.drop(self.act2(self.norm2(h))))
        return h + self.skip(x)

class SelfAttention2d(nn.Module):
    """
    Multi-head self-attention at 2D bottleneck resolution.
    Only used at the smallest spatial size to keep memory low.
    """
    def __init__(self, ch: int, num_heads: int = 2):
        super().__init__()
        assert ch % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = ch // num_heads
        self.norm = norm(ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # (B, H, D, HW)
        def reshape(t):
            return t.view(b, self.num_heads, self.head_dim, h * w)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.softmax(torch.einsum("bhdk,bhdn->bhkn", q * scale, k), dim=-1)  # (B,H,HW,HW)
        out  = torch.einsum("bhkn,bhdn->bhdk", attn, v)                               # (B,H,D,HW)

        out = out.contiguous().view(b, c, h, w)
        out = self.proj(out)
        return out + x_in

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

# ----------------------------
# Scaled-down high-perf U-Net
# ----------------------------

class UNetSmall(nn.Module):
    """
    Scaled-down, stronger-than-minimal U-Net.

    Signature unchanged:
      UNetSmall(in_ch: int = 6, base: int = 64, out_ch: int = 3)

    - 3 downs (1.0x -> 0.5x -> 0.25x -> 0.125x)
    - Channel plan: base -> 2*base -> 3*base -> 3*base (capped)
    - One ResBlock per encoder/decoder stage; two ResBlocks + 1 attention at bottleneck
    - Depthwise-separable convs to cut memory/compute
    - Attention only at the bottleneck with 2 heads
    """
    def __init__(self, in_ch: int = 6, base: int = 64, out_ch: int = 3):
        super().__init__()
        ch1 = base
        ch2 = base * 2
        ch3 = base * 3
        ch4 = base * 3  # cap deep width

        # --- Encoder ---
        self.enc1 = ResBlockLite(in_ch, ch1, dropout=0.0)  # H, W
        self.down1 = Downsample(ch1)

        self.enc2 = ResBlockLite(ch1, ch2, dropout=0.0)    # H/2, W/2
        self.down2 = Downsample(ch2)

        self.enc3 = ResBlockLite(ch2, ch3, dropout=0.0)    # H/4, W/4
        self.down3 = Downsample(ch3)

        # --- Bottleneck (H/8, W/8) ---
        self.mid1 = ResBlockLite(ch3, ch4, dropout=0.0)
        self.mid_attn = SelfAttention2d(ch4, num_heads=2)
        self.mid2 = ResBlockLite(ch4, ch4, dropout=0.0)

        # --- Decoder ---
        self.up3  = Upsample(ch4)
        self.dec3 = ResBlockLite(ch4 + ch3, ch3, dropout=0.0)

        self.up2  = Upsample(ch3)
        self.dec2 = ResBlockLite(ch3 + ch2, ch2, dropout=0.0)

        self.up1  = Upsample(ch2)
        self.dec1 = ResBlockLite(ch2 + ch1, ch1, dropout=0.0)

        self.out = nn.Sequential(
            norm(ch1), nn.SiLU(inplace=True),
            nn.Conv2d(ch1, out_ch, kernel_size=3, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)            # (B, ch1, H,   W)
        d1 = self.down1(e1)          # (B, ch1, H/2, W/2)

        e2 = self.enc2(d1)           # (B, ch2, H/2, W/2)
        d2 = self.down2(e2)          # (B, ch2, H/4, W/4)

        e3 = self.enc3(d2)           # (B, ch3, H/4, W/4)
        d3 = self.down3(e3)          # (B, ch3, H/8, W/8)

        # Bottleneck
        m  = self.mid1(d3)
        m  = self.mid_attn(m)
        m  = self.mid2(m)

        # Decoder
        u3 = self.up3(m)                         # -> (B, ch4, H/4, W/4)
        u3 = torch.cat([u3, e3], dim=1)          # (B, ch4+ch3, H/4, W/4)
        u3 = self.dec3(u3)                       # (B, ch3, H/4, W/4)

        u2 = self.up2(u3)                        # -> (B, ch3, H/2, W/2)
        u2 = torch.cat([u2, e2], dim=1)          # (B, ch3+ch2, H/2, W/2)
        u2 = self.dec2(u2)                       # (B, ch2, H/2, W/2)

        u1 = self.up1(u2)                        # -> (B, ch2, H, W)
        u1 = torch.cat([u1, e1], dim=1)          # (B, ch2+ch1, H, W)
        u1 = self.dec1(u1)                       # (B, ch1, H, W)

        return self.out(u1)
