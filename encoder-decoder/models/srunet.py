# encoder-decoder/models/srunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Building blocks
# ----------------------------

def norm(ch: int) -> nn.GroupNorm:
    # GroupNorm is batch-size agnostic and stable for SR
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)

class ResBlock(nn.Module):
    """
    Residual Conv block with GroupNorm and SiLU.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.block1 = nn.Sequential(
            norm(in_ch), nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            norm(out_ch), nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip(x)

class SelfAttention2d(nn.Module):
    """
    Lightweight multi-head self-attention over 2D feature maps.
    Operates on (B, C, H, W); attention over HW tokens.
    """
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        assert ch % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        self.qkv = nn.Conv2d(ch, ch * 3, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)
        self.norm = norm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        def split_heads(t):
            return t.view(b, self.num_heads, self.head_dim, h * w)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.softmax(torch.einsum("bhdk,bhdn->bhkn", q * scale, k), dim=-1)  # (B,H,HW,HW)
        out = torch.einsum("bhkn,bhdn->bhdk", attn, v)  # (B,H,D,HW)

        out = out.contiguous().view(b, c, h, w)
        out = self.proj(out)
        return out + x_in

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)

# ----------------------------
# Super-Resolution U-Net (encoder–decoder)
# ----------------------------

class SRUNet(nn.Module):
    """
    Encoder–decoder for SR with residual prediction.

    Args:
        in_ch:   input channels (default 3 for LR RGB)
        base:    base channel width (64 is a solid default)
        out_ch:  output channels (3 for RGB detail map or SR)
        residual: if True, network predicts Δ and returns LR + Δ.
                  if False, returns direct prediction.
        use_attn: enable a small attention block at 128×128 and bottleneck.

    Input:
        x: [B, 3, H, W] in [0,1] (LR upsampled to target size, e.g., 512x512)

    Output:
        If residual=True: SR = x + Δ (not clamped here; clamp outside)
        If residual=False: direct prediction (same shape as input)
    """
    def __init__(self, in_ch: int = 3, base: int = 64, out_ch: int = 3,
                 residual: bool = True, use_attn: bool = True):
        super().__init__()
        self.residual = residual
        self.use_attn = use_attn

        c1 = base          # 64
        c2 = base * 2      # 128
        c3 = base * 4      # 256
        c4 = base * 4      # 256 (cap depth to keep memory moderate)

        # Encoder
        self.enc1_1 = ResBlock(in_ch, c1, dropout=0.0)
        self.enc1_2 = ResBlock(c1,   c1, dropout=0.0)
        self.down1  = Downsample(c1)           # /2

        self.enc2_1 = ResBlock(c1,   c2, dropout=0.0)
        self.enc2_2 = ResBlock(c2,   c2, dropout=0.0)
        self.down2  = Downsample(c2)           # /4

        self.enc3_1 = ResBlock(c2,   c3, dropout=0.0)
        self.enc3_2 = ResBlock(c3,   c3, dropout=0.0)
        self.attn3  = SelfAttention2d(c3, num_heads=4) if use_attn else nn.Identity()
        self.down3  = Downsample(c3)           # /8

        # Bottleneck
        self.mid_1    = ResBlock(c3, c4, dropout=0.0)
        self.mid_attn = SelfAttention2d(c4, num_heads=4) if use_attn else nn.Identity()
        self.mid_2    = ResBlock(c4, c4, dropout=0.0)

        # Decoder
        self.up3    = Upsample(c4)             # /4
        self.dec3_1 = ResBlock(c4 + c3, c3, dropout=0.0)
        self.dec3_2 = ResBlock(c3,      c3, dropout=0.0)
        self.dec3_attn = SelfAttention2d(c3, num_heads=4) if use_attn else nn.Identity()

        self.up2    = Upsample(c3)             # /2
        self.dec2_1 = ResBlock(c3 + c2, c2, dropout=0.0)
        self.dec2_2 = ResBlock(c2,      c2, dropout=0.0)

        self.up1    = Upsample(c2)             # /1
        self.dec1_1 = ResBlock(c2 + c1, c1, dropout=0.0)
        self.dec1_2 = ResBlock(c1,      c1, dropout=0.0)

        # Output head predicts detail Δ (or direct RGB if residual=False)
        self.head = nn.Sequential(
            norm(c1), nn.SiLU(inplace=True),
            nn.Conv2d(c1, out_ch, kernel_size=3, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)  # robust default for SiLU
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1_1(x)
        e1 = self.enc1_2(e1)
        d1 = self.down1(e1)

        e2 = self.enc2_1(d1)
        e2 = self.enc2_2(e2)
        d2 = self.down2(e2)

        e3 = self.enc3_1(d2)
        e3 = self.enc3_2(e3)
        e3 = self.attn3(e3)
        d3 = self.down3(e3)

        # Bottleneck
        m = self.mid_1(d3)
        m = self.mid_attn(m)
        m = self.mid_2(m)

        # Decoder
        u3 = self.up3(m)
        u3 = torch.cat([u3, e3], dim=1)
        u3 = self.dec3_1(u3)
        u3 = self.dec3_2(u3)
        u3 = self.dec3_attn(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2_1(u2)
        u2 = self.dec2_2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1_1(u1)
        u1 = self.dec1_2(u1)

        delta = self.head(u1)
        return x + delta if self.residual else delta
