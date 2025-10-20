# minimal-satSR/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Building blocks
# ----------------------------

def norm(ch: int) -> nn.GroupNorm:
    # GroupNorm is stable for small batch sizes
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)

class ResBlock(nn.Module):
    """
    Residual block: GN -> SiLU -> Conv(3x3) -> Dropout -> GN -> SiLU -> Conv(3x3)
    With a learnable skip if in/out channels differ.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.block1 = nn.Sequential(norm(in_ch), nn.SiLU(inplace=True),
                                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.block2 = nn.Sequential(norm(out_ch), nn.SiLU(inplace=True),
                                    nn.Dropout(dropout),
                                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip(x)

class SelfAttention2d(nn.Module):
    """
    Multi-head self-attention over 2D feature maps.
    Uses 1x1 convs for q/k/v and a 1x1 proj back to channels.
    Operates at (B, C, H, W). Attention is computed over HW tokens.
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
        qkv = self.qkv(x)  # (B, 3C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # each (B, C, H, W)

        # reshape to (B, heads, dim, HW)
        def reshape_heads(t):
            t = t.view(b, self.num_heads, self.head_dim, h * w)  # (B,H,D,HW)
            return t

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.softmax(torch.einsum("bhdk,bhdn->bhkn", q * scale, k), dim=-1)  # (B,H,HW,HW)
        out = torch.einsum("bhkn,bhdn->bhdk", attn, v)  # (B,H,D,HW)

        # merge heads
        out = out.contiguous().view(b, -1, h * w)  # (B,C,HW)
        out = out.view(b, -1, h, w)  # (B,C,H,W)
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
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

# ----------------------------
# High-performance U-Net
# ----------------------------

class UNetSmall(nn.Module):
    """
    Upgraded U-Net (residual + attention) with the SAME signature as before.

    Input:  [B, in_ch,  H, W]   (e.g., [x_t | LR | optional s_map])
    Output: [B, out_ch, H, W]   (epsilon prediction or x0 if you change the loss)

    Notes:
    - Depth: 4 stages total (3 downsamples -> spatial: 1.0x, 0.5x, 0.25x, 0.125x)
    - Channels: base -> 2x -> 4x -> 4x (capped at 4x base to keep memory reasonable)
    - Attention at lower resolutions where it matters (0.25x and 0.125x by default).
    - Keep defaults identical to previous signature: in_ch=6, base=64, out_ch=3
      (If you use an extra s_map channel, pass in_ch=7 from your training/infer code.)
    """
    def __init__(self, in_ch: int = 7, base: int = 64, out_ch: int = 3):
        super().__init__()
        ch1 = base
        ch2 = base * 2
        ch3 = base * 4
        ch4 = base * 4  # cap the deepest width

        # --- Encoder ---
        # Stage 1 (H, W)
        self.enc1_1 = ResBlock(in_ch, ch1, dropout=0.1)
        self.enc1_2 = ResBlock(ch1, ch1, dropout=0.1)
        self.down1  = Downsample(ch1)

        # Stage 2 (H/2, W/2)
        self.enc2_1 = ResBlock(ch1, ch2, dropout=0.1)
        self.enc2_2 = ResBlock(ch2, ch2, dropout=0.1)
        self.down2  = Downsample(ch2)

        # Stage 3 (H/4, W/4) + attention
        self.enc3_1 = ResBlock(ch2, ch3, dropout=0.1)
        self.enc3_2 = ResBlock(ch3, ch3, dropout=0.1)
        self.attn3  = SelfAttention2d(ch3, num_heads=4)
        self.down3  = Downsample(ch3)

        # Bottleneck (H/8, W/8) + attention
        self.mid_1  = ResBlock(ch3, ch4, dropout=0.1)
        self.mid_attn = SelfAttention2d(ch4, num_heads=4)
        self.mid_2  = ResBlock(ch4, ch4, dropout=0.1)

        # --- Decoder ---
        self.up3    = Upsample(ch4)
        self.dec3_1 = ResBlock(ch4 + ch3, ch3, dropout=0.1)
        self.dec3_2 = ResBlock(ch3, ch3, dropout=0.1)
        self.dec3_attn = SelfAttention2d(ch3, num_heads=4)

        self.up2    = Upsample(ch3)
        self.dec2_1 = ResBlock(ch3 + ch2, ch2, dropout=0.1)
        self.dec2_2 = ResBlock(ch2, ch2, dropout=0.1)

        self.up1    = Upsample(ch2)
        self.dec1_1 = ResBlock(ch2 + ch1, ch1, dropout=0.1)
        self.dec1_2 = ResBlock(ch1, ch1, dropout=0.1)

        # Output head
        self.out = nn.Sequential(
            norm(ch1), nn.SiLU(inplace=True),
            nn.Conv2d(ch1, out_ch, kernel_size=3, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)  # safe default for SiLU
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
        m  = self.mid_1(d3)
        m  = self.mid_attn(m)
        m  = self.mid_2(m)

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

        return self.out(u1)
