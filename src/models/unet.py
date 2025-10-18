"""
unet.py

U-Net backbone for conditional SR diffusion (input = concat(noisy_hr, lr)).
Fixed to ensure spatial alignment of skip connections:
- Last DOWN level does NOT downsample (Identity)
- Last UP level does NOT upsample (Identity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- Residual Block --------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


# (Optional) Simple self-attention block — kept here but not used in v1.
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, C, H * W).transpose(1, 2)  # B, HW, C
        k = k.reshape(B, C, H * W)                  # B, C, HW
        v = v.reshape(B, C, H * W).transpose(1, 2)  # B, HW, C
        attn = torch.softmax(torch.bmm(q, k) / (C ** 0.5), dim=-1)  # B, HW, HW
        h = torch.bmm(attn, v).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(h)


# -------- U-Net --------

class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels=6,         # 3 noisy HR + 3 LR condition
        out_channels=3,        # predict eps or v
        base_channels=64,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_resolutions=(),   # not used in v1 to keep simple
        dropout=0.0,
    ):
        super().__init__()
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()

        # Initial conv
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        ch_in = base_channels

        # ---- Down path ----
        feat_channels = []
        L = len(channel_mult)
        for i, mult in enumerate(channel_mult):
            ch_out = base_channels * mult

            # Residual stack at this level
            blocks = nn.ModuleList([ResidualBlock(ch_in, ch_out, dropout)])
            for _ in range(num_res_blocks - 1):
                blocks.append(ResidualBlock(ch_out, ch_out, dropout))
            self.down.append(blocks)

            feat_channels.append(ch_out)
            ch_in = ch_out

            # Downsample except at the last level (keep spatial size)
            if i != L - 1:
                self.downsample.append(nn.Conv2d(ch_in, ch_in, 3, stride=2, padding=1))
            else:
                self.downsample.append(nn.Identity())

        # ---- Middle ----
        self.middle = nn.ModuleList([
            ResidualBlock(ch_in, ch_in, dropout),
            # AttentionBlock(ch_in),  # keep disabled in v1
            ResidualBlock(ch_in, ch_in, dropout),
        ])

        # ---- Up path (mirror) ----
        for i, ch_skip in enumerate(reversed(feat_channels)):
            # concat current feature (ch_in) with corresponding skip (ch_skip)
            blocks = nn.ModuleList(
                [ResidualBlock(ch_in + ch_skip, ch_skip, dropout)]
                + [ResidualBlock(ch_skip, ch_skip, dropout) for _ in range(num_res_blocks - 1)]
            )
            self.up.append(blocks)

            # Upsample except at the final (top) level
            if i != L - 1:
                self.upsample.append(nn.ConvTranspose2d(ch_skip, ch_skip, 4, 2, 1))
            else:
                self.upsample.append(nn.Identity())

            ch_in = ch_skip  # update current channels

        # ---- Output ----
        self.out_norm = nn.GroupNorm(8, ch_in)
        self.out_conv = nn.Conv2d(ch_in, out_channels, 3, padding=1)

    def forward(self, x):
        skips = []
        h = self.input_conv(x)

        # Down
        for blocks, down in zip(self.down, self.downsample):
            for block in blocks:
                h = block(h)
            skips.append(h)   # store feature BEFORE spatial downsample
            h = down(h)

        # Middle
        for m in self.middle:
            h = m(h)

        # Up
        for blocks, up in zip(self.up, self.upsample):
            h = torch.cat([h, skips.pop()], dim=1)  # concat with matching spatial size
            for block in blocks:
                h = block(h)
            h = up(h)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)
