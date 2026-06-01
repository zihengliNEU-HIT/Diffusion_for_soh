"""
SOH_model.py (v3 - image-only with spatial attention and configurable channels)
-----------------------------------------------------------------------------
Architecture:
  Input : GAF (B, 1, 128, 128)
  CNN   : 4 x ResBlock, channels controlled by cnn_channels, stride 2 each stage
          128x128 -> 64 -> 32 -> 16 -> 8
  Trans : img_tokens (B, 64, cnn_channels[-1])
          Multi-Head Self-Attn -> attention weights
          FFN
          mean-pool -> Linear -> SOH (B, 1)

Revision notes:
  v3: CNNEncoder receives cnn_channels instead of using fixed channel sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN building blocks

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        g = lambda c: min(8, c)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(g(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(g(out_ch), out_ch)
        self.act   = nn.SiLU()

        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.GroupNorm(g(out_ch), out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x):
        h = self.act(self.gn1(self.conv1(x)))
        h = self.act(self.gn2(self.conv2(h)))
        return h + self.skip(x)


class CNNEncoder(nn.Module):
    """
    Encode (B, 1, 128, 128) GAF images into
    (B, cnn_channels[-1], 8, 8) features.

    Channel sizes are fully controlled by cnn_channels.
    """
    def __init__(self, cnn_channels=(32, 64, 128, 256)):
        super().__init__()
        c = cnn_channels
        self.b1 = ResBlock(1,    c[0], stride=2)
        self.b2 = ResBlock(c[0], c[1], stride=2)
        self.b3 = ResBlock(c[1], c[2], stride=2)
        self.b4 = ResBlock(c[2], c[3], stride=2)

    def forward(self, x):
        return self.b4(self.b3(self.b2(self.b1(x))))  # (B, c[3], 8, 8)


# Transformer blocks

class SelfAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout,
            batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn: bool = False):
        h = self.norm(x)
        h, attn_w = self.attn(h, h, h, average_attn_weights=True)
        out = x + self.drop(h)
        if return_attn:
            return out, attn_w
        return out


class FFNBlock(nn.Module):
    def __init__(self, dim: int, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net  = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class TransformerRegressor(nn.Module):
    """
    Regress SOH from (B, feat_dim, 8, 8) features.

    Optionally returns a spatial attention map with shape (B, 8, 8).
    """
    def __init__(self, feat_dim: int = 256, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttnBlock(feat_dim, num_heads, dropout)
        self.ffn       = FFNBlock(feat_dim, expand=2, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, feat_map, return_attn: bool = False):
        B, C, H, W = feat_map.shape
        img_tokens = feat_map.flatten(2).permute(0, 2, 1)   # (B, HW, C)

        if return_attn:
            img_tokens, attn_w = self.self_attn(img_tokens, return_attn=True)
            spatial_attn = attn_w.mean(dim=1).reshape(B, H, W)  # (B, 8, 8)
        else:
            img_tokens = self.self_attn(img_tokens)
            spatial_attn = None

        img_tokens = self.ffn(img_tokens)
        pooled = img_tokens.mean(dim=1)      # (B, C)
        soh = self.head(pooled)              # (B, 1)

        if return_attn:
            return soh, spatial_attn
        return soh


# Top-level estimator

class SOHEstimator(nn.Module):
    """
    forward(gaf, return_attn=False) -> soh (B, 1)
    forward(gaf, return_attn=True)  -> (soh, spatial_attn (B, 8, 8))

    gaf : (B, 1, 128, 128)  [-1, 1]
    """
    def __init__(self, cnn_channels=(32, 64, 128, 256),
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert len(cnn_channels) == 4
        assert cnn_channels[-1] % num_heads == 0, \
            f"cnn_channels[-1]={cnn_channels[-1]} 必须能被 num_heads={num_heads} 整除"

        self.cnn = CNNEncoder(cnn_channels)           # pass cnn_channels through
        self.transformer = TransformerRegressor(
            feat_dim=cnn_channels[-1],                # match the CNN output width
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, gaf, return_attn: bool = False):
        feat = self.cnn(gaf)                          # (B, cnn_channels[-1], 8, 8)
        return self.transformer(feat, return_attn)


def build_model(cfg) -> SOHEstimator:
    model = SOHEstimator(
        cnn_channels=cfg.cnn_channels,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(cfg.device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SOHEstimator (image-only): {trainable:,} trainable / {total:,} total params")
    return model
