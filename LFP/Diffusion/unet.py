"""
unet.py
Content : Conditional U-Net with sinusoidal time embedding, FiLM modulation from
          scalar cond_vec, and self-attention at the deepest scale.
Run     : imported only, not executed directly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# -- time embedding ------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device   = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# -- attention block -----------------------------------------------------------

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm     = nn.GroupNorm(8, channels)
        self.query    = nn.Conv2d(channels, channels, 1)
        self.key      = nn.Conv2d(channels, channels, 1)
        self.value    = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        for m in [self.query, self.key, self.value, self.proj_out]:
            m.apply(init_weights)

    def forward(self, x):
        h    = self.norm(x)
        q    = self.query(h)
        k    = self.key(h)
        v    = self.value(h)
        B, C, H, W = q.shape
        q    = q.reshape(B, C, -1).permute(0, 2, 1)   # [B, HW, C]
        k    = k.reshape(B, C, -1)                     # [B, C, HW]
        v    = v.reshape(B, C, -1).permute(0, 2, 1)   # [B, HW, C]
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out  = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj_out(out)


# -- FiLM helper (shared pattern for Down / Up / Middle) ----------------------

def _build_film_mlp(cond_dim, out_channels):
    """Returns (mlp1, mlp2) with zero-init last layer."""
    def _mlp():
        m = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_channels * 2))
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
        return m
    return _mlp(), _mlp()


def _apply_film(h, cond_vec, mlp, has_cond):
    if (not has_cond) or (cond_vec is None):
        return h
    cond_vec = cond_vec.to(h.dtype).to(h.device)
    gb       = mlp(cond_vec)
    gamma, beta = gb.chunk(2, dim=1)
    gamma = gamma[:, :, None, None]
    beta  = beta[:, :, None, None]
    scale = 1.2
    return h * (1.0 + scale * gamma) + scale * beta


# -- down block ----------------------------------------------------------------

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False, cond_vector_dim=0):
        super().__init__()
        self.use_attention = use_attention
        self.has_cond      = cond_vector_dim > 0

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv1    = nn.Conv2d(in_channels,  out_channels, 3, padding=1)
        self.norm1    = nn.GroupNorm(8, out_channels)
        self.conv2    = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2    = nn.GroupNorm(8, out_channels)
        self.residual_conv = (nn.Conv2d(in_channels, out_channels, 1)
                              if in_channels != out_channels else nn.Identity())

        if use_attention:
            self.attention = AttentionBlock(out_channels)

        if self.has_cond:
            self.cond_mlp1, self.cond_mlp2 = _build_film_mlp(cond_vector_dim, out_channels)

        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        if isinstance(self.residual_conv, nn.Conv2d):
            self.residual_conv.apply(init_weights)

    def forward(self, x, time_emb, cond_vec=None):
        residual = self.residual_conv(x)

        h = self.norm1(self.conv1(x))
        h = _apply_film(h, cond_vec, self.cond_mlp1, self.has_cond) if self.has_cond else h
        h = F.silu(h)
        h = h + self.time_mlp(time_emb)[:, :, None, None]

        h = self.norm2(self.conv2(h))
        h = _apply_film(h, cond_vec, self.cond_mlp2, self.has_cond) if self.has_cond else h
        h = F.silu(h)

        if self.use_attention:
            h = self.attention(h)
        return h + residual


# -- up block ------------------------------------------------------------------

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False, cond_vector_dim=0):
        super().__init__()
        self.use_attention = use_attention
        self.has_cond      = cond_vector_dim > 0

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv1    = nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1)
        self.norm1    = nn.GroupNorm(8, out_channels)
        self.conv2    = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2    = nn.GroupNorm(8, out_channels)
        self.residual_conv = (nn.Conv2d(in_channels, out_channels, 1)
                              if in_channels != out_channels else nn.Identity())

        if use_attention:
            self.attention = AttentionBlock(out_channels)

        if self.has_cond:
            self.cond_mlp1, self.cond_mlp2 = _build_film_mlp(cond_vector_dim, out_channels)

        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        if isinstance(self.residual_conv, nn.Conv2d):
            self.residual_conv.apply(init_weights)

    def forward(self, x, skip_x, time_emb, cond_vec=None):
        residual = self.residual_conv(x)

        h = self.norm1(self.conv1(torch.cat([x, skip_x], dim=1)))
        h = _apply_film(h, cond_vec, self.cond_mlp1, self.has_cond) if self.has_cond else h
        h = F.silu(h)
        h = h + self.time_mlp(time_emb)[:, :, None, None]

        h = self.norm2(self.conv2(h))
        h = _apply_film(h, cond_vec, self.cond_mlp2, self.has_cond) if self.has_cond else h
        h = F.silu(h)

        if self.use_attention:
            h = self.attention(h)
        return h + residual


# -- middle block --------------------------------------------------------------

class MiddleBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, cond_vector_dim=0):
        super().__init__()
        self.has_cond = cond_vector_dim > 0

        self.time_mlp  = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, channels))
        self.conv1     = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1     = nn.GroupNorm(8, channels)
        self.attention = AttentionBlock(channels)
        self.conv2     = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2     = nn.GroupNorm(8, channels)

        if self.has_cond:
            self.cond_mlp1, self.cond_mlp2 = _build_film_mlp(cond_vector_dim, channels)

        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)

    def forward(self, x, time_emb, cond_vec=None):
        residual = x

        h = self.norm1(self.conv1(x))
        h = _apply_film(h, cond_vec, self.cond_mlp1, self.has_cond) if self.has_cond else h
        h = F.silu(h)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.attention(h)

        h = self.norm2(self.conv2(h))
        h = _apply_film(h, cond_vec, self.cond_mlp2, self.has_cond) if self.has_cond else h
        h = F.silu(h)

        return h + residual


# -- up/downsample -------------------------------------------------------------

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv.apply(init_weights)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, stride=2)
        self.conv.apply(init_weights)

    def forward(self, x):
        return self.conv(x)


# -- condition embedding -------------------------------------------------------

class ConditionEmbedding(nn.Module):
    def __init__(self, condition_channels, emb_dim):
        super().__init__()
        self.conv_in  = nn.Conv2d(condition_channels, emb_dim, 3, padding=1)
        self.conv_mid = nn.Conv2d(emb_dim, emb_dim, 3, padding=1)
        self.conv_out = nn.Conv2d(emb_dim, emb_dim, 3, padding=1)
        for m in [self.conv_in, self.conv_mid, self.conv_out]:
            m.apply(init_weights)

    def forward(self, x):
        h = F.silu(self.conv_in(x))
        h = F.silu(self.conv_mid(h))
        return self.conv_out(h)


# -- UNet ----------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        condition_channels=1,
        model_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        condition_emb_dim=64,
        use_attention=(False, False, True),
        num_res_blocks=2,
        cond_vector_dim=0,
    ):
        super().__init__()
        self.num_res_blocks    = num_res_blocks
        self.cond_vector_dim   = cond_vector_dim

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.condition_embedding = ConditionEmbedding(condition_channels, condition_emb_dim)
        self.conv_in  = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.cond_proj = nn.Conv2d(condition_emb_dim, model_channels, 1)
        self.cond_proj.apply(init_weights)

        # encoder
        self.down_blocks  = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = model_channels
        for i, mult in enumerate(channel_mults):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    DownBlock(in_ch, out_ch, time_emb_dim, use_attention[i], cond_vector_dim)
                )
                in_ch = out_ch
            if i != len(channel_mults) - 1:
                self.downsamplers.append(Downsample(in_ch))

        self.middle_block = MiddleBlock(in_ch, time_emb_dim, cond_vector_dim)

        # decoder
        self.up_blocks   = nn.ModuleList()
        self.upsamplers  = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.up_blocks.append(
                    UpBlock(in_ch, out_ch, time_emb_dim, use_attention[i], cond_vector_dim)
                )
                in_ch = out_ch
            if i != 0:
                self.upsamplers.append(Upsample(in_ch))

        self.norm_out = nn.GroupNorm(8, in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)
        self.conv_in.apply(init_weights)
        self.conv_out.apply(init_weights)

    def forward(self, x, timesteps, condition=None, cond_vec=None):
        time_emb = self.time_embedding(timesteps)

        h = self.conv_in(x)
        if condition is not None:
            h = h + self.cond_proj(self.condition_embedding(condition))

        skips = [h]

        for i, down_block in enumerate(self.down_blocks):
            h = down_block(h, time_emb, cond_vec=cond_vec)
            skips.append(h)
            if (i + 1) % self.num_res_blocks == 0 and (i // self.num_res_blocks) < len(self.downsamplers):
                h = self.downsamplers[i // self.num_res_blocks](h)

        h = self.middle_block(h, time_emb, cond_vec=cond_vec)

        for i, up_block in enumerate(self.up_blocks):
            if i > 0 and i % self.num_res_blocks == 0:
                h = self.upsamplers[i // self.num_res_blocks - 1](h)
            h = up_block(h, skips.pop(), time_emb, cond_vec=cond_vec)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)
