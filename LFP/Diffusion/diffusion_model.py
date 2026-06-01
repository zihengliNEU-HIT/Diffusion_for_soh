"""
diffusion_model.py
Content : DiffusionModel wrapper - EMA update, forward pass, DDIM/DDPM sample generation.
Run     : imported only, not executed directly.
"""

import math
import torch
import torch.nn as nn


class DiffusionModel(nn.Module):

    def __init__(self, network, ema_network, gdf_util, timesteps, ema=0.999):
        super().__init__()
        self.network     = network
        self.ema_network = ema_network
        self.timesteps   = timesteps
        self.gdf_util    = gdf_util
        self.ema         = ema
        self.training_step = 0

        total     = sum(p.numel() for p in network.parameters())
        trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print(f"[DiffusionModel] params={total:,}  trainable={trainable:,}  ema={ema}")

    def update_ema(self):
        """Warmup EMA: small ratio early, ramp to target."""
        if self.training_step < 1000:
            current_ema = 0.95
        else:
            current_ema = min(self.ema, 0.95 + 0.05 * (1.0 - math.exp(-(self.training_step - 1000) / 5000)))

        with torch.no_grad():
            for ep, p in zip(self.ema_network.parameters(), self.network.parameters()):
                if p.requires_grad:
                    ep.data.mul_(current_ema).add_(p.data, alpha=1 - current_ema)

        self.training_step += 1

    def forward(self, x, t, condition=None, cond_vec=None):
        return self.network(x, t, condition, cond_vec=cond_vec)

    @torch.no_grad()
    def generate_samples(self, conditions, device="cuda", cond_vec=None,
                         use_ddim=True, ddim_steps=50, ddim_eta=0.0):
        self.ema_network.eval()
        conditions = conditions.to(device)
        if cond_vec is not None:
            cond_vec = cond_vec.to(device)

        B, _, H, W = conditions.shape
        shape      = (B, 1, H, W)

        def model_fn(x, t, cond):
            return self.ema_network(x, t, cond, cond_vec=cond_vec)

        if use_ddim:
            x_T     = torch.randn(shape, device=device)
            samples = self.gdf_util.ddim_sample(
                model=model_fn, cond=conditions,
                steps=ddim_steps, eta=ddim_eta,
                x_T=x_T, clip_denoised=True,
            )
        else:
            samples = self.gdf_util.p_sample_loop(shape, model_fn, condition=conditions)

        return samples
