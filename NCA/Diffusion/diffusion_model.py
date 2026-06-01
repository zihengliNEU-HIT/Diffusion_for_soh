import torch
import torch.nn as nn
import math


class DiffusionModel(nn.Module):

    def __init__(self, network, ema_network, gdf_util, timesteps, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema
        self.training_step = 0

    def update_ema(self):
        if self.training_step < 1000:
            current_ema = 0.95
        else:
            current_ema = min(
                self.ema,
                0.95 + 0.05 * (1.0 - math.exp(-(self.training_step - 1000) / 5000)),
            )
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_network.parameters(), self.network.parameters()
            ):
                if param.requires_grad:
                    ema_param.data.mul_(current_ema).add_(
                        param.data, alpha=1 - current_ema
                    )
        self.training_step += 1

    def forward(self, x, t, condition=None, cond_vec=None):
        return self.network(x, t, condition, cond_vec=cond_vec)

    @torch.no_grad()
    def generate_samples(
        self,
        conditions,
        device="cuda",
        cond_vec=None,
        use_ddim=True,
        ddim_steps=50,
        ddim_eta=0.0,
    ):
        self.ema_network.eval()
        conditions = conditions.to(device)
        if cond_vec is not None and torch.is_tensor(cond_vec):
            cond_vec = cond_vec.to(device)
        batch_size = conditions.shape[0]
        h, w = (conditions.shape[2], conditions.shape[3])
        shape = (batch_size, 1, h, w)
        if use_ddim:

            def model_fn(x, t, cond):
                return self.ema_network(x, t, cond, cond_vec=cond_vec)

            x_T = torch.randn(shape, device=device)
            samples = self.gdf_util.ddim_sample(
                model=model_fn,
                cond=conditions,
                steps=ddim_steps,
                eta=ddim_eta,
                x_T=x_T,
                clip_denoised=True,
            )
        else:

            def model_fn(x, t, cond):
                return self.ema_network(x, t, cond, cond_vec=cond_vec)

            samples = self.gdf_util.p_sample_loop(shape, model_fn, condition=conditions)
        return samples
