"""
gaussian_diffusion.py
Content : GaussianDiffusion class - forward process, DDPM sampler, DDIM sampler,
          noise-estimation loss with adaptive weighting and reconstruction term.
Run     : imported only, not executed directly.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GaussianDiffusion:

    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.012,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
        device="cuda",
        schedule_type="cosine",
        cosine_s: float = 0.008,
    ):
        self.beta_start   = beta_start
        self.beta_end     = beta_end
        self.timesteps    = timesteps
        self.clip_min     = clip_min
        self.clip_max     = clip_max
        self.device       = device
        self.schedule_type = schedule_type
        self.num_timesteps = int(timesteps)

        print(f"[GaussianDiffusion] schedule={schedule_type}, steps={timesteps}")

        # beta schedule
        if schedule_type == "linear":
            self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        elif schedule_type == "cosine":
            steps = timesteps + 1
            s = float(cosine_s)
            x = np.linspace(0, timesteps, steps)
            ac = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            ac = ac / ac[0]
            betas = 1 - (ac[1:] / ac[:-1])
            self.betas = np.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        # derived quantities
        alphas            = 1.0 - self.betas
        alphas_cumprod    = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance  = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        def _t(x): return torch.tensor(x, dtype=torch.float32)

        self.betas                        = _t(self.betas)
        self.alphas_cumprod               = _t(alphas_cumprod)
        self.alphas_cumprod_prev          = _t(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod          = _t(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod= _t(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = _t(np.log(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod    = _t(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod  = _t(np.sqrt(1.0 / alphas_cumprod - 1))
        self.posterior_variance           = _t(posterior_variance)
        self.posterior_log_variance_clipped = _t(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = _t(self.betas.numpy() * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = _t((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

        self._to_device(device)

        # save schedule plot
        try:
            os.makedirs("./output", exist_ok=True)
            self._save_noise_schedule("./output/noise_schedule.png")
        except Exception as e:
            print(f"Cannot save noise schedule plot: {e}")

    # -- helpers --------------------------------------------------------------

    def _to_device(self, device):
        for attr in [
            "betas","alphas_cumprod","alphas_cumprod_prev",
            "sqrt_alphas_cumprod","sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod","sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod","posterior_variance",
            "posterior_log_variance_clipped","posterior_mean_coef1","posterior_mean_coef2",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))

    def _save_noise_schedule(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, data, title in zip(
            axes.flatten(),
            [self.betas, self.alphas_cumprod,
             self.sqrt_one_minus_alphas_cumprod, self.posterior_variance],
            ["Betas","Alphas Cumprod","Noise Ratio","Posterior Variance"],
        ):
            ax.plot(data.cpu().numpy()); ax.set_title(title); ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _extract(self, a, t, x_shape):
        """Gather coefficients at timestep t and reshape for broadcasting."""
        t   = t.to(a.device)
        out = a.gather(-1, t)
        return out.reshape(x_shape[0], *((1,) * (len(x_shape) - 1)))

    # -- forward process -------------------------------------------------------

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        var  = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        lv   = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, var, lv

    def q_sample(self, x_start, t, noise=None):
        """Add noise to x_start at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # -- reverse process -------------------------------------------------------

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x0_pred = self.predict_start_from_noise(x, t, pred_noise)
        if clip_denoised:
            x0_pred = x0_pred.clamp(self.clip_min, self.clip_max)
        mean = (
            self._extract(self.posterior_mean_coef1, t, x.shape) * x0_pred
            + self._extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        log_var = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        return mean, None, log_var

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Single DDPM reverse step."""
        mean, _, log_var = self.p_mean_variance(pred_noise, x, t, clip_denoised)
        noise       = torch.randn_like(x)
        nonzero     = (1 - (t == 0).float()).reshape(x.shape[0], *([1] * (len(x.shape) - 1)))
        variance    = (0.5 * log_var).exp().clamp(max=0.99)
        return mean + nonzero * variance * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, model, condition=None, clip_denoised=True):
        """Full DDPM reverse chain (slow, O(T) steps)."""
        device = self.betas.device
        img    = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t       = torch.full((shape[0],), i, device=device, dtype=torch.long)
            pred_n  = model(img, t, condition)
            img     = self.p_sample(pred_n, img, t, clip_denoised)
        return img

    # -- DDIM sampler ----------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(self, model, cond=None, steps=50, eta=0.0, x_T=None, clip_denoised=True):
        """DDIM reverse chain.  eta=0 -> deterministic; eta=1 -> stochastic."""
        device    = self.betas.device
        step_size = self.num_timesteps // steps
        idx       = torch.arange(self.num_timesteps - 1, -1, -step_size, device=device).long()
        if idx[-1] != 0:
            idx[-1] = 0

        if x_T is None:
            b, h, w = cond.shape[0], cond.shape[-2], cond.shape[-1]
            x = torch.randn((b, 1, h, w), device=device)
        else:
            x = x_T.to(device)

        for i, t in enumerate(idx):
            t_vec      = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            eps        = model(x, t_vec, cond)
            ab_t       = self._extract(self.alphas_cumprod, t_vec, x.shape)
            ab_t_sqrt  = self._extract(self.sqrt_alphas_cumprod, t_vec, x.shape)
            one_m_sqrt = self._extract(self.sqrt_one_minus_alphas_cumprod, t_vec, x.shape)
            x0_hat     = (x - one_m_sqrt * eps) / ab_t_sqrt
            if clip_denoised:
                x0_hat = x0_hat.clamp(self.clip_min, self.clip_max)
            if i == len(idx) - 1:
                return x0_hat
            t_prev     = idx[i + 1]
            tp_vec     = torch.full((x.size(0),), t_prev, device=device, dtype=torch.long)
            ab_prev    = self._extract(self.alphas_cumprod, tp_vec, x.shape)
            sigma      = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
            noise      = torch.randn_like(x) if eta > 0 else 0
            x = (torch.sqrt(ab_prev) * x0_hat
                 + torch.sqrt(1 - ab_prev - sigma ** 2) * eps
                 + sigma * noise)
        return x

    # -- training loss ---------------------------------------------------------

    def noise_estimation_loss(self, model, x_start, t, condition=None, noise=None):
        """
        Hybrid loss: noise-prediction MSE/L1 + reconstructed-x0 MSE/L1
        with adaptive time-dependent weighting.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy    = self.q_sample(x_start, t, noise)
        pred_noise = model(x_noisy, t, condition)

        # noise prediction loss
        mse_loss   = torch.nn.functional.mse_loss(noise, pred_noise)
        l1_loss    = torch.nn.functional.l1_loss(noise, pred_noise)
        noise_loss = 0.9 * mse_loss + 0.1 * l1_loss

        # reconstruction loss
        pred_x0    = self.predict_start_from_noise(x_noisy, t, pred_noise)
        pred_x0    = pred_x0.clamp(self.clip_min, self.clip_max)
        recon_loss = 0.8 * torch.nn.functional.mse_loss(pred_x0, x_start) \
                   + 0.2 * torch.nn.functional.l1_loss(pred_x0, x_start)

        # adaptive weight schedule
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0
        self._step_counter += 1
        progress      = min(self._step_counter / 1_000_000, 1.0)
        noise_weight  = 0.7 - 0.3 * progress
        recon_weight  = 0.3 + 0.3 * progress

        # time-step weighting (lower-noise steps weighted more)
        t_norm  = t.float() / self.timesteps
        tw      = torch.exp(-2 * t_norm).view(-1, 1, 1, 1)
        w_recon = (recon_loss * tw).mean()

        loss = noise_weight * noise_loss + recon_weight * w_recon

        # periodic logging
        if self._step_counter % 2000 == 0:
            with torch.no_grad():
                p  = (pred_x0 + 1.0) / 2.0
                tg = (x_start + 1.0) / 2.0
                mse_psnr = torch.mean((p - tg) ** 2)
                psnr     = 20 * torch.log10(1.0 / torch.sqrt(mse_psnr)) if mse_psnr > 0 else torch.tensor(100.0)
                print(f"[Step {self._step_counter}] "
                      f"noise={noise_loss.item():.4f}  recon={w_recon.item():.4f}  "
                      f"total={loss.item():.4f}  PSNR={psnr.item():.2f}dB")

        return loss
