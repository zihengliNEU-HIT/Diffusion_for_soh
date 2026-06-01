import numpy as np
import torch
import matplotlib.pyplot as plt
import os


class GaussianDiffusion:

    def __init__(
        self,
        beta_start=0.0001,
        beta_end=0.012,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
        device="cuda",
        schedule_type="cosine",
        cosine_s: float = 0.008,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device
        self.schedule_type = schedule_type
        self.num_timesteps = int(timesteps)
        if schedule_type == "linear":
            self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        elif schedule_type == "cosine":
            steps = timesteps + 1
            s = float(cosine_s)
            x = np.linspace(0, timesteps, steps)
            alphas_cumprod = np.cos((x / timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = np.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unsupported noise schedule: {schedule_type}")
        alphas = 1.0 - self.betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.betas = torch.tensor(self.betas, dtype=torch.float32)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
        self.alphas_cumprod_prev = torch.tensor(
            alphas_cumprod_prev, dtype=torch.float32
        )
        self.sqrt_alphas_cumprod = torch.tensor(
            np.sqrt(alphas_cumprod), dtype=torch.float32
        )
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32
        )
        self.log_one_minus_alphas_cumprod = torch.tensor(
            np.log(1.0 - alphas_cumprod), dtype=torch.float32
        )
        self.sqrt_recip_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 / alphas_cumprod), dtype=torch.float32
        )
        self.sqrt_recipm1_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=torch.float32
        )
        posterior_variance = (
            self.betas.numpy() * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = torch.tensor(posterior_variance, dtype=torch.float32)
        self.posterior_log_variance_clipped = torch.tensor(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch.float32
        )
        self.posterior_mean_coef1 = torch.tensor(
            self.betas.numpy() * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=torch.float32,
        )
        self.posterior_mean_coef2 = torch.tensor(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=torch.float32,
        )
        self._to_device(device)
        try:
            os.makedirs("./output", exist_ok=True)
            self._save_noise_schedule("./output/noise_schedule.png")
        except Exception as e:
            pass

    def _to_device(self, device):
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device
        )
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(
            device
        )
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)

    def _save_noise_schedule(self, save_path):
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.betas.cpu().numpy())
        plt.title("Betas")
        plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.plot(self.alphas_cumprod.cpu().numpy())
        plt.title("Alphas Cumprod")
        plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot(self.sqrt_one_minus_alphas_cumprod.cpu().numpy())
        plt.title("Noise Ratio")
        plt.grid(True)
        plt.subplot(2, 2, 4)
        plt.plot(self.posterior_variance.cpu().numpy())
        plt.title("Posterior Variance")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _extract(self, a, t, x_shape):
        batch_size = x_shape[0]
        t = t.to(a.device)
        out = a.gather(-1, t)
        return out.reshape(batch_size, *(1,) * (len(x_shape) - 1))

    def q_mean_variance(self, x_start, t):
        x_start_shape = x_start.shape
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        return (mean, variance, log_variance)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_start_shape = x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = x_t.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        x_t_shape = x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, self.clip_min, self.clip_max)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        posterior_variance = posterior_variance.clamp(min=1e-20)
        posterior_log_variance = torch.log(posterior_variance)
        return (model_mean, posterior_variance, posterior_log_variance)

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(
            x.shape[0], *[1] * (len(x.shape) - 1)
        )
        variance = (0.5 * model_log_variance).exp().clamp(max=0.99)
        return model_mean + nonzero_mask * variance * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, model, condition=None, clip_denoised=True):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in reversed(range(0, self.num_timesteps)):
            if i % 50 == 0 or i == self.num_timesteps - 1:
                pass
            timesteps = torch.full((b,), i, device=device, dtype=torch.long)
            pred_noise = model(img, timesteps, condition)
            img = self.p_sample(pred_noise, img, timesteps, clip_denoised=clip_denoised)
        return img

    def noise_estimation_loss(self, model, x_start, t, condition=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t, condition)
        mse_loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        l1_loss = torch.nn.functional.l1_loss(noise, predicted_noise)
        noise_loss = 0.8 * mse_loss + 0.2 * l1_loss
        predicted_x0 = self.predict_start_from_noise(x_noisy, t, predicted_noise)
        predicted_x0 = torch.clamp(predicted_x0, self.clip_min, self.clip_max)
        recon_mse = torch.nn.functional.mse_loss(predicted_x0, x_start)
        recon_l1 = torch.nn.functional.l1_loss(predicted_x0, x_start)
        recon_loss = 0.8 * recon_mse + 0.2 * recon_l1
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0
        self._step_counter += 1
        progress = min(self._step_counter / 23000, 1.0)
        noise_weight = 0.7 - 0.3 * progress
        recon_weight = 0.3 + 0.3 * progress
        t_normalized = t.float() / self.timesteps
        time_weight = torch.exp(-2 * t_normalized)
        time_weight = time_weight.view(-1, 1, 1, 1)
        weighted_recon_loss = (recon_loss * time_weight).mean()
        loss = noise_weight * noise_loss + recon_weight * weighted_recon_loss
        if self._step_counter % 2000 == 0:
            with torch.no_grad():
                pred_psnr = (predicted_x0 + 1.0) / 2.0
                target_psnr = (x_start + 1.0) / 2.0
                mse_psnr = torch.mean((pred_psnr - target_psnr) ** 2)
                if mse_psnr > 0:
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_psnr))
                else:
                    psnr = torch.tensor(100.0)
                if psnr.item() > 38:
                    pass
                elif psnr.item() > 35:
                    pass
                elif psnr.item() > 30:
                    pass
        return loss

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        cond=None,
        steps: int = 50,
        eta: float = 0.0,
        x_T: torch.Tensor = None,
        clip_denoised: bool = True,
    ):
        device = self.betas.device
        step_size = self.num_timesteps // steps
        idx = torch.arange(self.num_timesteps - 1, -1, -step_size, device=device).long()
        if idx[-1] != 0:
            idx[-1] = 0
        if x_T is None:
            if cond is not None:
                batch_size = cond.shape[0]
                h, w = cond.shape[-2:]
                x = torch.randn((batch_size, 1, h, w), device=device)
            else:
                raise ValueError("Either x_T or cond must be provided to infer shape.")
        else:
            x = x_T.to(device)
        for i, t in enumerate(idx):
            t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            eps_theta = model(x, t_tensor, cond)
            alpha_bar_t = self._extract(self.alphas_cumprod, t_tensor, x.shape)
            alpha_bar_t_sqrt = self._extract(
                self.sqrt_alphas_cumprod, t_tensor, x.shape
            )
            one_minus_alpha_bar_t_sqrt = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape
            )
            x0_hat = (x - one_minus_alpha_bar_t_sqrt * eps_theta) / alpha_bar_t_sqrt
            if clip_denoised:
                x0_hat = torch.clamp(x0_hat, self.clip_min, self.clip_max)
            if i == len(idx) - 1:
                return x0_hat
            t_prev = idx[i + 1]
            t_prev_tensor = torch.full(
                (x.size(0),), t_prev, device=device, dtype=torch.long
            )
            alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev_tensor, x.shape)
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev)
                / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )
            noise = torch.randn_like(x) if eta > 0 else 0
            x = (
                torch.sqrt(alpha_bar_prev) * x0_hat
                + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_theta
                + sigma * noise
            )
        return x
