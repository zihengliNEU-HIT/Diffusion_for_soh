import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import math
from dataset import get_dataloaders
from gaussian_diffusion import GaussianDiffusion
from unet import UNet
from diffusion_model import DiffusionModel
from visualization import save_comparison_grid


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.network.load_state_dict(checkpoint["model_state_dict"])
    model.ema_network.load_state_dict(checkpoint["ema_model_state_dict"])
    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    print(f"Loaded checkpoint from epoch {epoch} with validation loss {val_loss:.4f}")
    return (epoch, val_loss)


def calculate_metrics(target, generated):
    target = ((target + 1.0) / 2.0).astype(np.float32)
    generated = ((generated + 1.0) / 2.0).astype(np.float32)
    mse = mean_squared_error(target.flatten(), generated.flatten())
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 1.0
        psnr = 20 * math.log10(max_pixel) - 10 * math.log10(mse)
    target_mean = np.mean(target)
    generated_mean = np.mean(generated)
    target_var = np.var(target)
    generated_var = np.var(generated)
    target_generated_cov = np.mean(
        (target - target_mean) * (generated - generated_mean)
    )
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    luminance = (2 * target_mean * generated_mean + C1) / (
        target_mean**2 + generated_mean**2 + C1
    )
    contrast = (2 * np.sqrt(target_var) * np.sqrt(generated_var) + C2) / (
        target_var + generated_var + C2
    )
    structure = (target_generated_cov + C2 / 2) / (
        np.sqrt(target_var) * np.sqrt(generated_var) + C2 / 2
    )
    ssim = luminance * contrast * structure
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def test_model(diffusion_model, test_loader, output_dir, device, args, num_samples=16):
    diffusion_model.ema_network.eval()
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []
    sample_inputs = []
    sample_targets = []
    sample_generated = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            cond_vec = batch.get("cond_vec", None)
            if cond_vec is not None and torch.is_tensor(cond_vec):
                cond_vec = cond_vec.to(device)
            generated = diffusion_model.generate_samples(
                inputs,
                device=device,
                cond_vec=cond_vec,
                use_ddim=args.use_ddim,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
            )
            for i in range(inputs.shape[0]):
                target = targets[i, 0].cpu().numpy()
                gen = generated[i, 0].cpu().numpy()
                metrics = calculate_metrics(target, gen)
                all_metrics.append(metrics)
                if len(sample_inputs) < num_samples:
                    sample_inputs.append(inputs[i].cpu().numpy())
                    sample_targets.append(target)
                    sample_generated.append(gen)
    avg_metrics = {
        "mse": np.mean([m["mse"] for m in all_metrics]),
        "psnr": np.mean([m["psnr"] for m in all_metrics]),
        "ssim": np.mean([m["ssim"] for m in all_metrics]),
    }
    print("\nTest Results:")
    print(f"Average MSE: {avg_metrics['mse']:.4f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average SSIM: {avg_metrics['ssim']:.4f}")
    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Average MSE: {avg_metrics['mse']:.4f}\n")
        f.write(f"Average PSNR: {avg_metrics['psnr']:.2f} dB\n")
        f.write(f"Average SSIM: {avg_metrics['ssim']:.4f}\n")
    save_comparison_grid(
        sample_inputs,
        sample_targets,
        sample_generated,
        os.path.join(output_dir, "test_samples.png"),
        num_samples=min(num_samples, len(sample_inputs)),
    )
    return avg_metrics


def main(args=None):
    if args is None:
        from cs_main import parse_args

        args = parse_args()
    device = torch.device(args.device)
    print(f"[test] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    _, _, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    from cs_main import create_model

    diffusion_model = create_model(args)
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        best_model_path = os.path.join(args.checkpoint_dir, "model_epoch_220.pt")
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
        else:
            raise ValueError(
                "No checkpoint_path was provided and no default checkpoint was found. Please provide a valid checkpoint path."
            )
    load_checkpoint(diffusion_model, checkpoint_path, device)
    test_metrics = test_model(
        diffusion_model,
        test_loader,
        args.output_dir,
        device,
        args,
        args.num_samples if hasattr(args, "num_samples") else 16,
    )
    print("Testing completed!")


if __name__ == "__main__":
    main()
