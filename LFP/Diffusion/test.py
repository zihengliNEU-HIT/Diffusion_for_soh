"""
test.py
Content : Evaluate diffusion model on test split - MSE, PSNR, simplified SSIM,
          save comparison grid.
Run     : python cs_main.py --mode test --checkpoint_path ./checkpoints/model_best.pt
          (or standalone) python test.py --checkpoint_path ./checkpoints/model_best.pt
"""

import os
import math
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from dataset  import get_dataloaders
from visualization import save_comparison_grid


# -- metrics -------------------------------------------------------------------

def calculate_metrics(target, generated):
    """Return dict with mse, psnr, ssim.  Inputs are in [-1,1] numpy arrays."""
    tgt = ((target    + 1.0) / 2.0).astype(np.float32)
    gen = ((generated + 1.0) / 2.0).astype(np.float32)

    mse  = mean_squared_error(tgt.flatten(), gen.flatten())
    psnr = (20 * math.log10(1.0) - 10 * math.log10(mse)) if mse > 0 else 100.0

    tm, gm   = np.mean(tgt), np.mean(gen)
    tv, gv   = np.var(tgt),  np.var(gen)
    cov      = np.mean((tgt - tm) * (gen - gm))
    C1, C2   = (0.01) ** 2, (0.03) ** 2
    lum  = (2 * tm * gm + C1) / (tm**2 + gm**2 + C1)
    con  = (2 * np.sqrt(tv) * np.sqrt(gv) + C2) / (tv + gv + C2)
    stru = (cov + C2 / 2) / (np.sqrt(tv) * np.sqrt(gv) + C2 / 2)
    ssim = lum * con * stru

    return {"mse": mse, "psnr": psnr, "ssim": ssim}


# -- checkpoint loader ---------------------------------------------------------

def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location=device)
    model.network.load_state_dict(ck["model_state_dict"])
    model.ema_network.load_state_dict(ck["ema_model_state_dict"])
    print(f"Loaded epoch={ck['epoch']}  val_loss={ck['val_loss']:.4f}")
    return ck["epoch"], ck["val_loss"]


# -- test loop -----------------------------------------------------------------

def test_model(diffusion_model, test_loader, output_dir, device, args, num_samples=16):
    diffusion_model.ema_network.eval()
    os.makedirs(output_dir, exist_ok=True)

    all_metrics  = []
    vis_inputs   = []
    vis_targets  = []
    vis_generated= []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs   = batch["input"].to(device)
            targets  = batch["target"].to(device)
            cond_vec = batch.get("cond_vec", None)
            if cond_vec is not None:
                cond_vec = cond_vec.to(device)

            generated = diffusion_model.generate_samples(
                inputs, device=device, cond_vec=cond_vec,
                use_ddim=args.use_ddim, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
            )

            for i in range(inputs.shape[0]):
                tgt = targets[i, 0].cpu().numpy()
                gen = generated[i, 0].cpu().numpy()
                all_metrics.append(calculate_metrics(tgt, gen))

                if len(vis_inputs) < num_samples:
                    vis_inputs.append(inputs[i].cpu().numpy())
                    vis_targets.append(tgt)
                    vis_generated.append(gen)

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print(f"\nTest  MSE={avg['mse']:.4f}  PSNR={avg['psnr']:.2f}dB  SSIM={avg['ssim']:.4f}")

    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
        f.write(f"MSE={avg['mse']:.4f}\nPSNR={avg['psnr']:.2f}dB\nSSIM={avg['ssim']:.4f}\n")

    save_comparison_grid(vis_inputs, vis_targets, vis_generated,
                         os.path.join(output_dir, "test_samples.png"),
                         num_samples=min(num_samples, len(vis_inputs)))
    return avg


# -- entry point ---------------------------------------------------------------

def main(args=None):
    if args is None:
        from cs_main import parse_args
        args = parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size,
                                        num_workers=args.num_workers)

    from cs_main import create_model
    diffusion_model = create_model(args)

    ckpt = args.checkpoint_path or os.path.join(args.checkpoint_dir, "model_best.pt")
    if not os.path.exists(ckpt):
        raise ValueError(f"No checkpoint at {ckpt}")
    load_checkpoint(diffusion_model, ckpt, device)

    test_model(diffusion_model, test_loader, args.output_dir, device, args)
    print("Test done.")


if __name__ == "__main__":
    main()
