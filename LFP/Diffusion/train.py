"""
train.py
Content : Training loop - scheduler factory, checkpoint save/load,
          PSNR tracking, early stopping.
Run     : python cs_main.py --mode train --data_dir ./data --batch_size 16
          (or standalone) python train.py
"""

import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from dataset   import get_dataloaders
from visualization  import save_training_samples, plot_losses_with_psnr


# -- scheduler factory ---------------------------------------------------------

def create_scheduler(optimizer, args, val_loader_size):
    s = args.scheduler
    if s == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    elif s == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    elif s == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(args, "step_size", 10),
            gamma=getattr(args, "step_gamma", 0.5))
    return None


# -- metrics -------------------------------------------------------------------

def calculate_psnr(generated, target):
    gen = (generated + 1.0) / 2.0
    tgt = (target    + 1.0) / 2.0
    mse = torch.mean((gen - tgt) ** 2)
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item() if mse > 0 else 100.0


# -- checkpoint helpers --------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_dir,
                    psnr_value=None, is_best=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    data = {
        "epoch":                epoch,
        "model_state_dict":     model.network.state_dict(),
        "ema_model_state_dict": model.ema_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":             val_loss,
        "psnr":                 psnr_value,
    }
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()

    # save high-quality checkpoints separately
    if psnr_value is not None and psnr_value > 38.0:
        hq_dir = os.path.join(checkpoint_dir, "high_quality_models")
        os.makedirs(hq_dir, exist_ok=True)
        hq_path = os.path.join(hq_dir, f"model_epoch_{epoch}_psnr_{psnr_value:.2f}.pt")
        torch.save(data, hq_path)
        with open(os.path.join(hq_dir, "high_quality_log.txt"), "a") as f:
            f.write(f"Epoch {epoch}: PSNR={psnr_value:.2f}dB  Val={val_loss:.6f}\n")
        print(f"[HQ] PSNR={psnr_value:.2f}dB -> {hq_path}")

    if is_best:
        torch.save(data, os.path.join(checkpoint_dir, "model_best.pt"))
        print(f"Best model saved (epoch {epoch})")

    torch.save(data, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, float("inf")
    ck = torch.load(checkpoint_path, map_location=device)
    model.network.load_state_dict(ck["model_state_dict"])
    model.ema_network.load_state_dict(ck["ema_model_state_dict"])
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ck:
        scheduler.load_state_dict(ck["scheduler_state_dict"])
    print(f"Loaded checkpoint epoch={ck['epoch']}  val_loss={ck['val_loss']:.4f}")
    return ck["epoch"], ck["val_loss"]


# -- main training function ----------------------------------------------------

def main(args=None):
    if args is None:
        from cs_main import parse_args
        args = parse_args()

    device = torch.device(args.device)
    print(f"[train] device={device}")
    os.makedirs(args.output_dir,     exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        oversample_tail=getattr(args, "oversample_tail", False),
        oversample_tail_factor=getattr(args, "oversample_tail_factor", 3),
    )

    # save normalisation stats alongside checkpoints
    np.save(os.path.join(args.checkpoint_dir, "cond_mean.npy"), train_loader.dataset.cond_mean)
    np.save(os.path.join(args.checkpoint_dir, "cond_std.npy"),  train_loader.dataset.cond_std)

    from cs_main import create_model
    diffusion_model = create_model(args).to(device)

    optimizer = optim.AdamW(diffusion_model.network.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args, len(val_loader))

    # resume logic
    best_ckpt = os.path.join(args.checkpoint_dir, "model_best.pt")
    if os.path.exists(best_ckpt):
        start_epoch, best_val = load_checkpoint(
            diffusion_model, optimizer, scheduler, best_ckpt, device)
        start_epoch += 1
    else:
        pts = [f for f in os.listdir(args.checkpoint_dir)
               if f.startswith("model_epoch_") and f.endswith(".pt")]
        if pts:
            latest = sorted(pts, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            start_epoch, best_val = load_checkpoint(
                diffusion_model, optimizer, scheduler,
                os.path.join(args.checkpoint_dir, latest), device)
            start_epoch += 1
        else:
            start_epoch, best_val = 0, float("inf")

    train_losses, val_losses, psnr_history, lrs = [], [], [], []
    patience_counter = 0
    hq_count         = 0

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")

        # -- train -------------------------------------------------------------
        diffusion_model.network.train()
        epoch_loss, n = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            inputs   = batch["input"].to(device)
            targets  = batch["target"].to(device)
            cond_vec = batch.get("cond_vec", None)
            if cond_vec is not None:
                cond_vec = cond_vec.to(device)

            B = targets.shape[0]
            t = torch.randint(0, diffusion_model.timesteps, (B,), device=device).long()

            model_fn = lambda x, tt, c: diffusion_model.network(x, tt, c, cond_vec=cond_vec)
            loss = diffusion_model.gdf_util.noise_estimation_loss(
                model_fn, targets, t, condition=inputs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                diffusion_model.network.parameters(), args.clip_grad)
            optimizer.step()
            diffusion_model.update_ema()

            epoch_loss += loss.item()
            n          += 1
            # real-time display: current batch loss + running average + lr
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg":  f"{epoch_loss / n:.4f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        train_loss = epoch_loss / n
        train_losses.append(train_loss)
        lrs.append(optimizer.param_groups[0]["lr"])
        print(f"  Train  loss={train_loss:.4f}  lr={lrs[-1]:.2e}")

        # -- validate ----------------------------------------------------------
        if (epoch + 1) % args.eval_interval == 0:
            diffusion_model.ema_network.eval()
            val_loss, val_n = 0.0, 0
            psnr_vals       = []

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Val   {epoch+1}/{args.num_epochs}")
                for bidx, batch in enumerate(val_pbar):
                    inputs   = batch["input"].to(device)
                    targets  = batch["target"].to(device)
                    cond_vec = batch.get("cond_vec", None)
                    if cond_vec is not None:
                        cond_vec = cond_vec.to(device)

                    B = targets.shape[0]
                    t = torch.linspace(
                        0, diffusion_model.timesteps - 1, B, device=device).long()

                    ema_fn = lambda x, tt, c: diffusion_model.ema_network(
                        x, tt, c, cond_vec=cond_vec)
                    loss = diffusion_model.gdf_util.noise_estimation_loss(
                        ema_fn, targets, t, condition=inputs)

                    val_loss += loss.item()
                    val_n    += 1
                    # real-time val loss
                    val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

                    # PSNR on first 30 batches
                    if bidx < 30:
                        gen = diffusion_model.generate_samples(
                            inputs, device=device, cond_vec=cond_vec)
                        for i in range(min(32, B)):
                            psnr_vals.append(
                                calculate_psnr(gen[i:i+1], targets[i:i+1]))
                        # update bar with running PSNR
                        val_pbar.set_postfix({
                            "val_loss": f"{loss.item():.4f}",
                            "psnr":     f"{np.mean(psnr_vals):.2f}dB",
                        })

                        # save sample images on first batch
                        if bidx == 0:
                            avg_b = np.mean(psnr_vals[-min(8, B):])
                            tag   = "high_quality_" if avg_b > 38.0 else ""
                            save_training_samples(
                                inputs.cpu().numpy(),
                                targets.cpu().numpy(),
                                gen.cpu().numpy(),
                                os.path.join(args.output_dir,
                                             f"{tag}samples_epoch_{epoch+1}.png"),
                                num_samples=min(4, B),
                            )

            val_loss /= val_n
            avg_psnr  = np.mean(psnr_vals) if psnr_vals else 0.0
            max_psnr  = max(psnr_vals)      if psnr_vals else 0.0
            val_losses.append(val_loss)
            psnr_history.append(avg_psnr)
            print(f"  Val    loss={val_loss:.4f}  "
                  f"PSNR avg={avg_psnr:.2f}dB  max={max_psnr:.2f}dB")

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{args.patience})")

            save_checkpoint(diffusion_model, optimizer, scheduler,
                            epoch + 1, val_loss, args.checkpoint_dir,
                            psnr_value=avg_psnr, is_best=is_best)

            if avg_psnr > 38.0:
                hq_count += 1
                print(f"  HQ models collected: {hq_count}")

            if args.scheduler == "plateau" and scheduler is not None:
                scheduler.step(avg_psnr)

            if patience_counter >= args.patience:
                print(f"Early stopping. HQ models: {hq_count}")
                break

        if scheduler is not None and args.scheduler != "plateau":
            scheduler.step()

        plot_losses_with_psnr(
            train_losses, val_losses, psnr_history, args.output_dir, lrs)

    # final checkpoint
    save_checkpoint(
        diffusion_model, optimizer, scheduler,
        args.num_epochs,
        val_losses[-1] if val_losses else float("inf"),
        args.checkpoint_dir)

    print(f"\nTraining done.  HQ models: {hq_count}")
    if psnr_history:
        print(f"Best PSNR: {max(psnr_history):.2f}dB  "
              f"Avg PSNR: {np.mean(psnr_history):.2f}dB")


if __name__ == "__main__":
    main()