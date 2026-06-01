import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import math
import matplotlib

matplotlib.use("Agg")
from matplotlib.ticker import FuncFormatter
import matplotlib.text as mtext
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
from dataset import get_dataloaders, visualize_sample
from gaussian_diffusion import GaussianDiffusion
from unet import UNet
from diffusion_model import DiffusionModel
from visualization import save_training_samples


def create_scheduler(optimizer, args, val_loader_size):
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-06
        )
    elif args.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-06,
            cooldown=0,
        )
    elif args.scheduler == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(args, "step_size", 10),
            gamma=getattr(args, "step_gamma", 0.5),
        )
    else:
        return None


def calculate_psnr(generated, target):
    gen_imgs = (generated + 1.0) / 2.0
    target_imgs = (target + 1.0) / 2.0
    mse = torch.mean((gen_imgs - target_imgs) ** 2)
    if mse > 0:
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    else:
        return 100.0


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    val_loss,
    checkpoint_dir,
    psnr_value=None,
    is_best=False,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.network.state_dict(),
        "ema_model_state_dict": model.ema_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "psnr": psnr_value,
    }
    if scheduler is not None:
        checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
    if psnr_value is not None and psnr_value > 38.0:
        high_quality_dir = os.path.join(checkpoint_dir, "high_quality_models")
        os.makedirs(high_quality_dir, exist_ok=True)
        psnr_checkpoint_path = os.path.join(
            high_quality_dir, f"model_epoch_{epoch}_psnr_{psnr_value:.2f}.pt"
        )
        torch.save(checkpoint_data, psnr_checkpoint_path)
        print(
            f"High-quality model saved: PSNR={psnr_value:.2f}dB -> {psnr_checkpoint_path}"
        )
        log_file = os.path.join(high_quality_dir, "high_quality_log.txt")
        with open(log_file, "a") as f:
            f.write(
                f"Epoch {epoch}: PSNR={psnr_value:.2f}dB, Val_Loss={val_loss:.6f}\n"
            )
    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pt")
        torch.save(checkpoint_data, best_path)
        print(f"Best model saved to {best_path}")
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
    torch.save(checkpoint_data, checkpoint_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} does not exist.")
        return (0, float("inf"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.network.load_state_dict(checkpoint["model_state_dict"])
    model.ema_network.load_state_dict(checkpoint["ema_model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    print(f"Loaded checkpoint from epoch {epoch} with validation loss {val_loss:.4f}")
    return (epoch, val_loss)


def plot_losses_with_psnr(
    train_losses, val_losses, psnr_history, output_dir, learning_rates=None
):
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "text.usetex": False,
            "axes.unicode_minus": False,
        }
    ):
        plt.figure(figsize=(15, 10))

        def _sanitize_figure(fig):
            for txt in fig.findobj(mtext.Text):
                s = txt.get_text()
                if s:
                    s = s.replace(chr(0x2212), "-")
                    txt.set_text(s)

            def _fmt(x, pos):
                s = f"{float(x):g}"
                return s.replace(chr(0x2212), "-")

            for ax in fig.axes:
                ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
                ax.yaxis.set_major_formatter(FuncFormatter(_fmt))

        plt.subplot(2, 2, 1)
        plt.plot([float(v) for v in train_losses], label="Training Loss", color="blue")
        if val_losses:
            x_step = max(1, len(train_losses) // max(1, len(val_losses)))
            x_vals = list(range(0, len(train_losses), x_step))
            if len(x_vals) > len(val_losses):
                x_vals = x_vals[: len(val_losses)]
            elif len(x_vals) < len(val_losses):
                while len(x_vals) < len(val_losses):
                    x_vals.append(
                        x_vals[-1]
                        + (x_vals[1] - x_vals[0] if len(x_vals) > 1 else x_step)
                    )
            plt.plot(
                x_vals,
                [float(v) for v in val_losses],
                label="Validation Loss",
                color="orange",
            )
        plt.title("Training and Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 2, 2)
        if psnr_history:
            x_step = max(1, len(train_losses) // max(1, len(psnr_history)))
            psnr_x_vals = list(range(0, len(train_losses), x_step))
            if len(psnr_x_vals) > len(psnr_history):
                psnr_x_vals = psnr_x_vals[: len(psnr_history)]
            elif len(psnr_x_vals) < len(psnr_history):
                while len(psnr_x_vals) < len(psnr_history):
                    psnr_x_vals.append(
                        psnr_x_vals[-1]
                        + (
                            psnr_x_vals[1] - psnr_x_vals[0]
                            if len(psnr_x_vals) > 1
                            else x_step
                        )
                    )
            plt.plot(
                psnr_x_vals,
                [float(v) for v in psnr_history],
                label="PSNR",
                color="green",
                marker="o",
            )
            plt.axhline(
                y=38.0,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Target PSNR (38dB)",
            )
            plt.axhline(
                y=35.0,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label="Good PSNR (35dB)",
            )
            plt.axhline(
                y=30.0,
                color="yellow",
                linestyle="--",
                alpha=0.7,
                label="Decent PSNR (30dB)",
            )
        plt.title("PSNR Progress")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR (dB)")
        plt.legend()
        plt.grid(True)
        if learning_rates:
            plt.subplot(2, 2, 3)
            plt.plot(
                [float(v) for v in learning_rates],
                label="Learning Rate",
                color="purple",
            )
            plt.title("Learning Rate Schedule")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.legend()
            plt.grid(True)
            plt.yscale("log")
        plt.subplot(2, 2, 4)
        if psnr_history:
            psnr_vals = [float(v) for v in psnr_history]
            plt.hist(psnr_vals, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
            plt.axvline(
                x=38.0, color="red", linestyle="--", alpha=0.7, label="Target (38dB)"
            )
            mean_psnr = float(np.mean(psnr_vals))
            plt.axvline(
                x=mean_psnr,
                color="green",
                linestyle="-",
                alpha=0.7,
                label=f"Mean ({mean_psnr:.1f}dB)",
            )
            plt.title("PSNR Distribution")
            plt.xlabel("PSNR (dB)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
        fig = plt.gcf()
        _sanitize_figure(fig)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "training_metrics.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    else:
        return obj


def init_training_log(args):
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_name = datetime.now().strftime("train_%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_name}.json")
    log_data = {
        "run_name": run_name,
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "args": _to_json_safe(vars(args)),
        "epochs": [],
        "summary": {},
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    return (log_path, log_data)


def save_training_log(log_path, log_data):
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(log_data), f, ensure_ascii=False, indent=2)


def main(args=None):
    if args is None:
        from cs_main import parse_args

        args = parse_args()
    device = torch.device(args.device)
    print(f"[train] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_start_time = time.time()
    log_path, log_data = init_training_log(args)
    print(f"Training log will be saved to: {log_path}")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    from cs_main import create_model

    diffusion_model = create_model(args).to(device)
    optimizer = optim.AdamW(
        diffusion_model.network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = create_scheduler(optimizer, args, len(val_loader))
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "model_best.pt")
    if os.path.exists(best_checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(
            diffusion_model, optimizer, scheduler, best_checkpoint_path, device
        )
        start_epoch += 1
    else:
        checkpoint_files = [
            f
            for f in os.listdir(args.checkpoint_dir)
            if f.endswith(".pt") and f != "model_best.pt"
        ]
        if checkpoint_files:
            latest = sorted(
                checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            start_epoch, best_val_loss = load_checkpoint(
                diffusion_model,
                optimizer,
                scheduler,
                os.path.join(args.checkpoint_dir, latest),
                device,
            )
            start_epoch += 1
        else:
            start_epoch = 0
            best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    learning_rates = []
    psnr_history = []
    patience_counter = 0
    high_quality_count = 0
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        diffusion_model.network.train()
        epoch_loss = 0
        batch_count = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch in progress_bar:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            cond_vec = batch.get("cond_vec", None)
            if cond_vec is not None and torch.is_tensor(cond_vec):
                cond_vec = cond_vec.to(device)
            batch_size = targets.shape[0]
            t = torch.randint(
                0, diffusion_model.timesteps, (batch_size,), device=device
            ).long()
            model_fn = lambda x, tt, c: diffusion_model.network(
                x, tt, c, cond_vec=cond_vec
            )
            loss = diffusion_model.gdf_util.noise_estimation_loss(
                model_fn, targets, t, condition=inputs
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                diffusion_model.network.parameters(), args.clip_grad
            )
            optimizer.step()
            diffusion_model.update_ema()
            epoch_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({"loss": loss.item()})
        train_loss = epoch_loss / batch_count
        train_losses.append(train_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)
        print(f"Train Loss: {train_loss:.4f}, Learning Rate: {current_lr:.6f}")
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "learning_rate": float(current_lr),
        }
        if (epoch + 1) % args.eval_interval == 0:
            diffusion_model.ema_network.eval()
            val_loss = 0
            psnr_values = []
            val_count = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(
                    tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}")
                ):
                    inputs = batch["input"].to(device)
                    targets = batch["target"].to(device)
                    cond_vec = batch.get("cond_vec", None)
                    if cond_vec is not None and torch.is_tensor(cond_vec):
                        cond_vec = cond_vec.to(device)
                    batch_size = targets.shape[0]
                    t = torch.linspace(
                        0, diffusion_model.timesteps - 1, batch_size, device=device
                    ).long()
                    ema_fn = lambda x, tt, c: diffusion_model.ema_network(
                        x, tt, c, cond_vec=cond_vec
                    )
                    loss = diffusion_model.gdf_util.noise_estimation_loss(
                        ema_fn, targets, t, condition=inputs
                    )
                    val_loss += loss.item()
                    val_count += 1
                    if batch_idx < 15:
                        generated = diffusion_model.generate_samples(
                            inputs, device=device, cond_vec=cond_vec
                        )
                        for i in range(min(32, batch_size)):
                            psnr = calculate_psnr(
                                generated[i : i + 1], targets[i : i + 1]
                            )
                            psnr_values.append(psnr)
                        if batch_idx == 0:
                            avg_batch_psnr = sum(
                                psnr_values[-min(8, batch_size) :]
                            ) / min(8, batch_size)
                            if avg_batch_psnr > 38.0:
                                save_training_samples(
                                    inputs.cpu().numpy(),
                                    targets.cpu().numpy(),
                                    generated.cpu().numpy(),
                                    os.path.join(
                                        args.output_dir,
                                        f"high_quality_samples_epoch_{epoch + 1}_psnr_{avg_batch_psnr:.2f}.png",
                                    ),
                                    num_samples=min(4, inputs.shape[0]),
                                )
                                print(
                                    f"High-quality samples saved: PSNR={avg_batch_psnr:.2f}dB"
                                )
                            else:
                                save_training_samples(
                                    inputs.cpu().numpy(),
                                    targets.cpu().numpy(),
                                    generated.cpu().numpy(),
                                    os.path.join(
                                        args.output_dir,
                                        f"samples_epoch_{epoch + 1}.png",
                                    ),
                                    num_samples=min(4, inputs.shape[0]),
                                )
            val_loss = val_loss / val_count
            avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
            max_psnr = max(psnr_values) if psnr_values else 0
            val_losses.append(val_loss)
            psnr_history.append(avg_psnr)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"PSNR: Mean={avg_psnr:.2f}dB, Max={max_psnr:.2f}dB")
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}"
                )
            epoch_record.update(
                {
                    "val_loss": float(val_loss),
                    "avg_psnr": float(avg_psnr),
                    "max_psnr": float(max_psnr),
                    "is_best": bool(is_best),
                    "high_quality": bool(avg_psnr > 38.0),
                    "patience_counter": int(patience_counter),
                }
            )
            save_checkpoint(
                diffusion_model,
                optimizer,
                scheduler,
                epoch + 1,
                val_loss,
                args.checkpoint_dir,
                psnr_value=avg_psnr,
                is_best=is_best,
            )
            if avg_psnr > 38.0:
                high_quality_count += 1
                print(
                    f"Collected {high_quality_count} high-quality models (PSNR > 38dB)"
                )
            if args.scheduler == "plateau" and scheduler is not None:
                scheduler.step(avg_psnr)
            log_data["epochs"].append(epoch_record)
            best_epoch = None
            best_val = None
            best_psnr = None
            for item in log_data["epochs"]:
                if "val_loss" in item:
                    if best_val is None or item["val_loss"] < best_val:
                        best_val = item["val_loss"]
                        best_epoch = item["epoch"]
                        best_psnr = item.get("avg_psnr", None)
            log_data["summary"] = {
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "best_psnr_at_best_epoch": best_psnr,
                "high_quality_count": int(high_quality_count),
                "latest_epoch": int(epoch + 1),
            }
            save_training_log(log_path, log_data)
            if patience_counter >= args.patience:
                print(
                    f"Early stopping triggered; collected {high_quality_count} high-quality models"
                )
                break
        if (epoch + 1) % args.eval_interval != 0:
            log_data["epochs"].append(epoch_record)
            save_training_log(log_path, log_data)
        if scheduler is not None and args.scheduler != "plateau":
            scheduler.step()
        plot_losses_with_psnr(
            train_losses, val_losses, psnr_history, args.output_dir, learning_rates
        )
    save_checkpoint(
        diffusion_model,
        optimizer,
        scheduler,
        args.num_epochs,
        val_losses[-1] if val_losses else float("inf"),
        args.checkpoint_dir,
    )
    plot_losses_with_psnr(
        train_losses, val_losses, psnr_history, args.output_dir, learning_rates
    )
    run_end_time = time.time()
    log_data["end_time"] = datetime.now().isoformat(timespec="seconds")
    log_data["summary"]["total_time_sec"] = round(run_end_time - run_start_time, 2)
    log_data["summary"]["num_epochs_finished"] = len(log_data["epochs"])
    log_data["summary"]["final_train_loss"] = (
        float(train_losses[-1]) if train_losses else None
    )
    log_data["summary"]["final_val_loss"] = (
        float(val_losses[-1]) if val_losses else None
    )
    log_data["summary"]["max_psnr"] = float(max(psnr_history)) if psnr_history else None
    log_data["summary"]["mean_psnr"] = (
        float(np.mean(psnr_history)) if psnr_history else None
    )
    save_training_log(log_path, log_data)
    print(f"JSON log saved to: {log_path}")
    print("Training completed!")
    print(f"Collected {high_quality_count} high-quality models (PSNR > 38dB)")
    if psnr_history:
        print(f"Best PSNR: {max(psnr_history):.2f}dB")
        print(f"MeanPSNR: {np.mean(psnr_history):.2f}dB")


if __name__ == "__main__":
    main()
