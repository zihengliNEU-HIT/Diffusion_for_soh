import os
import math
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.io import savemat
from windowed_dataset import BatteryCAFDataset, custom_collate_fn
from gaussian_diffusion import GaussianDiffusion
from unet import UNet
from diffusion_model import DiffusionModel
from visualization import save_comparison_grid


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        psnr = 100.0
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


def parse_file_key(file_name):
    if "battery" in file_name:
        battery_id_str = file_name.split("battery")[1].split("_")[0]
    else:
        parts = file_name.split("_")
        battery_id_str = "1"
        for part in parts:
            if part.isdigit():
                battery_id_str = part
                break
    if "53C_54" in file_name:
        file_source = "53C_54"
    elif "56C_19" in file_name:
        file_source = "56C_19"
    elif "56C_36" in file_name:
        file_source = "56C_36"
    else:
        file_source = "unknown"
    return (f"{file_source}battery{battery_id_str}", file_source, int(battery_id_str))


def get_original_files(data_dir, split):
    pattern = f"_{split}_sliding.mat"
    original_dict = {}
    for file_name in os.listdir(data_dir):
        if pattern in file_name:
            file_path = os.path.join(data_dir, file_name)
            dict_key, file_source, battery_id = parse_file_key(file_name)
            original_dict[dict_key] = {
                "filename": file_name,
                "file_path": file_path,
                "file_source": file_source,
                "battery_id": battery_id,
            }
            print(f"Found file: {file_name}")
    return original_dict


def plot_psnr_analysis(segment_analysis, output_dir):
    if not segment_analysis:
        return
    psnr_values = [s["psnr"] for s in segment_analysis]
    psnr_array = np.array(psnr_values)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Window-level PSNR Distribution Analysis", fontsize=16, fontweight="bold"
    )
    axes[0, 0].hist(psnr_array, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0, 0].axvline(
        np.mean(psnr_array),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(psnr_array):.2f} dB",
    )
    axes[0, 0].axvline(
        np.median(psnr_array),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(psnr_array):.2f} dB",
    )
    axes[0, 0].set_xlabel("PSNR (dB)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("PSNR Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].boxplot(
        psnr_array, patch_artist=True, boxprops=dict(facecolor="lightblue")
    )
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].set_title("PSNR Box Plot")
    axes[0, 1].grid(True, alpha=0.3)
    if segment_analysis[0]["cond_vec"] is not None:
        cond_vecs = np.array([s["cond_vec"] for s in segment_analysis])
        n_dims = min(4, cond_vecs.shape[1])
        for dim in range(n_dims):
            axes[1, 0].scatter(
                cond_vecs[:, dim], psnr_array, alpha=0.6, s=10, label=f"Dim {dim}"
            )
        axes[1, 0].set_xlabel("Condition Vector Value")
        axes[1, 0].set_ylabel("PSNR (dB)")
        axes[1, 0].set_title("PSNR vs Condition Vectors")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        if cond_vecs.shape[1] > 1:
            correlations = []
            for dim in range(cond_vecs.shape[1]):
                corr = np.corrcoef(cond_vecs[:, dim], psnr_array)[0, 1]
                correlations.append(corr)
            x_pos = np.arange(len(correlations))
            bars = axes[1, 1].bar(x_pos, correlations, color="orange", alpha=0.7)
            axes[1, 1].set_xlabel("Condition Vector Dimension")
            axes[1, 1].set_ylabel("Correlation with PSNR")
            axes[1, 1].set_title("PSNR-CondVec Correlation")
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].grid(True, alpha=0.3)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{correlations[i]:.3f}",
                    ha="center",
                    va="bottom",
                )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No condition vector data",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 1].text(
            0.5,
            0.5,
            "No condition vector data",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "psnr_analysis_fullwin.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PSNR analysis plot saved to: {save_path}")


def plot_psnr_uncertainty(segment_analysis, output_dir):
    if not segment_analysis:
        return
    if "psnr_samples" not in segment_analysis[0]:
        print("No repeated sampling data; skip uncertainty plot")
        return
    from collections import defaultdict

    groups = defaultdict(list)
    for s in segment_analysis:
        groups[s["file_key"]].append(s)
    n_files = len(groups)
    fig, axes = plt.subplots(1, n_files, figsize=(6 * n_files, 5), sharey=True)
    if n_files == 1:
        axes = [axes]
    for ax, (file_key, segments) in zip(axes, sorted(groups.items())):
        segments_sorted = sorted(
            segments, key=lambda x: (x["cycle_idx"], x["segment_idx"])
        )
        cycle_indices = [s["cycle_idx"] for s in segments_sorted]
        psnr_all = np.array([s["psnr_samples"] for s in segments_sorted])
        psnr_mean = np.mean(psnr_all, axis=1)
        psnr_median = np.median(psnr_all, axis=1)
        psnr_q25 = np.percentile(psnr_all, 25, axis=1)
        psnr_q75 = np.percentile(psnr_all, 75, axis=1)
        psnr_min = np.min(psnr_all, axis=1)
        psnr_max = np.max(psnr_all, axis=1)
        for k in range(psnr_all.shape[1]):
            ax.scatter(
                cycle_indices,
                psnr_all[:, k],
                color="steelblue",
                alpha=0.12,
                s=8,
                zorder=1,
            )
        ax.fill_between(
            cycle_indices,
            psnr_min,
            psnr_max,
            color="gray",
            alpha=0.18,
            label="Min-Max",
            zorder=2,
        )
        ax.fill_between(
            cycle_indices,
            psnr_q25,
            psnr_q75,
            color="gray",
            alpha=0.32,
            label="IQR",
            zorder=3,
        )
        ax.plot(
            cycle_indices,
            psnr_mean,
            color="crimson",
            linewidth=1.8,
            label="Mean",
            zorder=4,
        )
        ax.plot(
            cycle_indices,
            psnr_median,
            color="navy",
            linewidth=1.8,
            linestyle="--",
            label="Median",
            zorder=5,
        )
        ax.set_title(file_key, fontsize=11)
        ax.set_xlabel("Cycle Index", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("PSNR (dB)", fontsize=10)
            ax.legend(fontsize=9)
    fig.suptitle("Window-level PSNR Uncertainty across Samples", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "psnr_uncertainty_fullwin.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PSNR uncertainty plot saved: {save_path}")


def save_psnr_classification(segment_analysis, output_dir):
    if not segment_analysis:
        return
    categories = {
        "PSNR_50_plus": [],
        "PSNR_40_to_50": [],
        "PSNR_30_to_40": [],
        "PSNR_below_30": [],
    }
    for segment in segment_analysis:
        psnr = segment["psnr"]
        if psnr >= 50:
            categories["PSNR_50_plus"].append(segment)
        elif psnr >= 40:
            categories["PSNR_40_to_50"].append(segment)
        elif psnr >= 30:
            categories["PSNR_30_to_40"].append(segment)
        else:
            categories["PSNR_below_30"].append(segment)
    output_file = os.path.join(output_dir, "psnr_classification_fullwin.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Window-level PSNR Classification Analysis\n")
        f.write("=" * 70 + "\n\n")
        for category_name, segments in categories.items():
            category_display = {
                "PSNR_50_plus": "PSNR >= 50 dB",
                "PSNR_40_to_50": "40 dB <= PSNR < 50 dB",
                "PSNR_30_to_40": "30 dB <= PSNR < 40 dB",
                "PSNR_below_30": "PSNR < 30 dB",
            }
            f.write(f"Category: {category_display[category_name]}\n")
            f.write(f"Count: {len(segments)}\n")
            f.write("-" * 40 + "\n")
            if not segments:
                f.write("No samples in this category\n\n")
                continue
            for i, segment in enumerate(segments, 1):
                f.write(f"Sample {i}:\n")
                f.write(f"  File Key       : {segment['file_key']}\n")
                f.write(f"  PSNR           : {segment['psnr']:.2f} dB\n")
                f.write(
                    f"  Battery / Cycle: battery{segment['battery_id']}_cycle{segment['cycle_idx']}\n"
                )
                f.write(f"  Segment Idx    : {segment['segment_idx']}\n")
                if segment["cond_vec"] is not None:
                    f.write(f"  CondVec(raw)   : {segment['cond_vec']}\n")
                else:
                    f.write("  CondVec(raw)   : None\n")
                f.write("\n")
            f.write("\n")
    print(f"PSNR classification saved to: {output_file}")
    print("\nPSNR Classification Summary:")
    for category_name, segments in categories.items():
        category_display = {
            "PSNR_50_plus": "PSNR >= 50 dB",
            "PSNR_40_to_50": "40 dB <= PSNR < 50 dB",
            "PSNR_30_to_40": "30 dB <= PSNR < 40 dB",
            "PSNR_below_30": "PSNR < 30 dB",
        }
        print(f"  {category_display[category_name]}: {len(segments)} samples")


def test_and_save_gaf(
    diffusion_model, test_loader, output_dir, device, args, orig_files, num_samples=16
):
    diffusion_model.ema_network.eval()
    os.makedirs(output_dir, exist_ok=True)
    per_file_data = {}
    for dict_key in orig_files:
        per_file_data[dict_key] = {
            "gaf_gen_list": [],
            "cond_vec_list": [],
            "info_list": [],
            "segment_idx_list": [],
        }
    all_metrics = []
    sample_inputs = []
    sample_targets = []
    sample_generated = []
    segment_analysis = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            cond_vec = batch.get("cond_vec", None)
            if cond_vec is not None and torch.is_tensor(cond_vec):
                cond_vec = cond_vec.to(device)
            cond_vec_raw = batch.get("cond_vec_raw", None)
            if cond_vec_raw is not None and torch.is_tensor(cond_vec_raw):
                cond_vec_raw = cond_vec_raw.cpu().numpy()
            n_samples = int(args.num_samples)
            generated_list = []
            for _ in range(n_samples):
                gen_k = diffusion_model.generate_samples(
                    inputs,
                    device=device,
                    cond_vec=cond_vec,
                    use_ddim=args.use_ddim,
                    ddim_steps=args.ddim_steps,
                    ddim_eta=args.ddim_eta,
                )
                generated_list.append(gen_k.cpu().numpy())
            generated_stack = np.stack(generated_list, axis=0)
            generated_mean = torch.tensor(np.mean(generated_stack, axis=0)).to(device)
            for i in range(inputs.shape[0]):
                target = targets[i, 0].cpu().numpy()
                gen_mean_i = generated_mean[i, 0].cpu().numpy()
                metrics = calculate_metrics(target, gen_mean_i)
                all_metrics.append(metrics)
                info_i = batch["info"][i]
                battery_id_val = int(info_i[0])
                cycle_idx = int(info_i[1])
                segment_idx = int(batch["segment_idx"][i].item())
                file_source_val = batch["file_source"][i]
                file_key = f"{file_source_val}battery{battery_id_val}"
                if file_key not in per_file_data:
                    print(
                        f"Warning: No matching battery file found for file_key={file_key}"
                    )
                    continue
                psnr_list = []
                for k in range(n_samples):
                    gen_k = generated_stack[k, i, 0]
                    m_k = calculate_metrics(target, gen_k)
                    psnr_list.append(m_k["psnr"])
                cond_raw_i = (
                    cond_vec_raw[i].astype(np.float32)
                    if cond_vec_raw is not None
                    else None
                )
                segment_analysis.append(
                    {
                        "file_key": file_key,
                        "psnr": metrics["psnr"],
                        "psnr_samples": psnr_list,
                        "cond_vec": cond_raw_i,
                        "battery_id": battery_id_val,
                        "cycle_idx": cycle_idx,
                        "segment_idx": segment_idx,
                        "file_source": file_source_val,
                    }
                )
                if len(sample_inputs) < num_samples:
                    sample_inputs.append(inputs[i].cpu().numpy())
                    sample_targets.append(target)
                    sample_generated.append(gen_mean_i)
                gen_20 = (generated_stack[:, i, 0] + 1.0) / 2.0
                per_file_data[file_key]["gaf_gen_list"].append(
                    gen_20.astype(np.float32)
                )
                per_file_data[file_key]["cond_vec_list"].append(cond_raw_i)
                per_file_data[file_key]["info_list"].append([battery_id_val, cycle_idx])
                per_file_data[file_key]["segment_idx_list"].append([segment_idx])
    avg_metrics = {
        "mse": np.mean([m["mse"] for m in all_metrics]) if all_metrics else np.nan,
        "psnr": np.mean([m["psnr"] for m in all_metrics]) if all_metrics else np.nan,
        "ssim": np.mean([m["ssim"] for m in all_metrics]) if all_metrics else np.nan,
    }
    print("\nTest Results:")
    print(f"Average MSE : {avg_metrics['mse']:.6f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average SSIM: {avg_metrics['ssim']:.4f}")
    print(
        f"Total window samples saved: {sum((len(v['gaf_gen_list']) for v in per_file_data.values()))}"
    )
    with open(os.path.join(output_dir, "test_metrics_fullwin.txt"), "w") as f:
        f.write(f"Average MSE: {avg_metrics['mse']:.6f}\n")
        f.write(f"Average PSNR: {avg_metrics['psnr']:.2f} dB\n")
        f.write(f"Average SSIM: {avg_metrics['ssim']:.4f}\n")
        f.write(
            f"Total window samples saved: {sum((len(v['gaf_gen_list']) for v in per_file_data.values()))}\n"
        )
    print("\n=== Window-level PSNR and condition-vector analysis ===")
    if segment_analysis:
        save_psnr_classification(segment_analysis, output_dir)
        plot_psnr_analysis(segment_analysis, output_dir)
        plot_psnr_uncertainty(segment_analysis, output_dir)
    if sample_inputs:
        save_comparison_grid(
            sample_inputs,
            sample_targets,
            sample_generated,
            os.path.join(output_dir, "test_samples_fullwin.png"),
            num_samples=min(num_samples, len(sample_inputs)),
        )
    saved_files = 0
    for dict_key, file_data in per_file_data.items():
        if len(file_data["gaf_gen_list"]) == 0:
            print(
                f"Warning: No valid window samples collected for {dict_key}, skip saving."
            )
            continue
        info_arr = np.array(file_data["info_list"], dtype=np.int32)
        seg_arr = np.array(file_data["segment_idx_list"], dtype=np.int32).reshape(-1, 1)
        sort_order = np.lexsort((seg_arr[:, 0], info_arr[:, 1]))
        gaf_gen_array = np.stack(file_data["gaf_gen_list"], axis=0)[sort_order]
        cond_vec_array = np.stack(file_data["cond_vec_list"], axis=0)[sort_order]
        info_arr = info_arr[sort_order]
        seg_arr = seg_arr[sort_order]
        original_info = orig_files[dict_key]
        out_dict = {
            "GAF_gen": gaf_gen_array.astype(np.float32),
            "COND_VEC": cond_vec_array.astype(np.float32),
            "INFO": info_arr.astype(np.int32),
            "SEGMENT_IDX": seg_arr.astype(np.int32),
        }
        base_name = original_info["filename"].replace("_sliding.mat", "")
        out_name = f"{base_name}_20gaf.mat"
        out_path = os.path.join(output_dir, out_name)
        savemat(out_path, out_dict)
        print(
            f"Saved: {out_path}  GAF_gen shape={gaf_gen_array.shape}, COND_VEC shape={cond_vec_array.shape}, INFO shape={info_arr.shape}, SEGMENT_IDX shape={seg_arr.shape}"
        )
        saved_files += 1
    print(f"\nGeneration completed! Total files saved: {saved_files}")
    return avg_metrics


def create_model(args):
    network = UNet(
        in_channels=1,
        out_channels=1,
        condition_channels=1,
        model_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        condition_emb_dim=32,
        use_attention=(False, False, True),
        num_res_blocks=2,
        cond_vector_dim=getattr(args, "cond_vector_dim", 24),
    ).to(args.device)
    ema_network = UNet(
        in_channels=1,
        out_channels=1,
        condition_channels=1,
        model_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        condition_emb_dim=32,
        use_attention=(False, False, True),
        num_res_blocks=2,
        cond_vector_dim=getattr(args, "cond_vector_dim", 24),
    ).to(args.device)
    gdf_util = GaussianDiffusion(
        timesteps=args.timesteps,
        device=args.device,
        schedule_type=args.noise_schedule,
        cosine_s=args.cosine_s,
    )
    diffusion_model = DiffusionModel(
        network=network,
        ema_network=ema_network,
        gdf_util=gdf_util,
        timesteps=args.timesteps,
        ema=args.ema,
    ).to(args.device)
    return diffusion_model


def _load_config(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_config_defaults(parser, config):
    for action in parser._actions:
        if action.dest in config:
            action.default = config[action.dest]


def parse_args():
    import argparse

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config", type=str, default=None, help="Path to a JSON configuration file."
    )
    base_args, _ = base_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        description="Selected-window reverse diffusion generation.",
        parents=[base_parser],
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./generated_gaf/train")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--use_ddim", action="store_true", default=True)
    parser.add_argument("--ddim_steps", type=int, default=30)
    parser.add_argument("--ddim_eta", type=float, default=1)
    parser.add_argument("--cosine_s", type=float, default=0.008)
    parser.add_argument("--ema", type=float, default=0.99995)
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--val_window_mode",
        type=str,
        default="random",
        choices=["random", "middle", "low", "high"],
    )
    parser.add_argument(
        "--test_window_mode",
        type=str,
        default="middle",
        choices=["random", "middle", "low", "high"],
    )
    parser.add_argument("--window_seed", type=int, default=42)
    parser.add_argument("--cond_vector_dim", type=int, default=24)
    _apply_config_defaults(parser, _load_config(base_args.config))
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    device = torch.device(args.device)
    print(f"[test_and_generate_selected_window] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    cond_mean = np.load(os.path.join(args.checkpoint_dir, "cond_mean.npy"))
    cond_std = np.load(os.path.join(args.checkpoint_dir, "cond_std.npy"))
    dataset = BatteryCAFDataset(
        args.data_dir,
        split=args.split,
        transform=None,
        cond_mean=cond_mean,
        cond_std=cond_std,
        val_window_mode=args.val_window_mode,
        test_window_mode=args.test_window_mode,
        random_seed=args.window_seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    diffusion_model = create_model(args)
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        best_model_path = os.path.join(args.checkpoint_dir, "model_epoch_120.pt")
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
        else:
            raise ValueError(
                "No checkpoint_path was provided and model_epoch_120.pt was not found."
            )
    load_checkpoint(diffusion_model, checkpoint_path, device)
    orig_files = get_original_files(args.data_dir, args.split)
    if not orig_files:
        raise ValueError(f"No original {args.split} files found")
    test_metrics = test_and_save_gaf(
        diffusion_model,
        loader,
        args.output_dir,
        device,
        args,
        orig_files,
        args.num_samples,
    )
    print("Test and generation completed!")
    return test_metrics


if __name__ == "__main__":
    main()
