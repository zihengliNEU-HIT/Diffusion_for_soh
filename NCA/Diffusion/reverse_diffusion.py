import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import math
from scipy.io import loadmat, savemat
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


def get_original_files(data_dir, split):
    pattern = f"_{split}_sliding.mat"
    original_dict = {}
    for file_name in os.listdir(data_dir):
        if pattern in file_name:
            file_path = os.path.join(data_dir, file_name)
            data_mat = loadmat(file_path)
            if "battery" in file_name:
                battery_id_str = file_name.split("battery")[1].split("_")[0]
            else:
                parts = file_name.split("_")
                battery_id_str = "1"
                for part in parts:
                    if part.isdigit():
                        battery_id_str = part
                        break
            if "025C" in file_name:
                battery_type_str = "025C"
            elif "1C" in file_name:
                battery_type_str = "1C"
            elif "05C" in file_name:
                battery_type_str = "05C"
            else:
                battery_type_str = "unknown"
            dict_key = f"{battery_type_str}battery{battery_id_str}"
            original_dict[dict_key] = {
                "filename": file_name,
                "gaf_shape": data_mat["GAF_cell"].shape,
                "num_slices": data_mat["GAF_cell"].shape[0],
                "battery_id": int(battery_id_str),
                "data": data_mat,
            }
    return original_dict


def plot_psnr_analysis(segment_analysis, output_dir):
    if not segment_analysis:
        return
    psnr_values = [s["psnr"] for s in segment_analysis]
    psnr_array = np.array(psnr_values)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("PSNR Distribution Analysis", fontsize=16, fontweight="bold")
    axes[0, 0].hist(psnr_array, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0, 0].axvline(
        np.mean(psnr_array),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(psnr_array):.2f}dB",
    )
    axes[0, 0].axvline(
        np.median(psnr_array),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(psnr_array):.2f}dB",
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
    plt.savefig(
        os.path.join(output_dir, "psnr_analysis.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_psnr_uncertainty(segment_analysis, output_dir, orig_files):
    if not segment_analysis:
        return
    if "psnr_samples" not in segment_analysis[0]:
        return
    from collections import defaultdict

    groups = defaultdict(list)
    for s in segment_analysis:
        group_key = f"{s['file_source']}battery{s['battery_id']}"
        groups[group_key].append(s)
    n_batteries = len(groups)
    fig, axes = plt.subplots(1, n_batteries, figsize=(6 * n_batteries, 5), sharey=True)
    if n_batteries == 1:
        axes = [axes]
    id_to_name = {
        v["battery_id"]: v["filename"].replace("_test_sliding.mat", "")
        for v in orig_files.values()
    }
    for ax, (battery_id, segments) in zip(axes, sorted(groups.items())):
        segments_sorted = sorted(segments, key=lambda x: x["cycle_idx"])
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
                alpha=0.15,
                s=8,
                zorder=1,
            )
        ax.fill_between(
            cycle_indices,
            psnr_min,
            psnr_max,
            color="gray",
            alpha=0.2,
            label="Min-Max",
            zorder=2,
        )
        ax.fill_between(
            cycle_indices,
            psnr_q25,
            psnr_q75,
            color="gray",
            alpha=0.35,
            label="IQR",
            zorder=3,
        )
        ax.plot(
            cycle_indices,
            psnr_mean,
            color="crimson",
            linewidth=2,
            label="Mean",
            zorder=4,
        )
        ax.plot(
            cycle_indices,
            psnr_median,
            color="navy",
            linewidth=2,
            linestyle="--",
            label="Median",
            zorder=5,
        )
        battery_name = battery_id
        ax.set_title(battery_name, fontsize=11)
        ax.set_xlabel("Cycle Index", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("PSNR (dB)", fontsize=10)
            ax.legend(fontsize=9)
    fig.suptitle("PSNR Uncertainty across Cycles)", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "psnr_uncertainty.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_psnr_classification(segment_analysis, output_dir, train_dataset):
    if not segment_analysis:
        return
    cond_mean = train_dataset.cond_mean
    cond_std = train_dataset.cond_std
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
    output_file = os.path.join(output_dir, "psnr_classification.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("PSNR Classification Analysis\n")
        f.write("=" * 50 + "\n\n")
        for category_name, segments in categories.items():
            category_display = {
                "PSNR_50_plus": "PSNR >= 50dB",
                "PSNR_40_to_50": "40dB <= PSNR < 50dB",
                "PSNR_30_to_40": "30dB <= PSNR < 40dB",
                "PSNR_below_30": "PSNR < 30dB",
            }
            f.write(f"Category: {category_display[category_name]}\n")
            f.write(f"Count: {len(segments)}\n")
            f.write("-" * 30 + "\n")
            if not segments:
                f.write("No segments in this category\n\n")
                continue
            for i, segment in enumerate(segments, 1):
                f.write(f"Segment {i}:\n")
                f.write(f"  PSNR: {segment['psnr']:.2f}dB\n")
                f.write(
                    f"  Battery Cycle: battery{segment['battery_id']}_cycle{segment['cycle_idx']}\n"
                )
                if segment["cond_vec"] is not None:
                    normalized_cond = segment["cond_vec"]
                    original_cond = normalized_cond * cond_std + cond_mean
                    f.write(f"  Normalized CondVec: {normalized_cond}\n")
                    f.write(f"  Original CondVec: {original_cond}\n")
                else:
                    f.write("  CondVec: None\n")
                f.write("\n")
            f.write("\n")
    for category_name, segments in categories.items():
        category_display = {
            "PSNR_50_plus": "PSNR >= 50dB",
            "PSNR_40_to_50": "40dB <= PSNR < 50dB",
            "PSNR_30_to_40": "30dB <= PSNR < 40dB",
            "PSNR_below_30": "PSNR < 30dB",
        }


def test_and_save_gaf(
    diffusion_model, test_loader, output_dir, device, args, orig_files, num_samples=16
):
    cond_mean = np.load(os.path.join(args.checkpoint_dir, "cond_mean.npy"))
    cond_std = np.load(os.path.join(args.checkpoint_dir, "cond_std.npy"))
    train_dataset = type(
        "obj", (object,), {"cond_mean": cond_mean, "cond_std": cond_std}
    )()
    diffusion_model.ema_network.eval()
    os.makedirs(output_dir, exist_ok=True)
    battery_gafs = {}
    cycle_best_psnr = {}
    for dict_key in orig_files:
        battery_gafs[dict_key] = [None] * orig_files[dict_key]["num_slices"]
        cycle_best_psnr[dict_key] = [0.0] * orig_files[dict_key]["num_slices"]
    all_metrics = []
    sample_inputs = []
    sample_targets = []
    sample_generated = []
    best_segment_count = 0
    segment_analysis = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            cond_vec = batch.get("cond_vec", None)
            if cond_vec is not None and torch.is_tensor(cond_vec):
                cond_vec = cond_vec.to(device)
            n_samples = 20
            generated_list = []
            for k in range(n_samples):
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
            generated = torch.tensor(np.mean(generated_stack, axis=0)).to(device)
            for i in range(inputs.shape[0]):
                target = targets[i, 0].cpu().numpy()
                gen = generated[i, 0].cpu().numpy()
                metrics = calculate_metrics(target, gen)
                all_metrics.append(metrics)
                info_list = batch["info"]
                sample_info = info_list[i]
                battery_id_val = int(sample_info[0])
                cycle_idx = int(sample_info[1]) - 1
                psnr_list = []
                for k in range(n_samples):
                    gen_k = generated_stack[k, i, 0]
                    m_k = calculate_metrics(target, gen_k)
                    psnr_list.append(m_k["psnr"])
                segment_analysis.append(
                    {
                        "psnr": metrics["psnr"],
                        "psnr_samples": psnr_list,
                        "cond_vec": (
                            cond_vec[i].cpu().numpy() if cond_vec is not None else None
                        ),
                        "battery_id": battery_id_val,
                        "cycle_idx": cycle_idx + 1,
                        "file_source": batch["file_source"][i],
                    }
                )
                if len(sample_inputs) < num_samples:
                    sample_inputs.append(inputs[i].cpu().numpy())
                    sample_targets.append(target)
                    sample_generated.append(gen)
                matched_key = None
                file_source_val = batch["file_source"][i]
                for k, v in orig_files.items():
                    if v["battery_id"] == battery_id_val and k.startswith(
                        file_source_val
                    ):
                        matched_key = k
                        break
                if matched_key is None:
                    continue
                current_psnr = metrics["psnr"]
                if current_psnr > cycle_best_psnr[matched_key][cycle_idx]:
                    cycle_best_psnr[matched_key][cycle_idx] = current_psnr
                    gen_normalized = (gen + 1) / 2
                    battery_gafs[matched_key][cycle_idx] = gen_normalized.astype(
                        np.float32
                    )
                    best_segment_count += 1
    avg_metrics = {
        "mse": np.mean([m["mse"] for m in all_metrics]),
        "psnr": np.mean([m["psnr"] for m in all_metrics]),
        "ssim": np.mean([m["ssim"] for m in all_metrics]),
    }
    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Average MSE: {avg_metrics['mse']:.4f}\n")
        f.write(f"Average PSNR: {avg_metrics['psnr']:.2f} dB\n")
        f.write(f"Average SSIM: {avg_metrics['ssim']:.4f}\n")
        f.write(f"Best segments selected: {best_segment_count}\n")
    if segment_analysis:
        save_psnr_classification(segment_analysis, output_dir, train_dataset)
        plot_psnr_analysis(segment_analysis, output_dir)
        plot_psnr_uncertainty(segment_analysis, output_dir, orig_files)
    if sample_inputs:
        save_comparison_grid(
            sample_inputs,
            sample_targets,
            sample_generated,
            os.path.join(output_dir, "test_samples.png"),
            num_samples=min(num_samples, len(sample_inputs)),
        )
    saved_files = 0
    for dict_key, gaf_list in battery_gafs.items():
        original_info = orig_files[dict_key]
        n_required = original_info["gaf_shape"][0]
        for idx in range(len(gaf_list)):
            if gaf_list[idx] is None:
                gaf_list[idx] = np.zeros((128, 128), dtype=np.float32)
        gaf_list = gaf_list[:n_required]
        gaf_cell = np.empty(original_info["gaf_shape"], dtype=object)
        for idx in range(n_required):
            gaf_cell[idx, 0] = gaf_list[idx]
        out_dict = {k: v for k, v in original_info["data"].items() if k != "GAF_cell"}
        out_dict["GAF_cell"] = gaf_cell
        out_name = f"generated_{original_info['filename']}"
        out_path = os.path.join(output_dir, out_name)
        savemat(out_path, out_dict)
        saved_files += 1
    for dict_key in orig_files:
        best_psnrs = cycle_best_psnr[dict_key]
        valid_psnrs = [p for p in best_psnrs if p > 0]
        if valid_psnrs:
            pass
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
        cond_vector_dim=getattr(args, "cond_vector_dim", 22),
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
        cond_vector_dim=getattr(args, "cond_vector_dim", 22),
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


def parse_args():
    import argparse

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config", type=str, default=None, help="Path to a JSON configuration file."
    )
    base_args, _ = base_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        description="Reverse diffusion generation.", parents=[base_parser]
    )
    parser.add_argument("--data_dir", type=str, default="./data_for_generated")
    parser.add_argument("--output_dir", type=str, default="./generated_data")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--use_ddim", action="store_true", default=True)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ddim_eta", type=float, default=1)
    parser.add_argument("--cosine_s", type=float, default=0.008)
    parser.add_argument("--ema", type=float, default=0.99995)
    parser.add_argument("--cond_vector_dim", type=int, default=22)
    if base_args.config is not None:
        with open(base_args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        for action in parser._actions:
            if action.dest in config:
                action.default = config[action.dest]
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    from dataset import BatteryCAFDataset, custom_collate_fn

    cond_mean = np.load(os.path.join(args.checkpoint_dir, "cond_mean.npy"))
    cond_std = np.load(os.path.join(args.checkpoint_dir, "cond_std.npy"))
    test_dataset = BatteryCAFDataset(
        args.data_dir,
        split="test",
        transform=None,
        cond_mean=cond_mean,
        cond_std=cond_std,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
    )
    diffusion_model = create_model(args)
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        best_model_path = os.path.join(args.checkpoint_dir, "model_epoch_175.pt")
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
        else:
            raise ValueError(
                "No checkpoint_path was provided and no default checkpoint was found."
            )
    load_checkpoint(diffusion_model, checkpoint_path, device)
    orig_files = get_original_files(args.data_dir, "test")
    if not orig_files:
        raise ValueError("No original test files found")
    test_metrics = test_and_save_gaf(
        diffusion_model,
        test_loader,
        args.output_dir,
        device,
        args,
        orig_files,
        args.num_samples,
    )


if __name__ == "__main__":
    main()
