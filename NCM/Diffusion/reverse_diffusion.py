import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import math
import json
from scipy.io import loadmat, savemat
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
            if "1C" in file_name:
                battery_type_str = "1C"
            elif "2C" in file_name:
                battery_type_str = "2C"
            elif "3C" in file_name:
                battery_type_str = "3C"
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
            print(f"Found file: {file_name} (samples: {data_mat['GAF_cell'].shape[0]})")
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
    print(
        f"PSNR analysis plot saved to: {os.path.join(output_dir, 'psnr_analysis.png')}"
    )


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
    print(f"PSNR classification saved to: {output_file}")
    print("\nPSNR Classification Summary:")
    for category_name, segments in categories.items():
        category_display = {
            "PSNR_50_plus": "PSNR >= 50dB",
            "PSNR_40_to_50": "40dB <= PSNR < 50dB",
            "PSNR_30_to_40": "30dB <= PSNR < 40dB",
            "PSNR_below_30": "PSNR < 30dB",
        }
        print(f"  {category_display[category_name]}: {len(segments)} segments")


def test_and_save_gaf(
    diffusion_model, test_loader, output_dir, device, args, orig_files, num_samples=16
):
    from dataset import BatteryCAFDataset

    train_dataset = BatteryCAFDataset(args.data_dir, split="train", transform=None)
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
                info_list = batch["info"]
                sample_info = info_list[i]
                battery_id_val = int(sample_info[0])
                cycle_idx = int(sample_info[1]) - 1
                file_source_val = batch["file_source"][i]
                segment_analysis.append(
                    {
                        "psnr": metrics["psnr"],
                        "cond_vec": (
                            cond_vec[i].cpu().numpy() if cond_vec is not None else None
                        ),
                        "battery_id": battery_id_val,
                        "cycle_idx": cycle_idx + 1,
                    }
                )
                if len(sample_inputs) < num_samples:
                    sample_inputs.append(inputs[i].cpu().numpy())
                    sample_targets.append(target)
                    sample_generated.append(gen)
                matched_key = None
                for k, v in orig_files.items():
                    if v["battery_id"] == battery_id_val and k.startswith(
                        file_source_val
                    ):
                        matched_key = k
                        break
                if matched_key is None:
                    print(
                        f"Warning: No matching battery found for battery_id={battery_id_val}"
                    )
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
    print("\nTest Results:")
    print(f"Average MSE: {avg_metrics['mse']:.4f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average SSIM: {avg_metrics['ssim']:.4f}")
    print(f"Best segments selected: {best_segment_count}")
    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Average MSE: {avg_metrics['mse']:.4f}\n")
        f.write(f"Average PSNR: {avg_metrics['psnr']:.2f} dB\n")
        f.write(f"Average SSIM: {avg_metrics['ssim']:.4f}\n")
        f.write(f"Best segments selected: {best_segment_count}\n")
    print(f"\n=== Segment PSNR and condition-vector analysis ===")
    if segment_analysis:
        save_psnr_classification(segment_analysis, output_dir, train_dataset)
        plot_psnr_analysis(segment_analysis, output_dir)
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
                print(
                    f"Warning: No data for {dict_key} cycle {idx + 1}, filled with zeros"
                )
        gaf_list = gaf_list[:n_required]
        gaf_cell = np.empty(original_info["gaf_shape"], dtype=object)
        for idx in range(n_required):
            gaf_cell[idx, 0] = gaf_list[idx]
        out_dict = {k: v for k, v in original_info["data"].items() if k != "GAF_cell"}
        out_dict["GAF_cell"] = gaf_cell
        out_name = f"generated_{original_info['filename']}"
        out_path = os.path.join(output_dir, out_name)
        savemat(out_path, out_dict)
        print(f"Saved: {out_path}  Samples: {n_required}")
        saved_files += 1
    print(f"\nBest PSNR statistics per battery:")
    for dict_key in orig_files:
        best_psnrs = cycle_best_psnr[dict_key]
        valid_psnrs = [p for p in best_psnrs if p > 0]
        if valid_psnrs:
            print(
                f"{dict_key}: Mean={np.mean(valid_psnrs):.2f}dB, Max={np.max(valid_psnrs):.2f}dB, Min={np.min(valid_psnrs):.2f}dB"
            )
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
        condition_emb_dim=64,
        use_attention=(False, False, True),
        num_res_blocks=2,
        cond_vector_dim=getattr(args, "cond_vector_dim", 26),
    ).to(args.device)
    ema_network = UNet(
        in_channels=1,
        out_channels=1,
        condition_channels=1,
        model_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        condition_emb_dim=64,
        use_attention=(False, False, True),
        num_res_blocks=2,
        cond_vector_dim=getattr(args, "cond_vector_dim", 26),
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
    parser.add_argument("--timesteps", type=int, default=400)
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--use_ddim", action="store_true", default=True)
    parser.add_argument("--ddim_steps", type=int, default=30)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--cosine_s", type=float, default=0.008)
    parser.add_argument("--ema", type=float, default=0.99995)
    parser.add_argument("--cond_vector_dim", type=int, default=26)
    _apply_config_defaults(parser, _load_config(base_args.config))
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    device = torch.device(args.device)
    print(f"[test_and_generate] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    _, _, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    diffusion_model = create_model(args)
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        best_model_path = os.path.join(args.checkpoint_dir, "model_epoch_130.pt")
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
    print("Test and generation completed!")


if __name__ == "__main__":
    main()
