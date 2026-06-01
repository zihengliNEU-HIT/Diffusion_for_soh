"""
reverse_diffusion.py
Content : Load trained diffusion model, run N-sample DDIM ensemble on the test split,
          select best-PSNR fragment per cycle, save generated GAF_cell back to .mat,
          plot per-battery PSNR uncertainty curves.
Run     : python reverse_diffusion.py --checkpoint_path ./checkpoints/model_best.pt
          python reverse_diffusion.py --checkpoint_path ./checkpoints/model_epoch_60.pt --ddim_steps 20 --n_samples 50
"""

import os
import math
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat, savemat

from dataset   import BatteryCAFDataset, custom_collate_fn
from gaussian_diffusion import GaussianDiffusion
from unet           import UNet
from diffusion_model import DiffusionModel
from visualization  import save_comparison_grid


# -- model factory -------------------------------------------------------------

def _load_config_defaults(config_path):
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_back_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = _load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description="LFP reverse diffusion GAF generation")
    parser.add_argument("--config", type=str, default=pre_args.config, help="Path to a JSON configuration file.")
    parser.add_argument("--data_dir", type=str, default="./data_for_generated")
    parser.add_argument("--output_dir", type=str, default="./generated_data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--cosine_s", type=float, default=0.008)
    parser.add_argument("--ema", type=float, default=0.99995)
    parser.add_argument("--cond_vector_dim", type=int, default=24)
    parser.add_argument("--use_ddim", action="store_true", default=True)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.set_defaults(**defaults)
    return parser.parse_args(remaining)


def create_model(args):
    def _unet():
        return UNet(
            in_channels=1, out_channels=1, condition_channels=1,
            model_channels=32, channel_mults=(1, 2, 4),
            time_emb_dim=128, condition_emb_dim=64,
            use_attention=(False, False, True), num_res_blocks=2,
            cond_vector_dim=getattr(args, "cond_vector_dim", 24),
        ).to(args.device)

    network     = _unet()
    ema_network = _unet()
    gdf_util    = GaussianDiffusion(
        timesteps=args.timesteps, device=args.device,
        schedule_type=args.noise_schedule, cosine_s=args.cosine_s,
    )
    return DiffusionModel(network, ema_network, gdf_util, args.timesteps, args.ema).to(args.device)


def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location=device)
    model.network.load_state_dict(ck["model_state_dict"])
    model.ema_network.load_state_dict(ck["ema_model_state_dict"])
    print(f"Loaded epoch={ck['epoch']}  val_loss={ck['val_loss']:.4f}")


# -- metrics -------------------------------------------------------------------

def calculate_metrics(target, generated):
    tgt = ((target    + 1.0) / 2.0).astype(np.float32)
    gen = ((generated + 1.0) / 2.0).astype(np.float32)
    mse = mean_squared_error(tgt.flatten(), gen.flatten())
    psnr = (20 * math.log10(1.0) - 10 * math.log10(mse)) if mse > 0 else 100.0
    tm, gm = np.mean(tgt), np.mean(gen)
    tv, gv = np.var(tgt),  np.var(gen)
    cov    = np.mean((tgt - tm) * (gen - gm))
    C1, C2 = 0.01**2, 0.03**2
    ssim = ((2*tm*gm+C1)/(tm**2+gm**2+C1) *
            (2*np.sqrt(tv)*np.sqrt(gv)+C2)/(tv+gv+C2) *
            (cov+C2/2)/(np.sqrt(tv)*np.sqrt(gv)+C2/2))
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


# -- file scanner --------------------------------------------------------------

def get_original_files(data_dir, split):
    """Scan data_dir for *_{split}_sliding.mat and return a keyed dict."""
    pat  = f"_{split}_sliding.mat"
    orig = {}
    for fn in os.listdir(data_dir):
        if pat not in fn:
            continue
        fp   = os.path.join(data_dir, fn)
        data = loadmat(fp)

        bid_str = fn.split("battery")[1].split("_")[0] if "battery" in fn else "1"
        for tag in ("53C_54", "56C_19", "56C_36"):
            if tag in fn:
                btype = tag; break
        else:
            btype = "unknown"

        key = f"{btype}battery{bid_str}"
        orig[key] = {
            "filename":   fn,
            "gaf_shape":  data["GAF_cell"].shape,
            "num_slices": data["GAF_cell"].shape[0],
            "battery_id": int(bid_str),
            "data":       data,
        }
        print(f"Found: {fn}  ({data['GAF_cell'].shape[0]} cycles)")
    return orig


# -- PSNR uncertainty plot -----------------------------------------------------

def plot_psnr_uncertainty(segment_analysis, output_dir, orig_files):
    """Per-battery subplot: scatter + min/max band + IQR band + mean + median."""
    if not segment_analysis or "psnr_samples" not in segment_analysis[0]:
        return

    groups = defaultdict(list)
    for s in segment_analysis:
        groups[f"{s['file_source']}battery{s['battery_id']}"].append(s)

    n    = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (bkey, segs) in zip(axes, sorted(groups.items())):
        segs  = sorted(segs, key=lambda x: x["cycle_idx"])
        cidx  = [s["cycle_idx"] for s in segs]
        pall  = np.array([s["psnr_samples"] for s in segs])   # (N_cycles, K)
        mean  = np.mean(pall,  axis=1)
        med   = np.median(pall, axis=1)
        q25   = np.percentile(pall, 25, axis=1)
        q75   = np.percentile(pall, 75, axis=1)
        pmin  = np.min(pall, axis=1)
        pmax  = np.max(pall, axis=1)

        # scatter all K samples
        for k in range(pall.shape[1]):
            ax.scatter(cidx, pall[:, k], color="steelblue", alpha=0.15, s=8, zorder=1)
        ax.fill_between(cidx, pmin, pmax,  color="gray", alpha=0.20, label="Min-Max", zorder=2)
        ax.fill_between(cidx, q25,  q75,   color="gray", alpha=0.35, label="IQR",     zorder=3)
        ax.plot(cidx, mean, color="crimson", lw=2, label="Mean",   zorder=4)
        ax.plot(cidx, med,  color="navy",    lw=2, ls="--", label="Median", zorder=5)
        ax.set_title(bkey, fontsize=11)
        ax.set_xlabel("Cycle Index", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("PSNR (dB)", fontsize=10)
            ax.legend(fontsize=9)

    fig.suptitle("Window-level PSNR Uncertainty across Samples", fontsize=13)
    plt.tight_layout()
    out = os.path.join(output_dir, "psnr_uncertainty.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PSNR uncertainty plot -> {out}")


# -- PSNR distribution analysis ------------------------------------------------

def plot_psnr_analysis(segment_analysis, output_dir):
    psnrs = np.array([s["psnr"] for s in segment_analysis])
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("PSNR Distribution Analysis", fontsize=16)

    axes[0,0].hist(psnrs, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0,0].axvline(np.mean(psnrs),   color="red",   ls="--", label=f"Mean {np.mean(psnrs):.2f}dB")
    axes[0,0].axvline(np.median(psnrs), color="green", ls="--", label=f"Median {np.median(psnrs):.2f}dB")
    axes[0,0].set_title("PSNR Distribution"); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].boxplot(psnrs, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axes[0,1].set_title("PSNR Box Plot"); axes[0,1].grid(True, alpha=0.3)

    if segment_analysis[0]["cond_vec"] is not None:
        cvecs  = np.array([s["cond_vec"] for s in segment_analysis])
        n_dims = min(4, cvecs.shape[1])
        for d in range(n_dims):
            axes[1,0].scatter(cvecs[:,d], psnrs, alpha=0.6, s=10, label=f"Dim {d}")
        axes[1,0].set_title("PSNR vs CondVec"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

        corrs = [np.corrcoef(cvecs[:,d], psnrs)[0,1] for d in range(cvecs.shape[1])]
        bars  = axes[1,1].bar(range(len(corrs)), corrs, color="orange", alpha=0.7)
        for bar, c in zip(bars, corrs):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{c:.3f}",
                           ha="center", va="bottom")
        axes[1,1].set_title("PSNR-CondVec Correlation"); axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "psnr_analysis.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PSNR analysis -> {out}")


# -- PSNR classification text report ------------------------------------------

def save_psnr_classification(segment_analysis, output_dir, cond_mean, cond_std):
    buckets = {">=50dB":[], "40-50dB":[], "30-40dB":[], "<30dB":[]}
    for s in segment_analysis:
        p = s["psnr"]
        if p >= 50: buckets[">=50dB"].append(s)
        elif p >= 40: buckets["40-50dB"].append(s)
        elif p >= 30: buckets["30-40dB"].append(s)
        else: buckets["<30dB"].append(s)

    out = os.path.join(output_dir, "psnr_classification.txt")
    with open(out, "w") as f:
        f.write("PSNR Classification\n" + "="*50 + "\n\n")
        for cat, segs in buckets.items():
            f.write(f"{cat}: {len(segs)} segments\n" + "-"*30 + "\n")
            for s in segs:
                f.write(f"  battery{s['battery_id']}_cycle{s['cycle_idx']}  PSNR={s['psnr']:.2f}dB\n")
                if s["cond_vec"] is not None:
                    orig = s["cond_vec"] * cond_std + cond_mean
                    f.write(f"  OrigCondVec: {orig}\n")
            f.write("\n")
    print(f"PSNR classification -> {out}")


# -- main generation loop ------------------------------------------------------

def test_and_save_gaf(diffusion_model, test_loader, output_dir, device, args, orig_files, num_samples=16):
    """
    For each batch:
      1. Run N DDIM samples, average -> mean GAF.
      2. Track best-PSNR segment per cycle for mat export.
      3. Collect per-sample PSNR list for uncertainty plots.
    """
    cond_mean = np.load(os.path.join(args.checkpoint_dir, "cond_mean.npy"))
    cond_std  = np.load(os.path.join(args.checkpoint_dir, "cond_std.npy"))

    diffusion_model.ema_network.eval()
    os.makedirs(output_dir, exist_ok=True)

    battery_gafs    = {k: [None] * v["num_slices"] for k, v in orig_files.items()}
    cycle_best_psnr = {k: [0.0]  * v["num_slices"] for k, v in orig_files.items()}

    all_metrics      = []
    vis_inputs       = []
    vis_targets      = []
    vis_generated    = []
    segment_analysis = []
    best_seg_count   = 0
    n_samples        = args.n_samples

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating"):
            inputs   = batch["input"].to(device)
            targets  = batch["target"].to(device)
            cond_vec = batch.get("cond_vec", None)
            if cond_vec is not None:
                cond_vec = cond_vec.to(device)

            # N-sample ensemble
            gen_list = []
            for _ in range(n_samples):
                g = diffusion_model.generate_samples(
                    inputs, device=device, cond_vec=cond_vec,
                    use_ddim=args.use_ddim, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
                )
                gen_list.append(g.cpu().numpy())

            gen_stack = np.stack(gen_list, axis=0)           # (K, B, 1, H, W)
            generated = torch.tensor(np.mean(gen_stack, axis=0)).to(device)

            for i in range(inputs.shape[0]):
                tgt = targets[i, 0].cpu().numpy()
                gen = generated[i, 0].cpu().numpy()
                m   = calculate_metrics(tgt, gen)
                all_metrics.append(m)

                # per-sample PSNR for uncertainty
                psnr_list = [calculate_metrics(tgt, gen_stack[k, i, 0])["psnr"]
                             for k in range(n_samples)]

                info       = batch["info"][i]
                battery_id = int(info[0])
                cycle_idx  = int(info[1]) - 1   # 0-based
                fsrc       = batch["file_source"][i]

                segment_analysis.append({
                    "psnr":         m["psnr"],
                    "psnr_samples": psnr_list,
                    "cond_vec":     cond_vec[i].cpu().numpy() if cond_vec is not None else None,
                    "battery_id":   battery_id,
                    "cycle_idx":    cycle_idx + 1,
                    "file_source":  fsrc,
                })

                if len(vis_inputs) < num_samples:
                    vis_inputs.append(inputs[i].cpu().numpy())
                    vis_targets.append(tgt)
                    vis_generated.append(gen)

                # match to orig_files key
                matched = next((k for k, v in orig_files.items()
                                if v["battery_id"] == battery_id and k.startswith(fsrc)), None)
                if matched is None:
                    print(f"[Warning] No match for battery_id={battery_id}  src={fsrc}")
                    continue

                # keep best-PSNR segment
                if m["psnr"] > cycle_best_psnr[matched][cycle_idx]:
                    cycle_best_psnr[matched][cycle_idx] = m["psnr"]
                    battery_gafs[matched][cycle_idx]    = ((gen + 1) / 2).astype(np.float32)
                    best_seg_count += 1

    # summary metrics
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print(f"\nResults  MSE={avg['mse']:.4f}  PSNR={avg['psnr']:.2f}dB  SSIM={avg['ssim']:.4f}  "
          f"best_segs={best_seg_count}")

    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
        f.write(f"MSE={avg['mse']:.4f}\nPSNR={avg['psnr']:.2f}dB\nSSIM={avg['ssim']:.4f}\n")

    # analysis plots
    plot_psnr_analysis(segment_analysis, output_dir)
    plot_psnr_uncertainty(segment_analysis, output_dir, orig_files)
    save_psnr_classification(segment_analysis, output_dir, cond_mean, cond_std)

    # visualisation grid
    if vis_inputs:
        save_comparison_grid(vis_inputs, vis_targets, vis_generated,
                             os.path.join(output_dir, "test_samples.png"),
                             num_samples=min(num_samples, len(vis_inputs)))

    # export .mat files
    saved = 0
    for key, gaf_list in battery_gafs.items():
        info     = orig_files[key]
        n_req    = info["num_slices"]
        gaf_list = [(g if g is not None else np.zeros((128, 128), dtype=np.float32))
                    for g in gaf_list[:n_req]]
        gaf_cell = np.empty(info["gaf_shape"], dtype=object)
        for idx, g in enumerate(gaf_list):
            gaf_cell[idx, 0] = g
        out_dict = {k: v for k, v in info["data"].items() if k != "GAF_cell"}
        out_dict["GAF_cell"] = gaf_cell
        out_path = os.path.join(output_dir, f"generated_{info['filename']}")
        savemat(out_path, out_dict)
        print(f"Saved: {out_path}  ({n_req} cycles)")
        saved += 1

    # per-battery PSNR stats
    for key in orig_files:
        ps = [p for p in cycle_best_psnr[key] if p > 0]
        if ps:
            print(f"{key}: mean={np.mean(ps):.2f}dB  max={np.max(ps):.2f}dB  min={np.min(ps):.2f}dB")

    print(f"Done. Files saved: {saved}")
    return avg


# -- entry point ---------------------------------------------------------------

def main(args=None):
    if args is None:
        args = parse_back_args()

    device = torch.device(args.device)
    print(f"[reverse_diffusion] device={device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # load normalisation stats from training checkpoint dir
    cond_mean = np.load(os.path.join(args.checkpoint_dir, "cond_mean.npy"))
    cond_std  = np.load(os.path.join(args.checkpoint_dir, "cond_std.npy"))

    test_dataset = BatteryCAFDataset(
        args.data_dir, split="test", transform=None,
        cond_mean=cond_mean, cond_std=cond_std,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
    )

    model = create_model(args)

    ckpt = args.checkpoint_path or os.path.join(args.checkpoint_dir, "model_epoch_60.pt")
    load_checkpoint(model, ckpt, device)

    orig_files = get_original_files(args.data_dir, "test")
    if not orig_files:
        raise ValueError("No test .mat files found in data_dir.")

    test_and_save_gaf(model, test_loader, args.output_dir, device, args,
                      orig_files, args.num_samples)
    print("Generation complete.")


if __name__ == "__main__":
    main()
