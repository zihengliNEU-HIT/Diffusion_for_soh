"""Checkpoint I/O, evaluation metrics, and visualization utilities."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from collections import defaultdict


# Checkpoint

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer else None,
        'val_loss':        val_loss,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and ckpt.get('optimizer_state'):
        optimizer.load_state_dict(ckpt['optimizer_state'])
    epoch    = ckpt.get('epoch', 0)
    val_loss = ckpt.get('val_loss', float('inf'))
    print(f"Loaded checkpoint: epoch={epoch}, val_loss={val_loss:.5f}")
    return epoch, val_loss


# Denormalization

def denorm_soh(x_norm, soh_min, soh_max):
    return np.asarray(x_norm) * (soh_max - soh_min) + soh_min


# Metrics

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    eps  = 1e-8
    err  = pred - gt
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    mape = np.mean(np.abs(err) / (np.abs(gt) + eps)) * 100.0
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((gt - gt.mean()) ** 2)
    r2   = 1.0 - ss_res / (ss_tot + eps)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE(%)': mape, 'R2': r2}


# Loss curve

def plot_loss_curve(train_losses, val_losses, eval_interval, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='Train Loss', color='steelblue')
    if val_losses:
        val_epochs = [(i + 1) * eval_interval for i in range(len(val_losses))]
        ax.plot(val_epochs, val_losses, label='Val Loss',
                color='tomato', marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('SOH Estimator Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Loss curve -> {save_path}")


# Regression plot

def plot_regression(gt, pred, std, bids, save_path):
    unique_ids = sorted(np.unique(bids))
    cmap       = plt.get_cmap('tab20', len(unique_ids))
    id2color   = {bid: cmap(i) for i, bid in enumerate(unique_ids)}

    fig, ax = plt.subplots(figsize=(6, 6))
    for bid in unique_ids:
        mask = bids == bid
        ax.errorbar(gt[mask], pred[mask], yerr=std[mask],
                    fmt='o', markersize=4, alpha=0.7,
                    color=id2color[bid], ecolor=id2color[bid],
                    elinewidth=0.8, capsize=2, label=f'Battery {bid}')

    lo = min(gt.min(), pred.min()) * 0.98
    hi = max(gt.max(), pred.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2, label='Ideal')

    metrics = compute_metrics(gt, pred)
    info = (f"MAE={metrics['MAE']:.4f} Ah\n"
            f"RMSE={metrics['RMSE']:.4f} Ah\n"
            f"MAPE={metrics['MAPE(%)']:.2f}%\n"
            f"R2={metrics['R2']:.4f}")
    ax.text(0.04, 0.96, info, transform=ax.transAxes,
            fontsize=9, va='top', bbox=dict(boxstyle='round', alpha=0.15))

    ax.set_xlabel('True Capacity (Ah)')
    ax.set_ylabel('Predicted Capacity (Ah)')
    ax.set_title('SOH Regression - Test Set')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Regression plot -> {save_path}")


# Trajectory plot

def plot_trajectory_per_battery(bids, cycle_idxs, gt, pred, std, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for bid in sorted(np.unique(bids)):
        mask  = bids == bid
        cyc   = cycle_idxs[mask]
        order = np.argsort(cyc)
        cyc   = cyc[order]
        g, p, s = gt[mask][order], pred[mask][order], std[mask][order]

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(cyc, g, 'k-',  linewidth=1.5, label='True')
        ax.plot(cyc, p, 'r-',  linewidth=1.5, label='Pred Mean')
        ax.fill_between(cyc, p - s, p + s, color='red', alpha=0.2, label='+/-1 std')

        metrics = compute_metrics(g, p)
        ax.set_title(f'Battery {bid} - MAE={metrics["MAE"]:.4f} Ah  '
                     f'RMSE={metrics["RMSE"]:.4f} Ah  R2={metrics["R2"]:.4f}')
        ax.set_xlabel('Cycle Index')
        ax.set_ylabel('Capacity (Ah)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fpath = os.path.join(output_dir, f'trajectory_battery{bid}.png')
        plt.savefig(fpath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Trajectory -> {fpath}")


# Spatial attention visualization

def _attn_to_heatmap(attn_map: np.ndarray, img_size: int = 128) -> np.ndarray:
    """
    Convert an (8, 8) attention map to a normalized heatmap.

    Returns an array with shape (img_size, img_size) and values in [0, 1].
    """
    a = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float()  # (1,1,8,8)
    a = F.interpolate(a, size=(img_size, img_size), mode='bilinear',
                      align_corners=False)
    a = a.squeeze().numpy()
    a_min, a_max = a.min(), a.max()
    if a_max - a_min > 1e-8:
        a = (a - a_min) / (a_max - a_min)
    return a


def save_attention_maps(
    gaf_imgs: torch.Tensor,
    attn_maps: np.ndarray,
    battery_id: int,
    cycle_idx: int,
    soh_gt: float,
    soh_pred_mean: float,
    save_dir: str,
    img_size: int = 128,
) -> None:
    """
    gaf_imgs  : (K, 1, H, W)  Tensor with values in [0, 1]
    attn_maps : (K, 8, 8)     ndarray

    Each GAF is shown as one column: the first row is the original image, and
    the second row is the attention heatmap overlay. At most five images are
    displayed to keep the figure compact.
    """
    os.makedirs(save_dir, exist_ok=True)

    K = gaf_imgs.shape[0]
    show = min(K, 5)

    fig, axes = plt.subplots(2, show, figsize=(3 * show, 6))
    if show == 1:
        axes = axes[:, np.newaxis]

    colormap = cm.get_cmap('jet')

    for j in range(show):
        gaf_np  = gaf_imgs[j, 0].numpy()               # (H,W) [0,1]
        heatmap = _attn_to_heatmap(attn_maps[j], img_size)  # (H,W) [0,1]

        # Top row: original GAF.
        axes[0, j].imshow(gaf_np, cmap='gray', vmin=0, vmax=1)
        axes[0, j].set_title(f'GAF  #{j+1}', fontsize=9)
        axes[0, j].axis('off')

        # Bottom row: GAF with attention overlay.
        axes[1, j].imshow(gaf_np, cmap='gray', vmin=0, vmax=1)
        overlay = axes[1, j].imshow(heatmap, cmap='jet',
                                    alpha=0.45, vmin=0, vmax=1)
        axes[1, j].set_title(f'Attn #{j+1}', fontsize=9)
        axes[1, j].axis('off')

    # colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.35])
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Attention')

    fig.suptitle(
        f'Battery {battery_id}  Cycle {cycle_idx}\n'
        f'GT={soh_gt:.4f} Ah   Pred={soh_pred_mean:.4f} Ah',
        fontsize=10, y=1.01
    )
    plt.tight_layout(rect=[0, 0, 0.91, 1])

    fname = f'battery{battery_id}_cycle{cycle_idx:04d}_attn.png'
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Attn map -> {fpath}")


def save_attention_comparison(
    gaf_real: np.ndarray,
    attn_real: np.ndarray,
    gaf_gen_list: list,
    attn_gen_list: list,
    labels: list,
    battery_id: int,
    cycle_idx: int,
    soh_gt: float,
    save_dir: str,
    img_size: int = 128,
) -> None:
    """
    Compare attention maps for the real GAF and multiple generated GAF images.

    gaf_real      : (H, W)        ndarray [0, 1]
    attn_real     : (8, 8)        ndarray
    gaf_gen_list  : list of (H, W) ndarray [0, 1]
    attn_gen_list : list of (8, 8) ndarray
    labels        : list of str, for example ['low', 'mid', 'high']

    Layout: one column per image group. The first row shows GAF images, and the
    second row shows attention overlays.
    """
    os.makedirs(save_dir, exist_ok=True)

    all_gafs  = [gaf_real]  + list(gaf_gen_list)
    all_attns = [attn_real] + list(attn_gen_list)
    all_labels = ['Real'] + list(labels)
    ncols = len(all_gafs)

    fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6))

    for j, (gaf, attn, lbl) in enumerate(zip(all_gafs, all_attns, all_labels)):
        heatmap = _attn_to_heatmap(attn, img_size)

        axes[0, j].imshow(gaf, cmap='gray', vmin=0, vmax=1)
        axes[0, j].set_title(lbl, fontsize=10,
                             color='black' if lbl == 'Real' else 'steelblue')
        axes[0, j].axis('off')

        axes[1, j].imshow(gaf, cmap='gray', vmin=0, vmax=1)
        axes[1, j].imshow(heatmap, cmap='jet', alpha=0.45, vmin=0, vmax=1)
        axes[1, j].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.35])
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Attention')

    fig.suptitle(
        f'Attention Comparison  Battery {battery_id}  Cycle {cycle_idx}\n'
        f'SOH GT = {soh_gt:.4f} Ah',
        fontsize=10
    )
    plt.tight_layout(rect=[0, 0, 0.91, 1])

    fname = f'battery{battery_id}_cycle{cycle_idx:04d}_attn_compare.png'
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Attn compare -> {fpath}")
