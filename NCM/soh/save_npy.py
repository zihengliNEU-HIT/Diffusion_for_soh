"""
save_npy.py
-----------
Run inference on each *_test_soh.mat file and save one .npy file per battery.
The output filename is derived from the MAT filename, for example
1Cbattery4.npy, which avoids collisions between repeated bat_idx values.

Each .npy file is a structured array with these fields:
    gt        : float32 - ground-truth capacity in Ah after denormalization
    pred      : float32 - mean prediction across K generated GAF images in Ah
    bat_idx   : float32 - original battery id from INFO[:, 0]
    cycle_idx : float32 - cycle index from INFO[:, 1]

Load example:
    arr = np.load('1Cbattery4.npy', allow_pickle=True)
    arr['gt'], arr['pred'], arr['cycle_idx']

Run:
    python save_npy.py
"""

import os
import re
import glob

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from SOH_config import SOHConfig
from SOH_model import build_model
from SOH_utils import load_checkpoint, compute_metrics, denorm_soh


# File discovery

def find_test_files(data_dir):
    """Return [(fpath, label), ...], where label removes the _test_soh.mat suffix."""
    found = []
    for fpath in sorted(glob.glob(os.path.join(data_dir, '*_test_soh.mat'))):
        basename = os.path.basename(fpath)
        label    = re.sub(r'_test_soh\.mat$', '', basename, flags=re.I)
        found.append((fpath, label))
    return found


# Single-file MAT loading

def load_one_file(fpath):
    """
    Return (gaf_gen, capacity, info), or None if the file is invalid.

      gaf_gen  : float32 [N, K, H, W]   values in [0, 1]
      capacity : float32 [N]
      info     : int32   [N, 2]
    """
    fname = os.path.basename(fpath)
    try:
        mat = sio.loadmat(fpath, variable_names=['GAF_gen', 'CAPACITY', 'INFO'])
    except Exception as e:
        print(f'  [Warning] Cannot load {fname}: {e}')
        return None

    if 'GAF_gen' not in mat or 'CAPACITY' not in mat or 'INFO' not in mat:
        print(f'  [Warning] {fname}: missing required field, skipping')
        return None

    gaf_gen  = mat['GAF_gen'].astype(np.float32)    # [N, K, H, W]
    capacity = mat['CAPACITY'].astype(np.float32).reshape(-1)
    info     = mat['INFO'].astype(np.int32)          # [N, 2]

    if gaf_gen.ndim != 4:
        print(f'  [Warning] {fname}: GAF_gen ndim={gaf_gen.ndim}, expected 4, skipping')
        return None
    if gaf_gen.shape[0] != capacity.shape[0] or gaf_gen.shape[0] != info.shape[0]:
        print(f'  [Warning] {fname}: row mismatch, skipping')
        return None

    return gaf_gen, capacity, info


# Lightweight dataset for a single MAT file.

class _SingleFileDataset(Dataset):
    """Return (gaf_gen_norm [K,1,H,W], soh_norm float, bat_idx, cycle_idx)."""
    def __init__(self, gaf_gen, capacity, info, soh_min, soh_max):
        # Normalize gaf_gen from [0, 1] to [-1, 1].
        self.gaf  = torch.from_numpy(gaf_gen).unsqueeze(2)   # [N,K,1,H,W]
        self.gaf  = self.gaf * 2.0 - 1.0

        soh_norm  = (capacity - soh_min) / max(soh_max - soh_min, 1e-8)
        self.soh  = torch.from_numpy(soh_norm.reshape(-1, 1))  # [N,1]
        self.info = info   # [N,2]

    def __len__(self):
        return len(self.soh)

    def __getitem__(self, idx):
        return (
            self.gaf[idx],           # [K,1,H,W]
            self.soh[idx],           # [1]
            int(self.info[idx, 0]),  # bat_idx
            int(self.info[idx, 1]),  # cycle_idx
        )


def _collate(batch):
    gaf  = torch.stack([b[0] for b in batch])   # [B,K,1,H,W]
    soh  = torch.stack([b[1] for b in batch])   # [B,1]
    bids = [b[2] for b in batch]
    cycs = [b[3] for b in batch]
    return gaf, soh, bids, cycs


# Inference

@torch.no_grad()
def predict_file(model, gaf_gen, capacity, info, soh_min, soh_max,
                 batch_size, device):
    """
    Return (gt_raw, pred_mean_raw, bat_idx_arr, cycle_idx_arr).

    All outputs are float32 ndarrays with length N.
    """
    model.eval()
    ds     = _SingleFileDataset(gaf_gen, capacity, info, soh_min, soh_max)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=_collate)

    gt_l, mean_l, bid_l, cyc_l = [], [], [], []

    for gaf, soh_gt, bids, cycs in loader:
        B, K, _, H, W = gaf.shape
        gaf_flat  = gaf.view(B * K, 1, H, W).to(device)    # (B*K,1,H,W)
        pred_flat = model(gaf_flat).cpu().numpy()            # (B*K,1)
        pred_bk   = pred_flat.reshape(B, K)                 # (B,K), normalized
        gt_norm   = soh_gt.numpy().reshape(-1)              # (B,)

        # Denormalize each prediction before averaging, matching SOH_test.py.
        for i in range(B):
            preds_raw = denorm_soh(pred_bk[i], soh_min, soh_max)  # (K,)
            mean_l.append(preds_raw.mean())

        gt_l.append(denorm_soh(gt_norm, soh_min, soh_max))
        bid_l.extend(bids)
        cyc_l.extend(cycs)

    return (
        np.concatenate(gt_l).astype(np.float32),
        np.array(mean_l, dtype=np.float32),
        np.array(bid_l, dtype=np.float32),
        np.array(cyc_l, dtype=np.float32),
    )


# NPY saving

def save_battery_npy(label, gt, pred_mean, bat_idx, cycle_idx, output_dir):
    dtype = np.dtype([
        ('gt',        np.float32),
        ('pred',      np.float32),
        ('bat_idx',   np.float32),
        ('cycle_idx', np.float32),
    ])
    arr = np.empty(len(gt), dtype=dtype)
    arr['gt']        = gt
    arr['pred']      = pred_mean
    arr['bat_idx']   = bat_idx
    arr['cycle_idx'] = cycle_idx

    out = os.path.join(output_dir, f'{label}.npy')
    np.save(out, arr)

    rmse = float(np.sqrt(np.mean((gt - pred_mean) ** 2)))
    print(f'  Saved  {label}.npy  ({len(gt)} cycles)  RMSE={rmse:.5f} Ah')


# Plotting

def _metrics(gt, pred):
    rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
    mae  = float(np.mean(np.abs(gt - pred)))
    r2   = 1 - np.sum((gt - pred) ** 2) / (np.sum((gt - gt.mean()) ** 2) + 1e-9)
    return rmse, mae, float(r2)


def plot_battery_regression(label, gt, pred, cycle_idx, cfg, output_dir):
    """Create one regression plot per battery, colored by cycle index."""
    rmse, mae, r2 = _metrics(gt, pred)

    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)

    sc = ax.scatter(gt, pred, c=cycle_idx, cmap='viridis',
                    s=20, alpha=0.75, linewidths=0)

    lo = min(gt.min(), pred.min())
    hi = max(gt.max(), pred.max())
    pad = 0.03 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1.1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.text(0.04, 0.96,
            f'RMSE={rmse:.4f} Ah\nMAE={mae:.4f} Ah\nR²={r2:.4f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

    ax.set_title(f'{label}  n={len(gt)} cycles', fontsize=10)
    ax.set_xlabel('True Capacity (Ah)')
    ax.set_ylabel('Predicted Capacity (Ah)')
    ax.grid(alpha=0.3)

    fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, label='Cycle Index')

    out = os.path.join(output_dir, f'{label}_regression.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot   {label}_regression.png')


def plot_battery_trajectory(label, gt, pred, cycle_idx, output_dir):
    """Create one degradation trajectory plot per battery."""
    order = np.argsort(cycle_idx)
    cyc   = cycle_idx[order]
    g, p  = gt[order], pred[order]

    rmse, mae, r2 = _metrics(g, p)

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    ax.plot(cyc, g, 'k-', lw=1.5, label='True')
    ax.plot(cyc, p, 'r-', lw=1.5, label='Pred')

    ax.set_title(f'{label}  RMSE={rmse:.4f} Ah  MAE={mae:.4f} Ah  R²={r2:.4f}',
                 fontsize=9)
    ax.set_xlabel('Cycle Index')
    ax.set_ylabel('Capacity (Ah)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out = os.path.join(output_dir, f'{label}_trajectory.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot   {label}_trajectory.png')


def plot_overall_grid(all_labels, all_gt, all_pred, all_cyc, cfg, output_dir):
    """Summarize all batteries in one regression grid."""
    n     = len(all_labels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 4.8 * nrows),
                              constrained_layout=True)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    sc_last = None
    for i, (lbl, gt, pred, cyc) in enumerate(
            zip(all_labels, all_gt, all_pred, all_cyc)):
        ax = axes_flat[i]
        rmse, mae, r2 = _metrics(gt, pred)

        sc = ax.scatter(gt, pred, c=cyc, cmap='viridis',
                        s=8, alpha=0.65, linewidths=0)
        sc_last = sc

        lo = min(gt.min(), pred.min())
        hi = max(gt.max(), pred.max())
        pad = 0.03 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        ax.text(0.04, 0.96,
                f'RMSE={rmse:.4f}\nMAE={mae:.4f}\nR²={r2:.4f}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel('True (Ah)', fontsize=8)
        ax.set_ylabel('Pred (Ah)', fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if sc_last is not None:
        cb_axes = axes_flat[:n].tolist() if n > 1 else axes_flat[0]
        fig.colorbar(sc_last, ax=cb_axes, shrink=0.6, pad=0.02,
                     label='Cycle Index')

    gt_cat   = np.concatenate(all_gt)
    pred_cat = np.concatenate(all_pred)
    rmse_ov, mae_ov, r2_ov = _metrics(gt_cat, pred_cat)
    fig.suptitle(
        f'All Batteries — RMSE={rmse_ov:.4f} Ah  '
        f'MAE={mae_ov:.4f} Ah  R²={r2_ov:.4f}',
        fontsize=11
    )

    out = os.path.join(output_dir, 'all_batteries_regression.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Overall grid → all_batteries_regression.png')


# Main entry point

def main():
    cfg = SOHConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device(cfg.device)
    print(f'\nDevice      : {device}')
    print(f'Data        : {cfg.data_dir}')
    print(f'Checkpoints : {cfg.checkpoint_dir}')
    print(f'Output      : {cfg.output_dir}')

    # Load the scaler.
    soh_min = float(np.load(os.path.join(cfg.checkpoint_dir, 'soh_min.npy')))
    soh_max = float(np.load(os.path.join(cfg.checkpoint_dir, 'soh_max.npy')))
    print(f'SOH range   : [{soh_min:.4f}, {soh_max:.4f}] Ah\n')

    # Load the model.
    ckpt_path = cfg.checkpoint_path or os.path.join(cfg.checkpoint_dir, 'best_model.pt')
    model = build_model(cfg)
    load_checkpoint(model, None, ckpt_path, device)
    print()

    # Find test files.
    test_files = find_test_files(cfg.data_dir)
    if not test_files:
        raise RuntimeError(
            f'No *_test_soh.mat found in {cfg.data_dir}'
        )
    print(f'Found {len(test_files)} test file(s):')
    for _, lbl in test_files:
        print(f'  {lbl}')
    print()

    # Run inference and save outputs file by file.
    all_labels, all_gt, all_pred, all_cyc = [], [], [], []

    for fpath, label in test_files:
        result = load_one_file(fpath)
        if result is None:
            continue
        gaf_gen, capacity, info = result

        gt, pred_mean, bat_idx, cycle_idx = predict_file(
            model, gaf_gen, capacity, info,
            soh_min, soh_max, cfg.batch_size, device
        )

        # Save NPY.
        save_battery_npy(label, gt, pred_mean,
                         bat_idx, cycle_idx, cfg.output_dir)

        # Per-battery plots.
        plot_battery_regression(label, gt, pred_mean,
                                cycle_idx, cfg, cfg.output_dir)
        plot_battery_trajectory(label, gt, pred_mean,
                                cycle_idx, cfg.output_dir)

        all_labels.append(label)
        all_gt.append(gt)
        all_pred.append(pred_mean)
        all_cyc.append(cycle_idx)

    # Overall regression grid.
    if all_gt:
        plot_overall_grid(all_labels, all_gt, all_pred,
                          all_cyc, cfg, cfg.output_dir)

    # Overall metrics.
    if all_gt:
        gt_all   = np.concatenate(all_gt)
        pred_all = np.concatenate(all_pred)
        rmse, mae, r2 = _metrics(gt_all, pred_all)
        print(f'\n── Overall ({len(gt_all)} cycles) ────────────────────────')
        print(f'  RMSE = {rmse:.6f} Ah')
        print(f'  MAE  = {mae:.6f} Ah')
        print(f'  R²   = {r2:.6f}')

    print(f'\nDone. Results saved to: {cfg.output_dir}')


if __name__ == '__main__':
    main()
