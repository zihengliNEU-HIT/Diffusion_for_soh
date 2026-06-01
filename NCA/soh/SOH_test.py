"""
SOH_test.py (v2 - image-only with spatial attention visualization)

CUDA_VISIBLE_DEVICES=2 python SOH_test.py
-----------------------------------------------------------------------------
Inference:
  Each cycle has K generated GAF images. The model predicts each image
  independently and reports mean +/- std across the K predictions.
  Spatial attention maps can also be saved for each generated GAF image.

Outputs:
  metrics.txt              MAE / RMSE / MAPE / R2
  regression_plot.png      regression scatter plot
  trajectory_*.png         per-battery degradation trajectories
  predictions.npy          full prediction records
  attn_maps/               spatial attention visualizations
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from SOH_config import SOHConfig
from SOH_model import build_model
from SOH_utils import (load_checkpoint, compute_metrics,
                       plot_regression, plot_trajectory_per_battery,
                       plot_loss_curve, denorm_soh,
                       save_attention_maps)


@torch.no_grad()
def run_inference(model, test_loader, device, soh_min, soh_max,
                  save_attn=True, attn_dir=None, attn_vis_n=6):
    """
    Return records as a list of dictionaries:
        battery_id, cycle_idx,
        soh_gt_raw, soh_pred_mean_raw, soh_pred_std_raw, soh_preds_raw
    """
    model.eval()
    records = []

    # Save attention maps for at most attn_vis_n cycles per battery.
    attn_counter = {}

    for batch in test_loader:
        gaf_gen  = batch['gaf_gen'].to(device)     # (B,K,1,H,W)
        soh_gt   = batch['soh'].cpu().numpy()       # (B,1)
        bids     = batch['battery_id']
        cidxs    = batch['cycle_idx']
        B, K, _, H, W = gaf_gen.shape

        # SOH inference.
        gaf_flat  = gaf_gen.view(B * K, 1, H, W)
        pred_flat = model(gaf_flat).cpu().numpy()   # (B*K,1)
        pred_all  = pred_flat.reshape(B, K)         # (B,K)

        # Attention inference for each GAF image.
        if save_attn and attn_dir:
            _, attn_flat = model(gaf_flat, return_attn=True)
            # attn_flat: (B*K, 8, 8)
            attn_all = attn_flat.cpu().numpy().reshape(B, K, 8, 8)
        else:
            attn_all = None

        for i in range(B):
            bid  = int(bids[i])
            cidx = int(cidxs[i])

            preds_norm = pred_all[i]
            gt_norm    = float(soh_gt[i, 0])

            preds_raw = denorm_soh(preds_norm, soh_min, soh_max)
            gt_raw    = denorm_soh(np.array([gt_norm]), soh_min, soh_max)[0]

            records.append({
                'battery_id':        bid,
                'cycle_idx':         cidx,
                'soh_gt_raw':        gt_raw,
                'soh_pred_mean_raw': preds_raw.mean(),
                'soh_pred_std_raw':  preds_raw.std(),
                'soh_preds_raw':     preds_raw,
            })

            # Save attention maps.
            if attn_all is not None:
                cnt = attn_counter.get(bid, 0)
                if cnt < attn_vis_n:
                    # gaf_imgs: (K,1,H,W) -> denormalize to [0,1]
                    gaf_imgs = gaf_gen[i].cpu()              # (K,1,H,W)
                    gaf_imgs = (gaf_imgs + 1.0) / 2.0        # [0,1]
                    attn_maps = attn_all[i]                  # (K,8,8)

                    save_attention_maps(
                        gaf_imgs=gaf_imgs,
                        attn_maps=attn_maps,
                        battery_id=bid,
                        cycle_idx=cidx,
                        soh_gt=gt_raw,
                        soh_pred_mean=preds_raw.mean(),
                        save_dir=attn_dir,
                    )
                    attn_counter[bid] = cnt + 1

    records.sort(key=lambda r: (r['battery_id'], r['cycle_idx']))
    return records


def main(cfg: SOHConfig = None):
    if cfg is None:
        from SOH_config import parse_args
        cfg = parse_args()

    device = torch.device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)
    attn_dir = os.path.join(cfg.output_dir, 'attn_maps')
    os.makedirs(attn_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)

    # Load the scaler.
    ckpt_dir = cfg.checkpoint_dir
    soh_min  = float(np.load(os.path.join(ckpt_dir, 'soh_min.npy')))
    soh_max  = float(np.load(os.path.join(ckpt_dir, 'soh_max.npy')))
    print(f"SOH denormalization range: [{soh_min:.4f}, {soh_max:.4f}] Ah")

    # Data
    from SOH_data_process import SOHDataset, collate_fn
    from torch.utils.data import DataLoader

    test_ds = SOHDataset(cfg.data_dir, 'test',
                         soh_min=soh_min, soh_max=soh_max)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        persistent_workers=(cfg.num_workers > 0),
    )

    # Model
    model = build_model(cfg)
    ckpt_path = cfg.checkpoint_path or os.path.join(ckpt_dir, 'best_model.pt')
    load_checkpoint(model, None, ckpt_path, device)

    # Inference
    print(f"Starting inference on {len(test_ds)} sample(s) ...")
    records = run_inference(
        model, test_loader, device, soh_min, soh_max,
        save_attn=True, attn_dir=attn_dir,
        attn_vis_n=cfg.attn_vis_n,
    )

    # Collect arrays.
    gt    = np.array([r['soh_gt_raw']        for r in records])
    pred  = np.array([r['soh_pred_mean_raw'] for r in records])
    std   = np.array([r['soh_pred_std_raw']  for r in records])
    bids  = np.array([r['battery_id']         for r in records])
    cidxs = np.array([r['cycle_idx']          for r in records])

    # Metrics
    metrics = compute_metrics(gt, pred)
    print("\n===== Test Results =====")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    with open(os.path.join(cfg.output_dir, 'metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    # Visualization
    # plot_regression(gt, pred, std, bids,
    #                 os.path.join(cfg.output_dir, 'regression_plot.png'))
    # plot_trajectory_per_battery(bids, cidxs, gt, pred, std, cfg.output_dir)


    plot_regression(gt, pred, np.zeros_like(pred), bids,
                os.path.join(cfg.output_dir, 'regression_plot.png'))
    plot_trajectory_per_battery(bids, cidxs, gt, pred, np.zeros_like(pred), cfg.output_dir)

    # Save predictions.
    np.save(os.path.join(cfg.output_dir, 'predictions.npy'),
            np.array([{
                'battery_id': r['battery_id'],
                'cycle_idx':  r['cycle_idx'],
                'gt':   r['soh_gt_raw'],
                'mean': r['soh_pred_mean_raw'],
                'std':  r['soh_pred_std_raw'],
                'preds': r['soh_preds_raw'],
            } for r in records], dtype=object))

    print(f"\nAll results saved to {cfg.output_dir}")
    print(f"Attention maps saved to {attn_dir}")


if __name__ == "__main__":
    main()
