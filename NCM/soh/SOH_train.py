"""
SOH_train.py (v2 - image-only)
-----------------------------------------------------------------------------
Dual-path loss:
  Path A: GAF_real (1 image) -> pred_real -> Huber(pred_real, SOH)
  Path B: GAF_gen  (K imgs)  -> K preds -> mean prediction -> Huber(mean, SOH)
  Total = L_real + lambda1 * L_gen

COND_VEC and consistency loss are not used. Validation uses only the generated
GAF path to match the test-time setting.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from SOH_config import SOHConfig
from SOH_data_process import get_dataloaders
from SOH_model import build_model
from SOH_utils import save_checkpoint, load_checkpoint, plot_loss_curve


# def dual_path_loss(model, batch, cfg, device):
#     """
#     Path A: GAF_real (B,1,H,W) -> MSE
#     Path B: GAF_gen  (B,K,1,H,W) -> flatten -> (B*K,1,H,W) -> MSE
#     Return total_loss, loss_real.item(), loss_gen.item()
#     """
#     gaf_real = batch['gaf_real'].to(device)    # (B,1,H,W)
#     gaf_gen  = batch['gaf_gen'].to(device)     # (B,K,1,H,W)
#     soh_gt   = batch['soh'].to(device)         # (B,1)

#     B, K, _, H, W = gaf_gen.shape

#     # Path A: real GAF.
#     pred_real = model(gaf_real)                # (B,1)
#     loss_real = nn.functional.mse_loss(pred_real, soh_gt)

#     # Path B: use all generated GAF images.
#     gaf_gen_flat = gaf_gen.view(B * K, 1, H, W)
#     soh_gt_rep   = soh_gt.repeat_interleave(K, dim=0)          # (B*K,1)
#     pred_gen     = model(gaf_gen_flat)                          # (B*K,1)
#     loss_gen     = nn.functional.mse_loss(pred_gen, soh_gt_rep)


#     total = loss_real + cfg.lambda1 * loss_gen
#     return total, loss_real.item(), loss_gen.item()
def dual_path_loss(model, batch, cfg, device):
    gaf_real = batch['gaf_real'].to(device)    # (B,1,H,W)
    gaf_gen  = batch['gaf_gen'].to(device)     # (B,K,1,H,W)
    soh_gt   = batch['soh'].to(device)         # (B,1)

    B, K, _, H, W = gaf_gen.shape

    # Path A: real GAF.
    pred_real = model(gaf_real)                # (B,1)
    loss_real = nn.functional.huber_loss(pred_real, soh_gt, delta=0.1)

    # Path B: generated GAF images; average predictions before the loss.
    gaf_gen_flat  = gaf_gen.view(B * K, 1, H, W)
    pred_gen_all  = model(gaf_gen_flat).view(B, K, 1)   # (B,K,1)
    pred_gen_mean = pred_gen_all.mean(dim=1)             # (B,1)
    loss_gen      = nn.functional.huber_loss(pred_gen_mean, soh_gt, delta=0.1)

    total = loss_real + cfg.lambda1 * loss_gen
    return total, loss_real.item(), loss_gen.item()

def validate(model, val_loader, device):
    """
    Validation: run each generated GAF through the model, average predictions,
    and compute MSE. K is read from the validation MAT file.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            gaf_gen = batch['gaf_gen'].to(device)   # (B,K,1,H,W)
            soh_gt  = batch['soh'].to(device)
            B, K, _, H, W = gaf_gen.shape

            gaf_flat = gaf_gen.view(B * K, 1, H, W)
            pred     = model(gaf_flat).view(B, K, 1).mean(dim=1)   # (B,1)
            losses.append(nn.functional.mse_loss(pred, soh_gt).item())
    return float(np.mean(losses))


def main(cfg: SOHConfig = None):
    if cfg is None:
        from SOH_config import parse_args
        cfg = parse_args()

    device = torch.device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg.seed)

    # Data
    train_loader, val_loader, _, scaler = get_dataloaders(cfg)

    # Save the scaler.
    np.save(os.path.join(cfg.checkpoint_dir, 'soh_min.npy'), np.array(scaler['soh_min']))
    np.save(os.path.join(cfg.checkpoint_dir, 'soh_max.npy'), np.array(scaler['soh_max']))
    print("Scaler 已保存")

    # Model
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if cfg.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
    elif cfg.scheduler == 'steplr':
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma)
    else:
        scheduler = None

    # Resume training.
    start_epoch   = 0
    best_val_loss = float('inf')
    if cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, cfg.checkpoint_path, device
        )

    # Training loop
    train_losses, val_losses = [], []
    patience_cnt = 0
    val_loss = float('inf')

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        ep_total = ep_real = ep_gen = 0.0
        t0 = time.time()

        for batch in train_loader:
            if 'gaf_real' not in batch:
                continue
            optimizer.zero_grad()
            loss, lr, lg = dual_path_loss(model, batch, cfg, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_total += loss.item()
            ep_real  += lr
            ep_gen   += lg

        n = len(train_loader)
        train_losses.append(ep_total / n)
        if scheduler:
            scheduler.step()

        if (epoch + 1) % cfg.eval_interval == 0:
            val_loss = validate(model, val_loader, device)
            val_losses.append(val_loss)
            dt = time.time() - t0

            print(f"Epoch [{epoch+1:03d}/{cfg.num_epochs}]  "
                  f"Train {ep_total/n:.5f} "
                  f"(real={ep_real/n:.5f} gen={ep_gen/n:.5f})  "
                  f"Val {val_loss:.5f}  "
                  f"LR {optimizer.param_groups[0]['lr']:.2e}  "
                  f"[{dt:.1f}s]")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt  = 0
                save_checkpoint(model, optimizer, epoch, val_loss,
                                os.path.join(cfg.checkpoint_dir, 'best_model.pt'))
                print(f"  ✓ Best model saved (val={val_loss:.5f})")
            else:
                patience_cnt += 1
                if patience_cnt >= cfg.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 20 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(cfg.checkpoint_dir,
                                         f'model_epoch_{epoch+1:03d}.pt'))

    plot_loss_curve(train_losses, val_losses, cfg.eval_interval,
                    os.path.join(cfg.output_dir, 'loss_curve.png'))
    print("训练完成")


if __name__ == "__main__":
    main()
