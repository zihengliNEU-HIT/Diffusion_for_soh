"""Central configuration for the image-only SOH estimator."""
import argparse
import torch


class SOHConfig:
    # Data
    data_dir        = "./soh_input/test_random"
    output_dir      = "./test_random_output"
    checkpoint_dir  = "./soh_checkpoints"

    # MAT field names
    field_gaf_real  = "GAF_real"        # [N, H, W], used by train only
    field_gaf_gen   = "GAF_gen"         # [N, K, H, W], K depends on split
    field_capacity  = "CAPACITY"        # [N, 1]
    field_info      = "INFO"            # [N, 2]

    # Image
    img_size        = 128

    # Reference GAF_gen counts for each split; actual K is read from MAT files.
    n_gen_train     = 60                # 3 windows x 20 images
    n_gen_val       = 20                # one selected window x 20 images
    n_gen_test      = 5                 # inference may use K=1 or K=5

    # Model
    cnn_channels    = (32, 64, 128, 256)
    num_heads       = 8
    dropout         = 0.1

    # Training
    batch_size      = 32
    num_epochs      = 200
    lr              = 1e-3
    weight_decay    = 1e-4
    patience        = 30
    eval_interval   = 5

    lambda1         = 0.05               # L_gen weight

    scheduler       = "cosine"
    step_size       = 20
    step_gamma      = 0.5

    # Visualization
    attn_vis_n      = 0                 # number of cycles visualized per battery

    # Runtime
    device          = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers     = 4
    seed            = 42

    # Testing
    checkpoint_path = None


def parse_args() -> SOHConfig:
    cfg = SOHConfig()
    p = argparse.ArgumentParser(description="SOH Estimator (image-only)")
    p.add_argument("--data_dir",        default=cfg.data_dir)
    p.add_argument("--output_dir",      default=cfg.output_dir)
    p.add_argument("--checkpoint_dir",  default=cfg.checkpoint_dir)
    p.add_argument("--checkpoint_path", default=cfg.checkpoint_path)
    p.add_argument("--batch_size",      type=int,   default=cfg.batch_size)
    p.add_argument("--num_epochs",      type=int,   default=cfg.num_epochs)
    p.add_argument("--lr",              type=float, default=cfg.lr)
    p.add_argument("--weight_decay",    type=float, default=cfg.weight_decay)
    p.add_argument("--lambda1",         type=float, default=cfg.lambda1)
    p.add_argument("--patience",        type=int,   default=cfg.patience)
    p.add_argument("--scheduler",       default=cfg.scheduler,
                   choices=["cosine", "steplr", "none"])
    p.add_argument("--device",          default=cfg.device)
    p.add_argument("--num_workers",     type=int,   default=cfg.num_workers)
    p.add_argument("--seed",            type=int,   default=cfg.seed)
    p.add_argument("--attn_vis_n",      type=int,   default=cfg.attn_vis_n)
    p.add_argument("--mode",            default="train",
                   choices=["train", "test"])

    args = p.parse_args()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
