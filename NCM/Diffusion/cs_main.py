import os
import argparse
import json
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
from dataset import get_dataloaders
from gaussian_diffusion import GaussianDiffusion
from unet import UNet
from diffusion_model import DiffusionModel
from visualization import visualize_diffusion_process


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
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config", type=str, default=None, help="Path to a JSON configuration file."
    )
    base_args, _ = base_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        description="Conditional diffusion training and testing entry point.",
        parents=[base_parser],
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Checkpoint path"
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "test"], default="train", help="Run mode"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=260, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--timesteps", type=int, default=400, help="Diffusion timesteps"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=10, help="Evaluation interval"
    )
    parser.add_argument("--save_interval", type=int, default=10, help="Save interval")
    parser.add_argument(
        "--patience", type=int, default=30, help="Early-stopping patience"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["steplr", "cosine", "plateau", "none"],
        default="steplr",
        help="Learning-rate scheduler type",
    )
    parser.add_argument(
        "--step_size", type=int, default=10, help="StepLR: Decay interval in epochs"
    )
    parser.add_argument(
        "--step_gamma",
        type=float,
        default=0.8,
        help="StepLR: Decay factor (new lr = lr * gamma)",
    )
    parser.add_argument(
        "--noise_schedule",
        type=str,
        choices=["linear", "cosine"],
        default="cosine",
        help="Noise schedule type",
    )
    parser.add_argument(
        "--clip_grad", type=float, default=1, help="Gradient clipping threshold"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="Weight decay"
    )
    parser.add_argument("--ema", type=float, default=0.99995, help="EMA coefficient")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of data-loading workers"
    )
    parser.add_argument(
        "--sample_idx", type=int, default=5, help="Sample index for visualization"
    )
    parser.add_argument(
        "--cosine_s",
        type=float,
        default=0.008,
        help="Cosine schedule smoothing parameter",
    )
    parser.add_argument(
        "--cond_vector_dim",
        type=int,
        default=26,
        help="Scalar condition-vector dimension (0 disables it)",
    )
    parser.add_argument(
        "--use_ddim", action="store_true", default=True, help="Use DDIM sampling"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=50, help="Number of DDIM sampling steps"
    )
    parser.add_argument(
        "--ddim_eta", type=float, default=0.0, help="DDIM stochasticity parameter"
    )
    _apply_config_defaults(parser, _load_config(base_args.config))
    return parser.parse_args()


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
        cond_vector_dim=args.cond_vector_dim,
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
        cond_vector_dim=args.cond_vector_dim,
    ).to(args.device)

    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    network.apply(init_weights)
    ema_network.load_state_dict(network.state_dict())
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


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.mode == "train":
        from train import main as train_main

        train_main(args)
    elif args.mode == "test":
        from test import main as test_main

        test_main(args)


if __name__ == "__main__":
    main()
