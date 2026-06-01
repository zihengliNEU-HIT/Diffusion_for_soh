import argparse
import json
import os
import torch
from gaussian_diffusion import GaussianDiffusion
from unet import UNet
from diffusion_model import DiffusionModel


def _load_config_defaults(config_path):
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = _load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description="LFP conditional diffusion training and testing")
    parser.add_argument("--config", type=str, default=pre_args.config, help="Path to a JSON configuration file.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=260)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--scheduler", type=str, choices=["steplr", "cosine", "plateau", "none"], default="steplr")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--step_gamma", type=float, default=0.8)
    parser.add_argument("--noise_schedule", type=str, choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--cosine_s", type=float, default=0.008)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema", type=float, default=0.99995)
    parser.add_argument("--cond_vector_dim", type=int, default=24)
    parser.add_argument("--use_ddim", action="store_true", default=True)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--oversample_tail", action="store_true", default=True)
    parser.add_argument("--oversample_tail_factor", type=int, default=3)
    parser.set_defaults(**defaults)
    return parser.parse_args(remaining)


def create_model(args):
    def make_unet():
        return UNet(
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

    network = make_unet()
    ema_network = make_unet()

    def init_weights(module):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    network.apply(init_weights)
    ema_network.load_state_dict(network.state_dict())
    gdf_util = GaussianDiffusion(
        timesteps=args.timesteps,
        device=args.device,
        schedule_type=args.noise_schedule,
        cosine_s=args.cosine_s,
    )
    return DiffusionModel(network, ema_network, gdf_util, args.timesteps, args.ema).to(args.device)


def main():
    args = parse_args()
    device = torch.device(args.device)
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
