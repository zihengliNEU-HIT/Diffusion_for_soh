import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision.utils import make_grid


def save_training_samples(inputs, targets, generated, save_path, num_samples=4):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(num_samples, inputs.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        axes[i, 0].imshow(inputs[i, 0], cmap="gray")
        axes[i, 0].set_title("FRAG")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(targets[i, 0], cmap="gray")
        axes[i, 1].set_title("Target")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(generated[i, 0], cmap="gray")
        axes[i, 2].set_title("Generated")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_comparison_grid(inputs, targets, generated, save_path, num_samples=16):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(num_samples, len(inputs))
    grid_size = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    if grid_size == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i in range(grid_size * grid_size):
        if i < n:
            inner_fig = plt.figure(figsize=(9, 3))
            inner_axes = [inner_fig.add_subplot(1, 3, j + 1) for j in range(3)]
            inner_axes[0].imshow(inputs[i][0], cmap="gray")
            inner_axes[0].set_title("FRAG")
            inner_axes[0].axis("off")
            inner_axes[1].imshow(targets[i], cmap="gray")
            inner_axes[1].set_title("Target")
            inner_axes[1].axis("off")
            inner_axes[2].imshow(generated[i], cmap="gray")
            inner_axes[2].set_title("Generated")
            inner_axes[2].axis("off")
            plt.tight_layout()
            temp_path = os.path.join(os.path.dirname(save_path), f"temp_sample_{i}.png")
            inner_fig.savefig(temp_path, dpi=100, bbox_inches="tight")
            plt.close(inner_fig)
            img = plt.imread(temp_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Sample {i + 1}")
            axes[i].axis("off")
            os.remove(temp_path)
        else:
            axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_comparison_grid_compact(inputs, targets, generated, save_path, num_samples=16):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(num_samples, len(inputs))
    grid_size = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    if grid_size == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i in range(grid_size * grid_size):
        if i < n:
            inner_fig = plt.figure(figsize=(9, 3))
            inner_axes = [inner_fig.add_subplot(1, 3, j + 1) for j in range(3)]
            inner_axes[0].imshow(inputs[i][0], cmap="gray")
            inner_axes[0].set_title("FRAG")
            inner_axes[0].axis("off")
            inner_axes[1].imshow(targets[i], cmap="gray")
            inner_axes[1].set_title("Target")
            inner_axes[1].axis("off")
            inner_axes[2].imshow(generated[i], cmap="gray")
            inner_axes[2].set_title("Generated")
            inner_axes[2].axis("off")
            plt.tight_layout()
            temp_path = os.path.join(os.path.dirname(save_path), f"temp_sample_{i}.png")
            inner_fig.savefig(temp_path, dpi=100, bbox_inches="tight")
            plt.close(inner_fig)
            img = plt.imread(temp_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Sample {i + 1}")
            axes[i].axis("off")
            os.remove(temp_path)
        else:
            axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attention_maps(model, condition, save_path=None):
    model.eval()
    attention_modules = []

    def hook_fn(module, input, output):
        attention_modules.append(module)

    hooks = []
    for name, module in model.named_modules():
        if "attention" in name:
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    with torch.no_grad():
        x = torch.randn((1, 1, 128, 128), device=condition.device)
        t = torch.zeros((1,), device=condition.device, dtype=torch.long)
        model(x, t, condition)
    for hook in hooks:
        hook.remove()
    num_attention_modules = len(attention_modules)
    if num_attention_modules == 0:
        return
    grid_size = int(np.ceil(np.sqrt(num_attention_modules)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    if grid_size == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, module in enumerate(attention_modules):
        if i < grid_size * grid_size:
            attn_weights = module.query(module.norm(x))
            attn_map = attn_weights.mean(dim=1)[0].cpu().numpy()
            im = axes[i].imshow(attn_map, cmap="viridis")
            axes[i].set_title(f"Attention {i + 1}")
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i])
    for i in range(num_attention_modules, grid_size * grid_size):
        axes[i].axis("off")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
