import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision.utils import make_grid


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
    print(f"Saved comparison grid to {save_path}")


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
    print(f"Saved training samples to {save_path}")


import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm


def visualize_diffusion_process(
    model,
    condition,
    target,
    cond_vec=None,
    device="cuda",
    selected_steps=None,
    save_path=None,
):
    model.ema_network.eval()
    condition = condition.to(device)
    target = target.to(device)
    mondrian_colors = ["#4573b4", "#acd2e5", "#fff4ae", "#fcb777"]
    cmap_mondrian = LinearSegmentedColormap.from_list(
        "mondrian", mondrian_colors, N=256
    )
    total_steps = model.timesteps
    if selected_steps is None:
        selected_steps = list(range(total_steps - 1, -1, -total_steps // 5))
        selected_steps = sorted(set(selected_steps), reverse=True)
    img = torch.randn((1, 1, target.shape[2], target.shape[3]), device=device)
    generated_at_steps = {}
    with torch.no_grad():
        for i in tqdm(
            reversed(range(0, total_steps)),
            desc="Generating process",
            total=total_steps,
        ):
            timestep = torch.full((1,), i, device=device, dtype=torch.long)
            pred_noise = model.ema_network(img, timestep, condition, cond_vec=cond_vec)
            img = model.gdf_util.p_sample(pred_noise, img, timestep)
            if i in selected_steps:
                generated_at_steps[i] = img.cpu().numpy()
    n = len(selected_steps)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for idx, t in enumerate(sorted(selected_steps, reverse=True)):
        gen = generated_at_steps[t][0, 0]
        tar = target[0, 0].cpu().numpy()
        axes[idx, 0].imshow(gen, cmap=cmap_mondrian)
        axes[idx, 0].set_title(f"Generated (t={t})")
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(tar, cmap=cmap_mondrian)
        axes[idx, 1].set_title("Target")
        axes[idx, 1].axis("off")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"Saved diffusion step visualization: {save_path}")
    plt.close(fig)
