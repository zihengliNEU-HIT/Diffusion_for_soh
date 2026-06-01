"""
visualization.py
Content : Plotting utilities - training sample grids, comparison grids,
          loss/PSNR curves.
Run     : imported only, not executed directly.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.ticker import FuncFormatter


# -- sample grids -------------------------------------------------------------

def save_training_samples(inputs, targets, generated, save_path, num_samples=4):
    """Save FRAG | Target | Generated grid for up to num_samples rows."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    n = min(num_samples, inputs.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    titles = ["FRAG", "Target", "Generated"]
    for i in range(n):
        for j, img in enumerate([inputs[i, 0], targets[i, 0], generated[i, 0]]):
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].set_title(titles[j])
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training samples -> {save_path}")


def save_comparison_grid(inputs, targets, generated, save_path, num_samples=16):
    """Save a grid where each cell contains FRAG | Target | Generated."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    n         = min(num_samples, len(inputs))
    grid_size = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes      = np.array(axes).flatten()

    for i in range(grid_size * grid_size):
        if i < n:
            inner = plt.figure(figsize=(9, 3))
            ia    = [inner.add_subplot(1, 3, k + 1) for k in range(3)]
            ia[0].imshow(inputs[i][0],  cmap="gray"); ia[0].set_title("FRAG");      ia[0].axis("off")
            ia[1].imshow(targets[i],    cmap="gray"); ia[1].set_title("Target");     ia[1].axis("off")
            ia[2].imshow(generated[i],  cmap="gray"); ia[2].set_title("Generated");  ia[2].axis("off")
            inner.tight_layout()
            tmp = os.path.join(os.path.dirname(save_path) or ".", f"_tmp_{i}.png")
            inner.savefig(tmp, dpi=100, bbox_inches="tight")
            plt.close(inner)
            axes[i].imshow(plt.imread(tmp))
            axes[i].set_title(f"Sample {i+1}")
            axes[i].axis("off")
            os.remove(tmp)
        else:
            axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison grid -> {save_path}")


# -- loss / PSNR curves --------------------------------------------------------

def plot_losses_with_psnr(train_losses, val_losses, psnr_history, output_dir, learning_rates=None):
    """4-panel figure: loss curves, PSNR curve, LR schedule, PSNR histogram."""

    def _sanitize(fig):
        for txt in fig.findobj(mtext.Text):
            s = txt.get_text()
            if s:
                txt.set_text(s.replace("\u2212", "-").replace("\u2013", "-"))
        def _fmt(x, pos):
            return f"{float(x):g}".replace("\u2212", "-")
        for ax in fig.axes:
            ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt))

    # ========== Save curve data to a text file ==========
    data_save_path = os.path.join(output_dir, 'training_data.txt')
    with open(data_save_path, 'w', encoding='utf-8') as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\tPSNR\tLearning_Rate\n")
        max_len = max(len(train_losses), len(val_losses), len(psnr_history), len(learning_rates) if learning_rates else 0)
        for i in range(max_len):
            train_loss = train_losses[i] if i < len(train_losses) else ""
            val_loss = val_losses[i] if i < len(val_losses) else ""
            psnr = psnr_history[i] if i < len(psnr_history) else ""
            lr = learning_rates[i] if learning_rates and i < len(learning_rates) else ""
            f.write(f"{i+1}\t{train_loss}\t{val_loss}\t{psnr}\t{lr}\n")
    print(f"Training data saved to: {data_save_path}")
    # ========== End data export ==========

    with plt.rc_context({"font.family": "DejaVu Sans", "text.usetex": False,
                         "axes.unicode_minus": False}):

        plt.figure(figsize=(15, 10))

        # loss curves
        plt.subplot(2, 2, 1)
        plt.plot([float(v) for v in train_losses], label="Train Loss", color="blue")
        if val_losses:
            step = max(1, len(train_losses) // max(1, len(val_losses)))
            xv   = list(range(0, len(train_losses), step))[:len(val_losses)]
            plt.plot(xv, [float(v) for v in val_losses], label="Val Loss", color="orange")
        plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(); plt.grid(True)

        # PSNR curve
        plt.subplot(2, 2, 2)
        if psnr_history:
            step  = max(1, len(train_losses) // max(1, len(psnr_history)))
            xp    = list(range(0, len(train_losses), step))[:len(psnr_history)]
            plt.plot(xp, [float(v) for v in psnr_history], color="green", marker="o", label="PSNR")
            for y, col, lbl in [(38, "red","Target 38dB"), (35,"orange","Good 35dB"), (30,"gold","Decent 30dB")]:
                plt.axhline(y, color=col, linestyle="--", alpha=0.7, label=lbl)
        plt.title("PSNR"); plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)")
        plt.legend(); plt.grid(True)

        # LR
        if learning_rates:
            plt.subplot(2, 2, 3)
            plt.plot([float(v) for v in learning_rates], color="purple", label="LR")
            plt.title("Learning Rate"); plt.xlabel("Epoch"); plt.ylabel("LR")
            plt.yscale("log"); plt.legend(); plt.grid(True)

        # PSNR histogram
        plt.subplot(2, 2, 4)
        if psnr_history:
            pv = [float(v) for v in psnr_history]
            plt.hist(pv, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
            plt.axvline(38.0, color="red",   linestyle="--", alpha=0.7, label="Target (38dB)")
            plt.axvline(float(np.mean(pv)), color="green", linestyle="-", alpha=0.7,
                        label=f"Mean ({float(np.mean(pv)):.1f}dB)")
            plt.title("PSNR Distribution"); plt.xlabel("PSNR (dB)"); plt.ylabel("Freq")
            plt.legend(); plt.grid(True)

        _sanitize(plt.gcf())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_metrics.png"), dpi=300, bbox_inches="tight")
        plt.close()
