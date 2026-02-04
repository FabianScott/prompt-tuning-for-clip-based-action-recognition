import os
import matplotlib.cm as cm
import torch
import math
import typer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Iterable, Optional
from loguru import logger
from tqdm import tqdm

from .configs.paths import FIGURES_DIR


# Plotting
def plot_heatmap(
        data,
        title,
        ax,
        xlabel="",
        ylabel="",
        xticks: Optional[Iterable] = None,
        yticks: Optional[Iterable] = None,
        cmap="plasma",
        fmt=".2f"
        ):
    """    Plots a heatmap with the given data and title.
    Args:
        data (np.ndarray): 2D iterable of the data to plot.
        title (str): The title of the heatmap.
        ax (matplotlib.axes.Axes): The axes to plot on.
        xticks (Optional[Iterable]): Optional x-ticks labels.
        yticks (Optional[Iterable]): Optional y-ticks labels.
    """
    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=[str(el) for el in xticks] if xticks is not None else None, # pyright: ignore[reportArgumentType]
        yticklabels=[str(el) for el in yticks] if yticks is not None else None, # pyright: ignore[reportArgumentType]
        cbar_kws={'label': title}
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_video_grid(video_tensor: torch.Tensor, save_path: Optional[str] = None, max_fig_size: int = 16, title="Video Grid"):
    """
    video_tensor: (frames, H, W, C) torch tensor
    save_path: optional path to save the figure
    max_fig_size: maximum figure dimension (width/height)
    """
    video_tensor = video_tensor.detach().cpu()
    frames, H, W, C = video_tensor.shape

    # Determine grid size: make nearly square
    cols = math.ceil(math.sqrt(frames))
    rows = math.ceil(frames / cols)

    # Figure size scaled to number of rows/cols
    fig_width = min(cols * 2, max_fig_size)
    fig_height = min(rows * 2, max_fig_size)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    for i in range(frames):
        frame = video_tensor[i].numpy().astype("uint8")
        axes[i].imshow(frame)
        axes[i].axis('off')

    for j in range(frames, rows*cols):
        axes[j].axis('off')

    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()




def plot_metric_distribution(
    scores_per_class: dict,
    metric: str,
    ax: plt.Axes,
    bins: int = 15,
    alpha: float = 0.7,
    color: str = 'blue',
    percentiles: list[float] = None,
    fontsize_percentiles: int = 10,
    percentage_axis: bool = True,
    grid: bool = True,
    ylim_max: Optional[float] = None
):
    values = [scores_per_class[c][metric] for c in scores_per_class]
    counts, bins_edges, patches = ax.hist(values, bins=bins, edgecolor='black', alpha=alpha, color=color)
    ax.set_title(f'{metric.capitalize()} Distribution')
    ax.set_xlabel(metric.capitalize())
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ylim_max) if ylim_max is not None else None
    ax.set_xticks(np.linspace(0, 1, 11))
    if grid:
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    # Add percentiles
    if percentiles:
        for p in percentiles:
            val = np.percentile(values, p)
            ax.axvline(val, color='black', linestyle='--', linewidth=1)
            ax.text(val+0.05, ax.get_ylim()[1]*0.95, f'{p}%', color='black', ha='center', fontsize=fontsize_percentiles)
    # Secondary y-axis (percentages)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())  # important: keep perfectly aligned

    if percentage_axis:
        ax2.yaxis.set_major_locator(ax.yaxis.get_major_locator())

        yticks = ax.get_yticks()  # use the EXACT positions of the primary axis
        total = sum(counts)
        ax2.set_yticklabels([f'{(y/total*100):.1f}%' for y in yticks])
        ax2.set_ylabel('Percentage')
    else:
        ax2.tick_params(axis='y', labelleft=False, labelright=False)


def plot_metrics_grid(
        scores_per_class: dict, 
        metrics=('f1', 'precision', 'recall', ), 
        savename: Optional[str] = None,
        show: bool = False,
        bins: int = 15,
        alpha: float = 0.7,
        colors: list[str] = ['blue', 'green', 'orange', 'red'],
        percentiles: list[float] = [5, 25, 50, 75, 95],
        fontsize_percentiles: int = 7
        ):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.flatten()

    max_ylim = 0
    for metric in metrics:
        values = [scores_per_class[c][metric] for c in scores_per_class]
        counts, _ = np.histogram(values, bins=bins, range=(0, 1))
        max_ylim = max(max_ylim, counts.max())

    for i, (ax, metric, color) in enumerate(zip(axes, metrics, colors)):
        plot_metric_distribution(
            scores_per_class, 
            metric, 
            ax, 
            bins=bins, 
            alpha=alpha, 
            color=color, 
            percentiles=percentiles, 
            fontsize_percentiles=fontsize_percentiles,
            percentage_axis=i==(len(axes)-1),
            ylim_max=max_ylim
            )    
    # # unify y-limit = biggest among all
    # max_ylim = max(ax.get_ylim()[1] for ax in axes)
    # for ax in axes:
    #     ax.set_ylim(0, max_ylim)   # force all same scale
    # Add grids after resetting y-limits
    for ax in axes:
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    
    axes[1].set_ylabel(None)
    axes[1].tick_params(axis='y', labelleft=False, labelright=False)

    axes[2].set_ylabel(None)
    axes[2].tick_params(axis='y', labelleft=False)

    plt.tight_layout()
    
    if savename is not None:
        save_path = Path(FIGURES_DIR) / savename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=400)
        print(f"Saved metric distribution figure to {save_path}")
        plt.close(fig)
    
    if show:
        plt.show()



def plot_histograms_over_epochs(
    hist_data_all: list,
    ax: plt.Axes,
    epochs: Optional[list[int]] = None,
    alpha: float = 0.8,
    linewidth: float = 2.0,
    fontsize: int = 12,
    percentage_axis: bool = True,
    grid: bool = True,
    ylim_max: Optional[float] = None,
    cmap: str = "viridis",
    label_prefix: str = "epoch",
):
    """
    hist_data_all: list of dicts with keys ['hist', 'bin_edges']
    """

    hists = torch.stack([h["hist"].float() for h in hist_data_all])  # [E, B]
    bin_edges = hist_data_all[0]["bin_edges"]

    if epochs is None:
        epochs = list(range(hists.shape[0]))

    if percentage_axis:
        hists = hists / hists.sum(dim=1, keepdim=True) * 100.0

    num_bins = hists.shape[1]
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, num_bins))

    for b in range(num_bins):
        ax.plot(
            epochs,
            hists[:, b].cpu(),
            label=f"{label_prefix} {bin_edges[b]:.2f}-{bin_edges[b+1]:.2f}",
            color=colors[b],
            alpha=alpha,
            linewidth=linewidth,
        )

    ax.set_xlabel("Epoch", fontsize=fontsize)
    ax.set_ylabel("Percentage" if percentage_axis else "Count", fontsize=fontsize)

    if ylim_max is not None:
        ax.set_ylim(top=ylim_max)

    if grid:
        ax.grid(True, alpha=0.3)

    ax.legend(fontsize=fontsize - 2, ncol=2)


def plot_gflops_vs_accuracy(
    models: list,
    ax: plt.Axes,
    alpha: float = 0.8,
    linewidth: float = 2.0,
    fontsize: int = 12,
    grid: bool = True,
    cmap: str = "viridis",
    ylim: Optional[tuple] = None,
    xlim: Optional[tuple] = None,
):

    colors = cm.get_cmap(cmap, len(models))

    for i, model in enumerate(models):
        gflops = model["gflops"]

        acc = model["accuracy"]

        ax.scatter(
            gflops,
            acc,
            color=colors(i),
            alpha=alpha,
            linewidths=linewidth,
            label=f'{model["name"]}',
        )
        ax.vlines(
            x=gflops,
            ymin=0,
            ymax=acc,
            colors=colors(i),
            linestyles='dashed',
            alpha=alpha - 0.2  # optional
        )

    ax.set_xlabel("GFLOPs (per inference)", fontsize=fontsize)
    ax.set_ylabel("Accuracy", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    ax.legend(fontsize=fontsize * 0.8)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if grid:
        ax.grid(True)


def plot_video_examples(
    examples: torch.Tensor,
    labels: torch.Tensor,
    class_names: list,
    nrow: int = 4,
    save_path: Optional[str] = None,
    show=False,
    title: Optional[str] = None,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    unnormalise: bool = True,
):
    import matplotlib.pyplot as plt

    if mean is None:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, -1, 1, 1)
    if std is None:
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, -1, 1, 1)
    
    if unnormalise:
        examples = examples * std + mean
    examples = torch.clamp(examples, 0.0, 1.0)

    B, V, T, C, H, W = examples.shape

    fig, axes = plt.subplots(
        nrows=(B + nrow - 1) // nrow,
        ncols=nrow,
        figsize=(nrow * 4, ((B + nrow - 1) // nrow) * 4)
    )
    axes = axes.flatten()
    # cycle through views 
    view_idx = 0
    for i in range(B):
        frame = examples[i, view_idx, 0].to(torch.float32)  # first view, first frame → (C, H, W)
        if labels.ndim == 1:
            label = labels[i].item()
            class_name = class_names[label]
        elif labels.ndim == 2:
            label_indices = torch.nonzero(labels[i]).squeeze().tolist()
            if isinstance(label_indices, int):
                print(f"Single label detected for multi-label example: {label_indices}, converting to list")
                label_indices = [label_indices]
            class_name = " & ".join([class_names[idx] for idx in label_indices])
        else:
            print(f"Unexpected labels shape: {labels.shape}, setting class_name to 'unknown'")
            class_name = "unknown"

        axes[i].imshow(frame.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(f"{class_name} view {view_idx}")
        axes[i].axis("off")
        view_idx = (view_idx + 1) % V  # next view for next example

    for i in range(B, len(axes)):
        axes[i].axis("off")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
        print(f"Saved video examples to {save_path}")
    if show:
        plt.show()

def plot_numpy_video(
    videos: np.ndarray,
    frame_names: list,
    nrow: int = 4,
    save_path: Optional[str] = None,
    show=False,
    title: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    T, C, H, W = videos.shape

    fig, axes = plt.subplots(
        nrows=(T + nrow - 1) // nrow,
        ncols=nrow,
        figsize=(nrow * 4, ((T + nrow - 1) // nrow) * 4)
    )
    axes = axes.flatten()

    for i in range(T):
        frame = videos[i].astype("uint8").transpose(1, 2, 0)  # (H, W, C)
        axes[i].imshow(frame)
        axes[i].set_title(frame_names[i])
        axes[i].axis("off")

    for j in range(T, len(axes)):
        axes[j].axis("off")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
        print(f"Saved video examples to {save_path}")
    if show:
        plt.show()


def plot_reliability_diagram(probs, labels, n_bins=15, save_path=None, show=False):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_edges = torch.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    acc_per_bin = []
    conf_per_bin = []
    samples_per_bin = []

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.any():
            acc_per_bin.append(accuracies[mask].float().mean().item())
            conf_per_bin.append(confidences[mask].mean().item())
            samples_per_bin.append(mask.sum().item())
        else:
            acc_per_bin.append(0.0)
            conf_per_bin.append(bin_centers[i].item())
            samples_per_bin.append(0)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.RdYlGn([(acc - min(acc_per_bin)) / (max(acc_per_bin) - min(acc_per_bin)) if max(acc_per_bin) > min(acc_per_bin) else 0.5 for acc in acc_per_bin])
    bars = plt.bar(bin_centers, acc_per_bin, width=1 / n_bins, edgecolor="black", color=colors)
    plt.plot([0, 1], [0, 1], "--", linewidth=2, color="red", label="Perfect Calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add sample count annotations
    for i, (bar, count) in enumerate(zip(bars, samples_per_bin)):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'n={count}', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.legend()
    plt.subplots_adjust(top=0.92)  # Add margin at top to prevent overlap with title
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved reliability diagram to {save_path}")
    if show:
        plt.show()
    plt.close()
