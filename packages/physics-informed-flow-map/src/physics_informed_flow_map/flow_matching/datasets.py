"""Dataset abstraction + registry. Swapping datasets = changing one config key."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset


@dataclass
class DatasetSpec:
    shape: tuple[int, ...]
    num_classes: int | None
    make_dataset: Callable[[], Dataset]
    visualize: Callable[[Tensor, Path], None]


def _make_gaussians(
    n_samples: int = 100_000,
    n_modes: int = 8,
    radius: float = 4.0,
    std: float = 0.5,
    seed: int = 0,
) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    angles = 2 * math.pi * torch.arange(n_modes) / n_modes
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    idx = torch.randint(0, n_modes, (n_samples,), generator=g)
    x = centers[idx] + std * torch.randn(n_samples, 2, generator=g)
    labels = torch.zeros(n_samples, dtype=torch.long)
    return TensorDataset(x.float(), labels)


def _make_mnist(data_dir: str = "data", image_size: int = 32) -> Dataset:
    transform = T.Compose(
        [T.Resize(image_size), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    )
    return cast(
        Dataset,
        torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        ),
    )


def _viz_scatter(samples: Tensor, path: Path) -> None:
    s = samples.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(s[:, 0], s[:, 1], s=3, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title("generated samples")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _viz_grid(samples: Tensor, path: Path) -> None:
    s = ((samples.detach().cpu().clamp(-1, 1) + 1) / 2)
    n = min(64, len(s))
    ncols = 8
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            ax.imshow(s[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


DATASETS: dict[str, DatasetSpec] = {
    "gaussians": DatasetSpec(
        shape=(2,), num_classes=None, make_dataset=_make_gaussians, visualize=_viz_scatter
    ),
    "mnist": DatasetSpec(
        shape=(1, 32, 32), num_classes=10, make_dataset=_make_mnist, visualize=_viz_grid
    ),
}
