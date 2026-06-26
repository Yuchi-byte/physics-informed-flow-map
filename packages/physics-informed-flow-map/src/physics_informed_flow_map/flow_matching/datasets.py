"""Dataset configs (discriminated union). Swapping datasets = changing one group.

Each variant owns its build + metadata; the module-level ``_make_*``/``_viz_*``
helpers do the actual work and are delegated to.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from pydantic import Field
from torch import Tensor
from torch.utils.data import Dataset, Subset, TensorDataset

from physics_informed_flow_map.experiment import Config
from physics_informed_flow_map.flow_matching.openfwi import (
    OpenFWIVelocityDataset,
    viz_velocity,
)


def _make_gaussians(
    n_samples: int = 100_000,
    n_modes: int = 8,
    radius: float = 4.0,
    std: float = 0.5,
    seed: int = 0,
) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    angles = 2 * math.pi * torch.arange(n_modes) / n_modes
    centers = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)], dim=1
    )
    idx = torch.randint(0, n_modes, (n_samples,), generator=g)
    x = centers[idx] + std * torch.randn(n_samples, 2, generator=g)
    labels = torch.zeros(n_samples, dtype=torch.long)
    return TensorDataset(x.float(), labels)


def _make_mnist(
    data_dir: str = "data", image_size: int = 32, train: bool = True
) -> Dataset:
    transform = T.Compose(
        [T.Resize(image_size), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    )
    return cast(
        Dataset,
        torchvision.datasets.MNIST(
            root=data_dir, train=train, download=True, transform=transform
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


def _viz_grid(samples: Tensor, path: Path, *, ncols: int = 8) -> None:
    s = (samples.detach().cpu().clamp(-1, 1) + 1) / 2
    n = min(ncols * 8, len(s))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            ax.imshow(s[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class GaussiansDatasetConfig(Config):
    """2D mixture-of-Gaussians toy dataset."""

    name: Literal["gaussians"] = "gaussians"
    n_modes: int = 8
    radius: float = 4.0
    std: float = 0.5
    n_samples: int = 100_000
    val_samples: int = 10_000

    @property
    def requires_download(self) -> bool:
        return False

    @property
    def shape(self) -> tuple[int, ...]:
        return (2,)

    @property
    def num_classes(self) -> int | None:
        return None

    def build(self) -> Dataset:
        return _make_gaussians(self.n_samples, self.n_modes, self.radius, self.std)

    def build_val(self) -> Dataset:
        return _make_gaussians(
            self.val_samples, self.n_modes, self.radius, self.std, seed=1
        )

    def visualize(self, samples: Tensor, path: Path, *, ncols: int = 8) -> None:
        _viz_scatter(samples, path)  # scatter ignores ncols


class MNISTDatasetConfig(Config):
    """MNIST digits, resized to a square and normalised to [-1, 1]."""

    name: Literal["mnist"] = "mnist"
    image_size: int = 32
    data_dir: str = "data"

    @property
    def requires_download(self) -> bool:
        return True

    @property
    def shape(self) -> tuple[int, ...]:
        return (1, self.image_size, self.image_size)

    @property
    def num_classes(self) -> int | None:
        return 10

    def build(self) -> Dataset:
        return _make_mnist(self.data_dir, self.image_size)

    def build_val(self) -> Dataset:
        return _make_mnist(self.data_dir, self.image_size, train=False)

    def visualize(self, samples: Tensor, path: Path, *, ncols: int = 8) -> None:
        _viz_grid(samples, path, ncols=ncols)


class OpenFWIDatasetConfig(Config):
    """OpenFWI subsurface velocity maps, normalised to [-1, 1]."""

    name: Literal["openfwi"] = "openfwi"
    data_dir: str = "data/openfwi"
    families: list[str] = ["FlatVel_A"]
    resolution: int = 64
    val_fraction: float = 0.1

    @property
    def requires_download(self) -> bool:
        return True

    @property
    def shape(self) -> tuple[int, ...]:
        return (1, self.resolution, self.resolution)

    @property
    def num_classes(self) -> int | None:
        return None

    def _split(self) -> tuple[OpenFWIVelocityDataset, list[int], list[int]]:
        full = OpenFWIVelocityDataset(
            Path(self.data_dir), self.families, self.resolution
        )
        n = len(full)
        n_val = max(1, int(self.val_fraction * n))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
        return full, perm[n_val:], perm[:n_val]

    def build(self) -> Dataset:
        full, train_idx, _ = self._split()
        return Subset(full, train_idx)

    def build_val(self) -> Dataset:
        full, _, val_idx = self._split()
        return Subset(full, val_idx)

    def visualize(self, samples: Tensor, path: Path, *, ncols: int = 8) -> None:
        viz_velocity(samples, path, ncols=ncols)


DatasetConfig = Annotated[
    GaussiansDatasetConfig | MNISTDatasetConfig | OpenFWIDatasetConfig,
    Field(discriminator="name"),
]


DATASETS: dict[str, DatasetConfig] = {
    "gaussians": GaussiansDatasetConfig(),
    "mnist": MNISTDatasetConfig(),
    "openfwi": OpenFWIDatasetConfig(),
}
