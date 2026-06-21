"""
PyTorch Dataset for the OpenFWI velocity map files.

Each .npy file contains a batch of 500 samples shaped (500, 1, 70, 70).
Files are memory-mapped so the full dataset does not need to fit in RAM.

Expected directory layout (after downloading):
    data_root/
        FlatVel-A/    (or any OpenFWI family folder)
            data/         <- seismic .npy files  (ignored here)
            model/        <- velocity map .npy files  (used here)
                vel_1.npy
                vel_2.npy
                ...
        CurveVel-A/
            model/
                ...

Usage:
    dataset = OpenFWIVelocityDataset("/workspace/data/openfwi", resolution=256)
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class OpenFWIVelocityDataset(Dataset):
    """
    Wraps OpenFWI velocity map .npy files.

    Args:
        root: root folder containing the downloaded OpenFWI families.
        families: list of family folder names to include. None = all found.
        resolution: spatial size to resize each 70×70 map to. None = keep 70×70.
        normalize: if True, linearly scale pixel values to [-1, 1] using the
                   dataset's empirical range [1500, 4500] m/s.
    """

    VMIN = 1500.0
    VMAX = 4500.0

    def __init__(self, root: str, families=None, resolution=None, normalize=True):
        self.root = root
        self.resolution = resolution
        self.normalize = normalize

        pattern = os.path.join(root, "**", "model", "*.npy")
        all_files = sorted(glob.glob(pattern, recursive=True))

        if families is not None:
            all_files = [f for f in all_files if any(fam in f for fam in families)]

        if not all_files:
            raise FileNotFoundError(
                f"No velocity .npy files found under {root}.\n"
                "Make sure the files are under <family>/model/*.npy"
            )

        # Build index: list of (file_path, sample_index_within_file)
        self._index = []
        for path in all_files:
            arr = np.load(path, mmap_mode="r")  # shape (500, 1, 70, 70)
            n = arr.shape[0]
            self._index.extend((path, i) for i in range(n))

        # Cache open mmap handles to avoid re-opening on every __getitem__
        self._mmap_cache: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self._index)

    def _get_mmap(self, path: str) -> np.ndarray:
        if path not in self._mmap_cache:
            self._mmap_cache[path] = np.load(path, mmap_mode="r")
        return self._mmap_cache[path]

    def __getitem__(self, idx):
        path, sample_idx = self._index[idx]
        arr = self._get_mmap(path)
        # arr[sample_idx] shape: (1, 70, 70), dtype float32
        x = torch.from_numpy(arr[sample_idx].copy()).float()  # copy off mmap

        if self.resolution is not None and self.resolution != 70:
            x = TF.resize(
                x,
                [self.resolution, self.resolution],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )

        if self.normalize:
            # scale from [VMIN, VMAX] → [-1, 1]
            x = (x - self.VMIN) / (self.VMAX - self.VMIN)  # [0, 1]
            x = x * 2.0 - 1.0  # [-1, 1]

        # Return (image, dummy_label) to match ImageFolder convention
        return x, 0
