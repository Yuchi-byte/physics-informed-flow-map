"""
Assessing whether the generative priors are overfitting. The conventionary validation loss assessment is not sufficient here.
The implementatin here:
1. brutally checking how similar the output image is to every training image.
2. the trianing image that is the most similar is then visualised side-by-side with the output image. If the two images don't look alike, that suggests the prior isn't overfititng.
"""

from __future__ import annotations

import math
import torch
from torch.utils.data import DataLoader


def assess_overfit(
    generated_output: torch.Tensor, train_dataloader: DataLoader
) -> tuple[torch.Tensor, float]:
    generated_B = generated_output.shape[0]
    min_mse = (torch.ones(generated_B) * math.inf).to(generated_output.device)
    min_mse_image = torch.rand_like(generated_output)
    for X_batch_train, _ in train_dataloader:
        X_batch_train = X_batch_train.to(generated_output.device)
        train_B = len(X_batch_train)

        mse = ((generated_output[:, None] - X_batch_train[None]) ** 2).mean(
            dim=(2, 3, 4)
        )
        assert mse.shape == (generated_B, train_B)
        vals, idx = mse.min(dim=1)
        assert vals.shape == (generated_B,)
        better = vals < min_mse
        assert better.shape == (generated_B,)

        min_mse_image = torch.where(
            better[:, None, None, None], X_batch_train[idx], min_mse_image
        )
        assert min_mse_image.shape == generated_output.shape

        min_mse = torch.where(better, vals, min_mse)
        assert min_mse.shape == (generated_B,)

    return min_mse_image, min_mse
