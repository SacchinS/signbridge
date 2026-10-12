"""The PyTorch sign model, for training only. The app runs exported weights with numpy (features.Classifier)."""

import numpy as np
import torch
from torch import nn


class TinyConv(nn.Module):
    def __init__(self, n_features, n_classes, width=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, n_classes),
        )

    def forward(self, x):  # x: (batch, frames, features)
        return self.net(x.transpose(1, 2))


def checkpoint_arrays(path):
    """A training checkpoint (.pt) as the flat arrays Classifier uses."""
    ckpt = torch.load(path, weights_only=False)
    arrays = {k: v.numpy() for k, v in ckpt["state_dict"].items() if v.dtype.is_floating_point}
    arrays["signs"] = np.array(ckpt["signs"])
    arrays["frames"] = np.array(ckpt["frames"])
    return arrays


def export(pt_path, npz_path=None):
    """Write the .npz the app loads, next to the .pt by default. Returns its path."""
    npz_path = npz_path or pt_path[: -len(".pt")] + ".npz"
    np.savez(npz_path, **checkpoint_arrays(pt_path))
    return npz_path


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        print("wrote", export(path))
