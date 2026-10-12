"""Landmark features and the sign classifier, shared by training scripts and the app.

Numpy only, so the app doesn't ship PyTorch. Training lives in signbridge/model.py.
"""

import numpy as np

UPPER_POSE = list(range(0, 23))  # face points, shoulders, elbows, wrists, hand points; no hips or legs
L_SHOULDER, R_SHOULDER = 11, 12


def fill_gaps(x):
    """Fill NaN frames (T, ...) from the nearest earlier frame, then the nearest later one, else 0."""
    x = x.copy()
    missing = np.isnan(x).reshape(len(x), -1).all(1)
    if missing.all():
        return np.zeros_like(x), missing
    idx = np.where(~missing, np.arange(len(x)), -1)
    idx = np.maximum.accumulate(idx)
    first = np.argmax(~missing)
    idx[idx < 0] = first
    return x[idx], missing


def featurize(take, frames):
    """Normalize to the shoulders and resample to a fixed number of frames. Returns (frames, features)."""
    pose = take["pose"][:, UPPER_POSE, :3]
    pose, _ = fill_gaps(pose)
    left, left_missing = fill_gaps(take["left"])
    right, right_missing = fill_gaps(take["right"])

    shoulders = take["pose"][:, [L_SHOULDER, R_SHOULDER], :3]
    center = np.nanmean(shoulders, axis=(0, 1))
    width = np.nanmedian(np.linalg.norm(shoulders[:, 0, :2] - shoulders[:, 1, :2], axis=1))
    width = width if np.isfinite(width) and width > 1e-3 else 0.2
    norm = lambda a: (a - center) / width

    feats = np.concatenate([
        norm(pose).reshape(len(pose), -1),
        np.where(left_missing[:, None], 0, norm(left).reshape(len(left), -1)),
        np.where(right_missing[:, None], 0, norm(right).reshape(len(right), -1)),
        left_missing[:, None].astype(np.float32),
        right_missing[:, None].astype(np.float32),
    ], axis=1)
    feats = np.nan_to_num(feats)

    t = take["t"] if len(take["t"]) == len(feats) else np.linspace(0, 1, len(feats))
    grid = np.linspace(t[0], t[-1], frames)
    out = np.stack([np.interp(grid, t, feats[:, j]) for j in range(feats.shape[1])], axis=1)
    return out.astype(np.float32)


def to_arrays(result):
    """One tracking result as (pose (33, 4), left (21, 3), right (21, 3)); NaN where missing."""
    pose = np.full((33, 4), np.nan, dtype=np.float32)
    if result.pose_landmarks:
        pose[:] = [(lm.x, lm.y, lm.z, lm.visibility if lm.visibility is not None else np.nan) for lm in result.pose_landmarks]
    hands = []
    for lms in (result.left_hand_landmarks, result.right_hand_landmarks):
        arr = np.full((21, 3), np.nan, dtype=np.float32)
        if lms:
            arr[:] = [(lm.x, lm.y, lm.z) for lm in lms]
        hands.append(arr)
    return pose, hands[0], hands[1]


def _conv1d(x, w, b):
    """x (T, C_in), w (C_out, C_in, K) with 'same' padding -> (T, C_out)."""
    k = w.shape[2]
    xp = np.pad(x, ((k // 2, k // 2), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(xp, k, axis=0)  # (T, C_in, K)
    return np.einsum("tck,ock->to", windows, w) + b


def _batchnorm(x, p, eps=1e-5):
    return (x - p["running_mean"]) / np.sqrt(p["running_var"] + eps) * p["weight"] + p["bias"]


class Classifier:
    """A trained model; predict() returns (label, prob). Loads the .npz weights written by
    signbridge.model.export (or a training .pt checkpoint, which needs PyTorch)."""

    def __init__(self, path):
        if path.endswith(".pt"):
            from signbridge.model import checkpoint_arrays

            arrays = checkpoint_arrays(path)
        else:
            arrays = dict(np.load(path, allow_pickle=False))
        self.signs = [str(s) for s in arrays.pop("signs")]
        self.frames = int(arrays.pop("frames"))
        self.p = arrays

    def _layer(self, prefix):
        return {k[len(prefix):]: v for k, v in self.p.items() if k.startswith(prefix)}

    def logits(self, x):
        """x (frames, features) -> class scores. Same maths as TinyConv in eval mode."""
        for conv, bn in (("net.0.", "net.1."), ("net.4.", "net.5.")):
            x = _conv1d(x, self.p[conv + "weight"], self.p[conv + "bias"])
            x = np.maximum(_batchnorm(x, self._layer(bn)), 0)
        return x.mean(0) @ self.p["net.10.weight"].T + self.p["net.10.bias"]

    def predict(self, take):
        z = self.logits(featurize(take, self.frames).astype(np.float64))
        probs = np.exp(z - z.max())
        probs /= probs.sum()
        return self.signs[int(probs.argmax())], float(probs.max())
