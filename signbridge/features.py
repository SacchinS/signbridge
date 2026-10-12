"""Landmark features and the sign classifier, shared by training scripts and the app."""

import numpy as np
import torch
from torch import nn

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


class Classifier:
    """A trained checkpoint (from train_pilot.py or train_popsign.py); predict() returns (label, prob)."""

    def __init__(self, path):
        ckpt = torch.load(path, weights_only=False)
        self.signs, self.frames = ckpt["signs"], ckpt["frames"]
        self.model = TinyConv(ckpt["features"], len(self.signs))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(self, take):
        x = torch.from_numpy(featurize(take, self.frames)[None])
        with torch.no_grad():
            probs = torch.softmax(self.model(x), 1)[0].numpy()
        return self.signs[int(probs.argmax())], float(probs.max())

