"""Task 2 step 3: train a tiny Conv1D on recorded landmark takes and report accuracy.

Evaluation holds out whole sessions (leave-one-session-out): for each session,
train on all the others and test on it. Reports per-sign accuracy and the
most common confusions. With a single session, --sanity runs a random split
instead, which only checks the pipeline and overstates real accuracy.

Run (from the repo root, inside .venv):
    python train_pilot.py [--epochs 60] [--frames 32] [--save models/pilot.pt]
    python train_pilot.py --sanity      # one session only: random 70/30 split
"""

import argparse
import collections
import glob
import os
import random

import numpy as np
import torch
from torch import nn

from signbridge.features import TinyConv, featurize

DATA_DIR = "data/landmarks"


def load_takes(signs=None):
    takes = []
    for sign_dir in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
        sign = os.path.basename(sign_dir)
        if signs and sign not in signs:
            continue
        for path in sorted(glob.glob(os.path.join(sign_dir, "*.npz"))):
            signer, session = os.path.basename(path).split("_")[:2]
            d = np.load(path)
            takes.append({"sign": sign, "session": f"{signer}_{session}", "path": path,
                          "pose": d["pose"], "left": d["left"], "right": d["right"], "t": d["t"]})
    return takes


def augment(x):
    """x: (frames, features). Random scale, shift and time stretch; keeps handedness."""
    x = x.copy()
    n_mask = 2
    coords = x[:, :-n_mask]
    coords *= np.random.uniform(0.9, 1.1)
    coords += np.random.normal(0, 0.03, size=(1, coords.shape[1]))
    x[:, :-n_mask] = coords
    frames = len(x)
    start, end = sorted(np.random.uniform(0, 0.15, 2) * [1, -1] + [0, frames - 1])
    grid = np.linspace(start, end, frames)
    return np.stack([np.interp(grid, np.arange(frames), x[:, j]) for j in range(x.shape[1])], axis=1).astype(np.float32)


def train(x_train, y_train, n_classes, epochs, seed, model=None, lr=3e-3):
    """Train a new TinyConv, or keep training `model` (for fine-tuning)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if model is None:
        model = TinyConv(x_train.shape[2], n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        order = np.random.permutation(len(x_train))
        for i in range(0, len(order), 16):
            b = order[i:i + 16]
            xb = torch.from_numpy(np.stack([augment(x_train[j]) for j in b]))
            yb = torch.from_numpy(y_train[b])
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def predict(model, x):
    with torch.no_grad():
        return model(torch.from_numpy(x)).argmax(1).numpy()


def report(signs, y_true, y_pred, title):
    print(f"\n== {title}: {np.mean(y_true == y_pred):.1%} ({(y_true == y_pred).sum()}/{len(y_true)})")
    for i, s in enumerate(signs):
        m = y_true == i
        if m.any():
            print(f"  {s:10s} {np.mean(y_pred[m] == i):6.1%}  ({(y_pred[m] == i).sum()}/{m.sum()})")
    confusions = collections.Counter((signs[t], signs[p]) for t, p in zip(y_true, y_pred) if t != p)
    if confusions:
        print("  confusions (true -> predicted):")
        for (t, p), n in confusions.most_common(6):
            print(f"    {t} -> {p}: {n}")
    print("  confusion matrix (rows = true, cols = predicted):")
    print("       " + "".join(f"{s[:5]:>7s}" for s in signs))
    for i, s in enumerate(signs):
        print(f"  {s[:5]:5s}" + "".join(f"{np.sum((y_true == i) & (y_pred == j)):7d}" for j in range(len(signs))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sanity", action="store_true", help="random 70/30 split instead of held-out sessions")
    p.add_argument("--save", help="after evaluating, train on all data and save weights here")
    args = p.parse_args()

    takes = load_takes()
    signs = sorted({t["sign"] for t in takes})
    sessions = sorted({t["session"] for t in takes})
    x = np.stack([featurize(t, args.frames) for t in takes])
    y = np.array([signs.index(t["sign"]) for t in takes])
    sess = np.array([t["session"] for t in takes])
    print(f"{len(takes)} takes, signs {signs}, sessions {sessions}, features {x.shape[1:]}")

    if args.sanity or len(sessions) < 2:
        if not args.sanity:
            print("Only one session recorded; running the random-split sanity check instead.")
        rng = np.random.default_rng(args.seed)
        test = np.zeros(len(y), bool)
        for c in range(len(signs)):
            idx = np.where(y == c)[0]
            test[rng.choice(idx, max(1, int(len(idx) * 0.3)), replace=False)] = True
        model = train(x[~test], y[~test], len(signs), args.epochs, args.seed)
        report(signs, y[test], predict(model, x[test]), "SANITY random split (same session; overstates accuracy)")
    else:
        all_true, all_pred = [], []
        for held in sessions:
            test = sess == held
            model = train(x[~test], y[~test], len(signs), args.epochs, args.seed)
            pred = predict(model, x[test])
            report(signs, y[test], pred, f"held-out session {held}")
            all_true.append(y[test])
            all_pred.append(pred)
        report(signs, np.concatenate(all_true), np.concatenate(all_pred), "ALL held-out sessions")

    if args.save:
        model = train(x, y, len(signs), args.epochs, args.seed)
        torch.save({"state_dict": model.state_dict(), "signs": signs, "frames": args.frames,
                    "features": x.shape[2]}, args.save)
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
