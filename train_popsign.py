"""Train a model on PopSign data only (none of my recordings) and save it for live use.

Uses every downloaded PopSign sequence (data/popsign/subset.csv, from fetch_popsign.py) for the signs in
SHARED, labelled with my sign names so live_classify.py and record_signs.py can load it. Before saving,
it reports accuracy on all my recorded takes, which the model never trained on.

Run (from the repo root, inside .venv):
    python train_popsign.py [--save models/popsign.pt] [--seeds 3]
    python live_classify.py --model models/popsign.pt
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from popsign_transfer import POPSIGN_DIR, SHARED, load_popsign
from train_pilot import featurize, load_takes, predict, report, train


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save", default="models/popsign.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--per-sign", type=int, help="cap per sign; default = the smallest sign's count, so classes are balanced")
    p.add_argument("--seeds", type=int, default=3, help="seeds to evaluate; the best is not picked, seed 0 is saved")
    args = p.parse_args()

    signs = sorted(SHARED)
    to_mine = {v: k for k, v in SHARED.items()}
    index = pd.read_csv(os.path.join(POPSIGN_DIR, "subset.csv"))
    index = index[index.sign.isin(SHARED.values())]
    cap = args.per_sign or index.sign.value_counts().min()
    index = index.groupby("sign").head(cap)
    x = np.stack([featurize(load_popsign(path), args.frames) for path in index.path])
    y = np.array([signs.index(to_mine[s]) for s in index.sign])
    print(f"PopSign: {len(y)} sequences, {index.participant_id.nunique()} signers, "
          + ", ".join(f"{s} {np.sum(y == i)}" for i, s in enumerate(signs)))

    mine = load_takes(signs)
    my_x = np.stack([featurize(t, args.frames) for t in mine])
    my_y = np.array([signs.index(t["sign"]) for t in mine])

    saved = None
    for seed in range(args.seeds):
        model = train(x, y, len(signs), args.epochs, seed)
        report(signs, my_y, predict(model, my_x), f"seed {seed}: PopSign-only model on my {len(my_y)} takes")
        if saved is None:
            saved = model
    torch.save({"state_dict": saved.state_dict(), "signs": signs, "frames": args.frames, "features": x.shape[2]}, args.save)
    print(f"\nsaved {args.save} (seed 0)")


if __name__ == "__main__":
    main()
