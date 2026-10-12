"""Train a model on PopSign data only (none of my recordings) and save it for live use.

Uses the downloaded PopSign sequences (data/popsign/subset.csv, from fetch_popsign.py) for the signs in
SHARED, labelled with my sign names so live_classify.py and record_signs.py can load it. Every other
downloaded sign becomes one OTHER class ("not a command"), so movements that aren't commands can be rejected.
A few other signs are held out of training to measure how often an unseen sign is rejected.
Before saving, it reports accuracy on all my recorded takes, which the model never trained on.

Run (from the repo root, inside .venv):
    python train_popsign.py [--save models/popsign.pt] [--seeds 3] [--no-other]
    python live_classify.py --model models/popsign.pt
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from popsign_transfer import POPSIGN_DIR, SHARED, load_popsign
from signbridge.features import featurize
from signbridge.model import export
from train_pilot import load_takes, predict, report, train

OTHER = "OTHER"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save", default="models/popsign.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--per-sign", type=int, help="cap per sign; default = the smallest sign's count, so classes are balanced")
    p.add_argument("--no-other", action="store_true", help="commands only, no OTHER class")
    p.add_argument("--other-ratio", type=float, default=2.0, help="OTHER examples per command-sign example cap")
    p.add_argument("--holdout", type=int, default=8, help="other signs kept out of training to test rejection")
    p.add_argument("--seeds", type=int, default=3, help="seeds to evaluate; the best is not picked, seed 0 is saved")
    args = p.parse_args()

    to_mine = {v: k for k, v in SHARED.items()}
    full = pd.read_csv(os.path.join(POPSIGN_DIR, "subset.csv"))
    full = full[[os.path.exists(os.path.join(POPSIGN_DIR, path)) for path in full.path]]
    index = full[full.sign.isin(SHARED.values())]
    cap = args.per_sign or index.sign.value_counts().min()
    index = index.groupby("sign").head(cap)
    others = full[~full.sign.isin(SHARED.values())]
    other_signs = sorted(others.sign.unique())
    held_signs = set(np.random.default_rng(0).choice(other_signs, min(args.holdout, len(other_signs)), replace=False))
    held = others[others.sign.isin(held_signs)]
    if args.no_other or others.empty:
        signs = sorted(SHARED)
    else:
        signs = sorted(SHARED) + [OTHER]
        train_others = others[~others.sign.isin(held_signs)]
        n = int(cap * args.other_ratio)
        train_others = train_others.sample(min(n, len(train_others)), random_state=0)
        index = pd.concat([index, train_others])
    x = np.stack([featurize(load_popsign(path), args.frames) for path in index.path])
    y = np.array([signs.index(to_mine.get(s, OTHER)) for s in index.sign])
    print(f"PopSign: {len(y)} sequences, {index.participant_id.nunique()} signers, "
          + ", ".join(f"{s} {np.sum(y == i)}" for i, s in enumerate(signs)))
    if OTHER in signs:
        print(f"OTHER: {index[~index.sign.isin(SHARED.values())].sign.nunique()} signs in training; "
              f"held out for the rejection test: {len(held)} sequences of {', '.join(sorted(held_signs))}")
        held_x = np.stack([featurize(load_popsign(path), args.frames) for path in held.path])

    mine = load_takes(signs)
    my_x = np.stack([featurize(t, args.frames) for t in mine])
    my_y = np.array([signs.index(t["sign"]) for t in mine])

    saved = None
    for seed in range(args.seeds):
        model = train(x, y, len(signs), args.epochs, seed)
        report(signs, my_y, predict(model, my_x), f"seed {seed}: PopSign-only model on my {len(my_y)} takes")
        if OTHER in signs:
            rejected = np.mean(predict(model, held_x) == signs.index(OTHER))
            print(f"  unseen PopSign signs rejected as OTHER: {rejected:.0%} ({len(held_x)} sequences)")
        if saved is None:
            saved = model
    torch.save({"state_dict": saved.state_dict(), "signs": signs, "frames": args.frames, "features": x.shape[2]}, args.save)
    export(args.save)  # .npz for the app
    print(f"\nsaved {args.save} (seed 0)")


if __name__ == "__main__":
    main()
