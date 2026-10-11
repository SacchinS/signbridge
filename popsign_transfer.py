"""Does PopSign (Kaggle asl-signs) data help recognize my own signing?

Uses only the signs both sets share. All tests are on my recorded takes,
holding out whole sessions:
  A. PopSign only: train on PopSign, test on all my takes (no data from me).
  B. Mine only: train on my other session(s) (baseline, as in train_pilot.py).
  C. Mixed: PopSign + my other session(s), trained together.
  D. Pretrain + fine-tune: train on PopSign, then fine-tune on my other session(s).
  E/F. Few-shot: like B and D but with only --shots of my takes per sign.
--trim cuts my takes to the frames where a hand is visible, since PopSign clips
are trimmed to the sign while my takes are fixed 2 s windows.

Needs fetch_popsign.py run first for the shared signs.
Run (from the repo root, inside .venv):
    python popsign_transfer.py [--trim] [--seeds 3] [--shots 3]
"""

import argparse
import os

import numpy as np
import pandas as pd

from train_pilot import featurize, load_takes, predict, report, train

POPSIGN_DIR = "data/popsign"
# my label -> PopSign label
SHARED = {"HELLO": "hello", "YES": "yes", "NO": "no", "THANK-YOU": "thankyou"}


def load_popsign(path):
    """One PopSign parquet as a take dict in my recording format."""
    d = pd.read_parquet(os.path.join(POPSIGN_DIR, path))
    frames = np.sort(d.frame.unique())
    out = {}
    for kind, n, key in (("pose", 33, "pose"), ("left_hand", 21, "left"), ("right_hand", 21, "right")):
        part = d[d.type == kind].sort_values(["frame", "landmark_index"])
        out[key] = part[["x", "y", "z"]].to_numpy(np.float32).reshape(len(frames), n, 3)
    out["pose"] = np.concatenate([out["pose"], np.full((len(frames), 33, 1), np.nan, np.float32)], axis=2)
    out["t"] = (frames - frames[0]) / 30.0
    return out


def trim(take, pad=2):
    """Keep only the frames where a hand is visible, plus a little padding."""
    hand = ~(np.isnan(take["left"]).all((1, 2)) & np.isnan(take["right"]).all((1, 2)))
    if hand.sum() < 2:
        return take
    idx = np.where(hand)[0]
    lo, hi = max(idx[0] - pad, 0), min(idx[-1] + pad + 1, len(hand))
    return {k: v[lo:hi] for k, v in take.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trim", action="store_true", help="trim my takes to hand-visible frames")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--shots", type=int, default=3, help="my takes per sign in the few-shot runs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--frames", type=int, default=32)
    args = p.parse_args()

    signs = sorted(SHARED)
    index = pd.read_csv(os.path.join(POPSIGN_DIR, "subset.csv"))
    index = index[index.sign.isin(SHARED.values())]
    to_mine = {v: k for k, v in SHARED.items()}
    pop_x = np.stack([featurize(load_popsign(path), args.frames) for path in index.path])
    pop_y = np.array([signs.index(to_mine[s]) for s in index.sign])

    mine = load_takes(signs)
    my_x = np.stack([featurize(trim(t) if args.trim else t, args.frames) for t in mine])
    my_y = np.array([signs.index(t["sign"]) for t in mine])
    sess = np.array([t["session"] for t in mine])
    sessions = sorted(set(sess))
    print(f"PopSign {len(pop_y)} sequences; mine {len(my_y)} takes, sessions {sessions}; trim={args.trim}")

    results = {k: ([], []) for k in "ABCDEF"}

    def add(key, y_true, y_pred):
        results[key][0].append(y_true)
        results[key][1].append(y_pred)

    for seed in range(args.seeds):
        pretrained = train(pop_x, pop_y, len(signs), args.epochs, seed)
        add("A", my_y, predict(pretrained, my_x))
        rng = np.random.default_rng(seed)
        for held in sessions:
            test, tr = sess == held, sess != held
            add("B", my_y[test], predict(train(my_x[tr], my_y[tr], len(signs), args.epochs, seed), my_x[test]))
            mixed = train(np.concatenate([pop_x, my_x[tr]]), np.concatenate([pop_y, my_y[tr]]), len(signs), args.epochs, seed)
            add("C", my_y[test], predict(mixed, my_x[test]))
            ft = train(pop_x, pop_y, len(signs), args.epochs, seed)
            ft = train(my_x[tr], my_y[tr], len(signs), args.epochs // 2, seed, model=ft, lr=1e-3)
            add("D", my_y[test], predict(ft, my_x[test]))

            few = np.concatenate([rng.choice(np.where(tr & (my_y == c))[0], args.shots, replace=False) for c in range(len(signs))])
            add("E", my_y[test], predict(train(my_x[few], my_y[few], len(signs), args.epochs * 3, seed), my_x[test]))
            ft = train(pop_x, pop_y, len(signs), args.epochs, seed)
            ft = train(my_x[few], my_y[few], len(signs), args.epochs, seed, model=ft, lr=1e-3)
            add("F", my_y[test], predict(ft, my_x[test]))

    names = {"A": "PopSign only (zero takes of mine)", "B": "mine only", "C": "PopSign + mine, mixed",
             "D": "PopSign pretrain, fine-tune on mine", "E": f"mine only, {args.shots} takes/sign",
             "F": f"PopSign pretrain, fine-tune on {args.shots} takes/sign"}
    for k, (t, pr) in results.items():
        report(signs, np.concatenate(t), np.concatenate(pr), f"{k}. {names[k]} ({args.seeds} seeds)")
    print("\nSUMMARY")
    for k, (t, pr) in results.items():
        print(f"  {k}. {names[k]:48s} {np.mean(np.concatenate(t) == np.concatenate(pr)):6.1%}")


if __name__ == "__main__":
    main()
