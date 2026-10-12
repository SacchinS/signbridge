"""Replay saved sign-sequence attempts (data/sequences/, from segment_live.py) through the segmenter offline.

Scores the same way as the live script mode, so segmenter settings can be compared without re-recording.
Treat results as a guide: tuning on the same few attempts that are scored overfits.

Run (from the repo root, inside .venv):
    python segment_replay.py [--model models/popsign.pt] [--still 0.5] [--gap 0.3] [--verbose]
"""

import argparse
import glob

import numpy as np

from segment_live import OTHER, OUT_DIR
from signbridge.features import Classifier
from signbridge.segment import Segmenter


def replay(path, classifier, seg_args, min_conf):
    d = np.load(path)
    seg = Segmenter(**seg_args)
    glosses, phrases = [], []
    for f in zip(d["t"], d["pose"], d["left"], d["right"]):
        for event in seg.update(f):
            if event[0] == "sign":
                label, prob = classifier.predict(event[1])
                if label != OTHER:
                    glosses.append(label if prob >= min_conf else "?")
            elif event[0] == "phrase":
                phrases.append(glosses)
                glosses = []
    if glosses:  # stream ended before the phrase timer did
        phrases.append(glosses)
    got = [g for p in phrases for g in p]
    return list(d["expected"]), got, len(phrases)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/popsign.npz")
    p.add_argument("--glob", default=f"{OUT_DIR}/*.npz")
    p.add_argument("--below", type=float, default=1.0)
    p.add_argument("--still", type=float, default=0.5)
    p.add_argument("--still-speed", type=float, default=0.008)
    p.add_argument("--gap", type=float, default=0.3)
    p.add_argument("--min-conf", type=float, default=0.6)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    classifier = Classifier(args.model)
    seg_args = {"below": args.below, "still": args.still, "still_speed": args.still_speed, "gap": args.gap}
    rows = []
    for path in sorted(glob.glob(args.glob)):
        expected, got, n_phrases = replay(path, classifier, seg_args, args.min_conf)
        if expected:
            rows.append((path, expected, got, n_phrases))
    exact = sum(e == g for _, e, g, _ in rows)
    count_ok = sum(len(e) == len(g) for _, e, g, _ in rows)
    one_phrase = sum(n == 1 for *_, n in rows)
    print(f"{len(rows)} attempts. Exact: {exact}/{len(rows)}. Right sign count: {count_ok}/{len(rows)}. "
          f"Ended as one phrase: {one_phrase}/{len(rows)}.  ({seg_args})")
    for path, e, g, n in rows:
        if args.verbose or e != g:
            print(f"  {path[-7:-4]} expected {' '.join(e):20s} got {' '.join(g) or '(nothing)':28s} phrases={n}")


if __name__ == "__main__":
    main()
