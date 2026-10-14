"""Export the in-app sign guide: one clean take per sign as a skeleton the app animates.

For each sign the model knows, picks the recorded take the classifier is most confident about, trims it to the
signing, normalizes it to the shoulders (shoulder midpoint at 0,0, shoulder width 1), mirrors it to match the
mirrored live preview, and writes macos/Assets/signs.json with a short description of the sign. Only landmarks
are exported, never video.

Run (from the repo root, inside .venv):
    python export_sign_guide.py
"""

import glob
import json
import os

import numpy as np

from signbridge.features import Classifier
from signbridge.segment import L_SHOULDER, R_SHOULDER

TAKES_DIR = "data/landmarks"
OUT = "macos/Assets/signs.json"
FACE = [0, 2, 5, 9, 10]  # nose, eyes, mouth corners
ARMS = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
MAX_GAP = 4  # hand dropouts up to this many frames are filled; longer ones mean the hand is down

GUIDE = {
    "HELLO": ("Like a salute", "Flat hand, fingers together. Start at the side of your forehead and move it "
              "out and away from you."),
    "YES": ("A nodding fist", "Make a fist and bob it up and down at the wrist, like a head nodding yes."),
    "NO": ("Fingers snap shut", "Hold up your index and middle fingers with your thumb, then snap them "
           "closed onto the thumb. Once or twice."),
    "THANK-YOU": ("From your chin, outward", "Flat hand, fingertips at your chin. Move the hand forward and "
                  "down toward the person you're thanking."),
}


def short_gaps_filled(hand):
    """(T, 21, 2) with NaN frames; fills gaps of up to MAX_GAP frames between seen frames."""
    seen = ~np.isnan(hand).reshape(len(hand), -1).all(1)
    out = hand.copy()
    idx = np.where(seen)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= MAX_GAP + 1:
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                out[t] = (1 - w) * hand[a] + w * hand[b]
    return out


def smooth(x, k=3):
    """Moving average over time that leaves NaN frames alone."""
    out = x.copy()
    for t in range(len(x)):
        window = x[max(0, t - k // 2): t + k // 2 + 1]
        if not np.isnan(x[t]).all():
            out[t] = np.nanmean(window, axis=0)
    return out


def export_take(take):
    pose, left, right = take["pose"][..., :2], take["left"][..., :2], take["right"][..., :2]
    shoulders = pose[:, [L_SHOULDER, R_SHOULDER]]
    center = np.nanmedian(shoulders.mean(1), axis=0)
    width = np.nanmedian(np.abs(shoulders[:, 0, 0] - shoulders[:, 1, 0]))

    def norm(p):
        q = (p - center) / width
        q[..., 0] *= -1  # mirror, like the live preview
        return q

    hands = [smooth(norm(short_gaps_filled(h))) for h in (left, right)]
    body = smooth(norm(pose[:, FACE + ARMS]))
    visible = [~np.isnan(h).reshape(len(h), -1).all(1) for h in hands]
    any_hand = visible[0] | visible[1]
    first, last = np.argmax(any_hand), len(any_hand) - np.argmax(any_hand[::-1])
    frames = []
    for t in range(max(0, first - 2), min(len(any_hand), last + 2)):
        frames.append({
            "body": np.round(np.nan_to_num(body[t]), 2).tolist(),
            "hands": [np.round(h[t], 2).tolist() if v[t] else None for h, v in zip(hands, visible)],
        })
    # the hand that moves most gets the motion arrow
    with np.errstate(all="ignore"):  # a hand that never shows has no travel
        travel = [np.nansum(np.linalg.norm(np.diff(np.nanmean(h, 1), axis=0), axis=1)) for h in hands]
    fps = (len(take["t"]) - 1) / float(take["t"][-1] - take["t"][0])
    return frames, int(np.argmax(travel)), round(fps, 1)


def main():
    clf = Classifier("models/popsign.npz")
    signs = []
    for name in clf.signs:
        name = str(name)
        scored = []
        for path in sorted(glob.glob(os.path.join(TAKES_DIR, name, "*.npz"))):
            take = dict(np.load(path))
            label, prob = clf.predict(take)
            if label == name:
                scored.append((prob, path, take))
        if not scored:
            print(f"{name}: no correctly classified take, skipped")
            continue
        prob, path, take = max(scored, key=lambda s: s[0])
        frames, mover, fps = export_take(take)
        title, how = GUIDE[name]
        signs.append({"name": name, "title": title, "how": how, "fps": fps, "mover": mover, "frames": frames})
        print(f"{name}: {os.path.basename(path)} ({prob:.0%}), {len(frames)} frames")
    with open(OUT, "w") as f:
        json.dump({"signs": signs}, f, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    main()
