"""Live sign sequences: cut signs at the rest position, classify each, end the phrase on a long rest.

A sign starts when a hand rises above the rest line (drawn on the preview, a little below the shoulders)
and ends when the hands drop below it or leave the frame. Each cut clip is classified right away. After
--phrase-rest seconds at rest, the phrase (the list of signs) is finished and printed.

Scripted test: the preview prompts a sequence (e.g. "YES NO"); sign it with your hands dropping to rest
between signs, then rest until the phrase ends. Each attempt is scored and its landmark stream saved to
data/sequences/ so the segmenter can be tuned offline.

Run (from the repo root, inside .venv):
    python segment_live.py [--model models/popsign.pt]              # script mode, default sequences
    python segment_live.py --free                                    # no prompts, just recognize
    python segment_live.py --script "HELLO THANK-YOU" "YES NO" --reps 3

Keys in the preview: x = discard the last scored attempt, s = skip the current prompt, q = quit.
"""

import argparse
import collections
import datetime
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

from record_signs import Classifier, to_arrays
from spike_tracking import WINDOW, LatestFrame, draw, make_landmarker

L_SHOULDER, R_SHOULDER = 11, 12
DEFAULT_SCRIPT = ["HELLO", "YES NO", "HELLO THANK-YOU", "NO YES", "THANK-YOU HELLO", "YES THANK-YOU NO"]
OUT_DIR = "data/sequences"


def rest_line(pose, below):
    """Image y of the rest line: `below` shoulder-widths under the shoulders, or None without a pose."""
    s = pose[[L_SHOULDER, R_SHOULDER], :2]
    if np.isnan(s).any():
        return None
    return s[:, 1].mean() + below * abs(s[0, 0] - s[1, 0])


def hands_up(pose, left, right, below):
    """True when a tracked hand's wrist is above the rest line."""
    line = rest_line(pose, below)
    for hand in (left, right):
        if not np.isnan(hand).all() and (line is None or hand[0, 1] < line):
            return True
    return False


class Segmenter:
    """Feeds on one frame at a time; returns finished sign clips and phrase ends."""

    def __init__(self, below=1.0, start_frames=3, end_frames=5, pad=3, min_frames=8, max_seconds=4.0, phrase_rest=1.5):
        self.below, self.start_frames, self.end_frames = below, start_frames, end_frames
        self.pad, self.min_frames, self.max_seconds, self.phrase_rest = pad, min_frames, max_seconds, phrase_rest
        self.reset()

    def reset(self):
        self.active = False
        self.history = collections.deque(maxlen=self.pad + self.start_frames)
        self.clip, self.run, self.quiet = [], 0, 0
        self.last_end, self.signs_in_phrase = None, 0

    def update(self, frame):
        """frame = (t, pose, left, right). Returns a list of events: ("start",), ("sign", take), ("phrase",)."""
        t, pose, left, right = frame
        up = hands_up(pose, left, right, self.below)
        events = []
        if not self.active:
            self.history.append(frame)
            self.run = self.run + 1 if up else 0
            if self.run >= self.start_frames:
                self.active, self.clip, self.quiet = True, list(self.history), 0
                events.append(("start",))
            elif self.signs_in_phrase and t - self.last_end >= self.phrase_rest:
                self.signs_in_phrase = 0
                events.append(("phrase",))
            return events
        self.clip.append(frame)
        self.quiet = 0 if up else self.quiet + 1
        too_long = t - self.clip[0][0] >= self.max_seconds
        if self.quiet >= self.end_frames or too_long:
            clip = self.clip[: len(self.clip) - self.quiet + min(self.quiet, self.pad)]
            self.active, self.run = False, 0
            self.history.clear()
            self.last_end = t
            if len(clip) >= self.min_frames:
                self.signs_in_phrase += 1
                events.append(("sign", {
                    "pose": np.stack([f[1] for f in clip]), "left": np.stack([f[2] for f in clip]),
                    "right": np.stack([f[3] for f in clip]), "t": np.array([f[0] for f in clip]) - clip[0][0],
                }))
            elif not self.signs_in_phrase:
                self.last_end = None
        return events


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/popsign.pt")
    p.add_argument("--free", action="store_true", help="no prompts or scoring")
    p.add_argument("--script", nargs="+", default=DEFAULT_SCRIPT, help="sequences to prompt, signs separated by spaces")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--below", type=float, default=1.0, help="rest line, in shoulder widths below the shoulders")
    p.add_argument("--phrase-rest", type=float, default=1.5, help="seconds at rest that end a phrase")
    p.add_argument("--min-conf", type=float, default=0.6, help="below this, a sign is shown as '?'")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--no-save", action="store_true", help="don't save attempt landmark streams")
    args = p.parse_args()

    classifier = Classifier(args.model)
    prompts = [] if args.free else [s.upper().split() for s in args.script] * args.reps
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")
    grabber = LatestFrame(cap)
    landmarker = make_landmarker()
    seg = Segmenter(below=args.below, phrase_rest=args.phrase_rest)
    lock = threading.Lock()
    state = {"result": None, "line": None, "active": False, "glosses": [], "last_phrase": "", "prompt": 0}
    stream, bounds = [], []
    scored = []  # (expected, got, confidences)
    saved = []  # attempt file per scored phrase, so x can delete it
    stop = threading.Event()
    session = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    clock = time.perf_counter()

    def finish_phrase():
        glosses = list(state["glosses"])
        text = " ".join(g for g, _ in glosses)
        line = f"PHRASE: {text}"
        expected = None
        if prompts and state["prompt"] < len(prompts):
            expected = prompts[state["prompt"]]
            got = [g for g, _ in glosses]
            ok = got == expected
            scored.append((expected, got, [c for _, c in glosses]))
            line += f"   expected {' '.join(expected)}: {'OK' if ok else 'WRONG'}"
            state["prompt"] += 1
        print(line, flush=True)
        if not args.no_save and stream:
            os.makedirs(OUT_DIR, exist_ok=True)
            path = os.path.join(OUT_DIR, f"me_{session}_{len(saved) + 1:03d}.npz")
            np.savez_compressed(
                path, pose=np.stack([f[1] for f in stream]), left=np.stack([f[2] for f in stream]),
                right=np.stack([f[3] for f in stream]), t=np.array([f[0] for f in stream]),
                expected=np.array(expected or [], dtype=str), glosses=np.array([g for g, _ in glosses], dtype=str),
                bounds=np.array(bounds, dtype=float).reshape(-1, 2))
            saved.append(path)
        state["last_phrase"] = text
        state["glosses"] = []
        stream.clear()
        bounds.clear()

    def track():
        seq, last_ts = 0, -1
        start_t = None
        while not stop.is_set():
            seq, frame = grabber.read(seq)
            if frame is None:
                continue
            now = time.perf_counter() - clock
            ts = max(int(now * 1000), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            pose, left, right = to_arrays(result)
            f = (now, pose, left, right)
            with lock:
                state["result"] = result
                state["line"] = rest_line(pose, args.below)
                stream.append(f)
                for event in seg.update(f):
                    if event[0] == "start":
                        state["active"] = True
                        start_t = now
                    elif event[0] == "sign":
                        state["active"] = False
                        label, prob = classifier.predict(event[1])
                        gloss = label if prob >= args.min_conf else "?"
                        state["glosses"].append((gloss, prob))
                        bounds.append((start_t, now))
                        print(f"  sign: {label} ({prob:.0%}){'' if gloss != '?' else '  -> ? (low confidence)'}", flush=True)
                    elif event[0] == "phrase":
                        finish_phrase()
                if not seg.active and not seg.signs_in_phrase and len(stream) > 90:
                    del stream[:-30]  # idle at rest: keep only the last second before a sign
                state["active"] = seg.active

    tracker = threading.Thread(target=track, daemon=True)
    tracker.start()
    print(f"Signs: {', '.join(classifier.signs)}. Drop your hands below the line between signs; "
          f"rest {args.phrase_rest}s to end a phrase. q = quit.", flush=True)

    shown = 0
    while True:
        shown, frame = grabber.read(shown)
        if frame is None:
            continue
        frame = frame.copy()
        with lock:
            result, line, active = state["result"], state["line"], state["active"]
            glosses = " ".join(g for g, _ in state["glosses"])
            last, i = state["last_phrase"], state["prompt"]
        if result is not None:
            draw(frame, result)
        h, w = frame.shape[:2]
        if line is not None:
            cv2.line(frame, (0, int(line * h)), (w, int(line * h)), (0, 200, 255), 2)
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "SIGNING" if active else "rest", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (0, 0, 255) if active else (0, 255, 0), 4)
        cv2.putText(frame, glosses, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        if last:
            cv2.putText(frame, f"last: {last}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if prompts:
            if i < len(prompts):
                cv2.putText(frame, f"Sign ({i + 1}/{len(prompts)}): {' '.join(prompts[i])}", (20, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 3)
            else:
                cv2.putText(frame, "Done. q to quit.", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 3)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        with lock:
            if key == ord("x") and scored:
                scored.pop()
                if saved:
                    os.remove(saved.pop())
                state["prompt"] -= 1
                print("  discarded last attempt; prompt repeats", flush=True)
            elif key == ord("s") and prompts and state["prompt"] < len(prompts):
                state["prompt"] += 1
                state["glosses"] = []
                seg.reset()
                print("  skipped", flush=True)

    stop.set()
    tracker.join(timeout=2)
    landmarker.close()
    grabber.stop()
    cap.release()
    cv2.destroyAllWindows()

    if scored:
        exact = sum(e == g for e, g, _ in scored)
        count_ok = sum(len(e) == len(g) for e, g, _ in scored)
        signs_ok = sum(a == b for e, g, _ in scored if len(e) == len(g) for a, b in zip(e, g))
        signs_n = sum(len(e) for e, g, _ in scored if len(e) == len(g))
        print(f"\n{len(scored)} phrases. Exact match: {exact}/{len(scored)} ({exact / len(scored):.0%}). "
              f"Right number of signs (segmentation): {count_ok}/{len(scored)} ({count_ok / len(scored):.0%}). "
              f"Sign accuracy where the count was right: {signs_ok}/{signs_n}.")
        for e, g, c in scored:
            if e != g:
                print(f"  expected {' '.join(e):22s} got {' '.join(g) or '(nothing)':22s} conf {' '.join(f'{x:.0%}' for x in c)}")


if __name__ == "__main__":
    main()
