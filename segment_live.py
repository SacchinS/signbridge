"""Live sign sequences: cut signs at pauses or the rest position, classify each, end the phrase on a rest.

A sign starts when a hand is up (above the rest line drawn on the preview) and moving. It ends when the
hands hold still for --still seconds, or drop below the line / leave the frame for --gap seconds (shorter
tracking dropouts are ignored). Each cut clip is classified right away. A still hold only separates signs;
the phrase (the list of signs) ends after --phrase-rest seconds with the hands down or out of view.

Scripted test: the preview prompts a sequence (e.g. "YES NO"); sign it with a short pause (or hands down)
between signs, then drop your hands until the phrase ends. Each attempt is scored and its landmark stream
saved to data/sequences/ so the segmenter can be tuned offline (segment_replay.py).

Run (from the repo root, inside .venv):
    python segment_live.py [--model models/popsign.pt]              # script mode, default sequences
    python segment_live.py --free                                    # no prompts, just recognize
    python segment_live.py --script "HELLO THANK-YOU" "YES NO" --reps 3

Keys in the preview: x = discard the last scored attempt, s = skip the current prompt, q = quit.
"""

import argparse
import datetime
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

from signbridge.features import Classifier, to_arrays
from signbridge.segment import Segmenter, rest_line
from signbridge.tracking import WINDOW, LatestFrame, draw, make_landmarker

DEFAULT_SCRIPT = ["HELLO", "YES NO", "HELLO THANK-YOU", "NO YES", "THANK-YOU HELLO", "YES THANK-YOU NO"]
OUT_DIR = "data/sequences"
OTHER = "OTHER"  # the model's "not a command" class, if it has one


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/popsign.npz")
    p.add_argument("--free", action="store_true", help="no prompts or scoring")
    p.add_argument("--script", nargs="+", default=DEFAULT_SCRIPT, help="sequences to prompt, signs separated by spaces")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--below", type=float, default=1.0, help="rest line, in shoulder widths below the shoulders")
    p.add_argument("--phrase-rest", type=float, default=1.5, help="seconds with hands down that end a phrase")
    p.add_argument("--still", type=float, default=0.5, help="seconds of holding still that end a sign")
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
    seg = Segmenter(below=args.below, still=args.still, phrase_rest=args.phrase_rest)
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
        if not glosses and not prompts:  # only non-command movements
            stream.clear()
            bounds.clear()
            return
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
                        bounds.append((start_t, now))
                        if label == OTHER:
                            print(f"  not a command ({prob:.0%}), ignored", flush=True)
                            continue
                        gloss = label if prob >= args.min_conf else "?"
                        state["glosses"].append((gloss, prob))
                        print(f"  sign: {label} ({prob:.0%}){'' if gloss != '?' else '  -> ? (low confidence)'}", flush=True)
                    elif event[0] == "phrase":
                        finish_phrase()
                if not seg.active and not seg.signs_in_phrase and len(stream) > 90:
                    del stream[:-30]  # idle at rest: keep only the last second before a sign
                state["active"] = seg.active

    tracker = threading.Thread(target=track, daemon=True)
    tracker.start()
    print(f"Signs: {', '.join(classifier.signs)}. Pause briefly between signs; "
          f"drop your hands for {args.phrase_rest}s to end a phrase. q = quit.", flush=True)

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
