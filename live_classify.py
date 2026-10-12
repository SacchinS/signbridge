"""Free-form live recognition: sign anything from the trained set and see the model's guess.

Press Space, a short countdown runs, then 2 seconds are captured and classified.
The guess and its confidence are shown in the preview and printed. Nothing is
saved. This is a demo and gut check; the scored test is
`record_signs.py --model ...`, which knows the right answer for each take.

Run (from the repo root, inside .venv):
    python live_classify.py [--model models/pilot.pt] [--seconds 2.0]

Keys in the preview window: Space = capture and classify, q = quit.
"""

import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

from record_signs import banner
from signbridge.features import Classifier, to_arrays
from signbridge.tracking import WINDOW, LatestFrame, draw, make_landmarker


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/pilot.pt")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--countdown", type=float, default=1.0)
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    classifier = Classifier(args.model)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")
    grabber = LatestFrame(cap)
    landmarker = make_landmarker()
    clock = time.perf_counter()
    last_ts, seq = -1, 0
    guess = ""

    def step(label, color):
        nonlocal seq, last_ts
        seq, frame = grabber.read(seq)
        if frame is None:
            return None, -1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((time.perf_counter() - clock) * 1000), last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        draw(frame, result)
        frame = cv2.flip(frame, 1)
        banner(frame, label, color)
        if guess:
            cv2.putText(frame, guess, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 4)
        cv2.putText(frame, "Signs: " + ", ".join(classifier.signs), (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(WINDOW, frame)
        return result, cv2.waitKey(1) & 0xFF

    print(f"Known signs: {', '.join(classifier.signs)}. Space = capture, q = quit.")
    while True:
        _, key = step("Space to sign", (0, 255, 0))
        if key == ord("q"):
            break
        if key != ord(" "):
            continue
        end = time.perf_counter() + args.countdown
        while time.perf_counter() < end:
            step("Ready...", (0, 200, 255))
        poses, lefts, rights, ts = [], [], [], []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            result, _ = step("SIGN NOW", (0, 0, 255))
            if result is None:
                continue
            pose, left, right = to_arrays(result)
            poses.append(pose)
            lefts.append(left)
            rights.append(right)
            ts.append(time.perf_counter() - t0)
        label, prob = classifier.predict({"pose": np.stack(poses), "left": np.stack(lefts), "right": np.stack(rights), "t": np.array(ts)})
        guess = f"{label} ({prob:.0%})"
        print(f"  guess: {guess}", flush=True)

    landmarker.close()
    grabber.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
