"""Task 2 step 2: record sign takes as landmarks only (no video is saved).

Each take is a fixed window of frames. The preview shows a countdown, then
"SIGN NOW" while recording. Saved per take, under data/landmarks/<SIGN>/:
    <signer>_<session>_<date>_<take>.npz with
    pose  (T, 33, 4)  x, y, z, visibility
    left  (T, 21, 3)  NaN when the hand is not found
    right (T, 21, 3)  NaN when the hand is not found
    t     (T,)        seconds since the take started
Face landmarks are not saved.

Run (from the repo root, inside .venv):
    python record_signs.py --session s1 [--signs HELLO YES NO THANK-YOU HELP] [--takes 10] [--seconds 2.0]
    python record_signs.py --session s2 --model models/pilot.pt   # also classify each take live

Signs are recorded one after another; the preview waits for Space before each
new sign. Keys: Space = start the next sign, x = discard the last saved take,
q = stop early.

With --model, each take is also classified right after it is saved, the guess is
shown in the preview, and per-sign accuracy is printed at the end. Recording a
new session this way doubles as a live held-out test.
Hold out whole sessions for testing, so record each session on a different
sitting (different time, lighting or clothes).
"""

import argparse
import datetime
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from spike_tracking import WINDOW, LatestFrame, draw, make_landmarker

DATA_DIR = "data/landmarks"
PILOT_SIGNS = ["HELLO", "YES", "NO", "THANK-YOU", "HELP"]


def to_arrays(result):
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
    def __init__(self, path):
        import torch

        from train_pilot import TinyConv

        self.torch = torch
        ckpt = torch.load(path, weights_only=False)
        self.signs, self.frames = ckpt["signs"], ckpt["frames"]
        self.model = TinyConv(ckpt["features"], len(self.signs))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(self, take):
        from train_pilot import featurize

        x = self.torch.from_numpy(featurize(take, self.frames)[None])
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(x), 1)[0].numpy()
        return self.signs[int(probs.argmax())], float(probs.max())


def banner(frame, text, color):
    cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signs", nargs="+", default=PILOT_SIGNS, help="gloss labels to record, in order")
    p.add_argument("--session", required=True, help="session id, e.g. s1; one sitting per session")
    p.add_argument("--signer", default="me")
    p.add_argument("--takes", type=int, default=10)
    p.add_argument("--seconds", type=float, default=2.0, help="length of each recorded take")
    p.add_argument("--rest", type=float, default=2.0, help="countdown before each take")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", help="classify each take with this trained model (from train_pilot.py --save)")
    args = p.parse_args()

    date = datetime.date.today().isoformat()

    classifier = Classifier(args.model) if args.model else None
    live_true, live_pred = [], []
    last_guess = ""

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")
    grabber = LatestFrame(cap)
    landmarker = make_landmarker()
    clock = time.perf_counter()
    last_ts = -1
    seq = 0
    saved = []
    sign = ""

    def step(label, color):
        """Track one new frame, show it with a banner; returns (result, key)."""
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
        cv2.putText(frame, f"{sign}  take {min(len(saved) + 1, args.takes)}/{args.takes}", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if last_guess:
            cv2.putText(frame, last_guess, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
        cv2.imshow(WINDOW, frame)
        return result, cv2.waitKey(1) & 0xFF

    print(f"Session {args.session}: {', '.join(args.signs)}, {args.takes} takes of {args.seconds}s each.")
    print("Keys in the preview: Space = start next sign, x = discard last take, q = quit.")
    quit_early = False
    for sign in [g.upper() for g in args.signs]:
        if quit_early:
            break
        out_dir = os.path.join(DATA_DIR, sign)
        os.makedirs(out_dir, exist_ok=True)
        saved = []
        print(f"Next: {sign}. Press Space in the preview window when ready.")
        while True:
            _, key = step(f"Next: {sign} (Space)", (0, 255, 0))
            if key == ord(" "):
                break
            if key == ord("q"):
                quit_early = True
                break
        while len(saved) < args.takes and not quit_early:
            end = time.perf_counter() + args.rest
            while time.perf_counter() < end:
                _, key = step(f"Ready... {int(end - time.perf_counter()) + 1}", (0, 200, 255))
                if key == ord("q"):
                    quit_early = True
                    break
                if key == ord("x") and saved:
                    os.remove(saved.pop())
                    print(f"  discarded take, {len(saved)} kept")
                    end = time.perf_counter() + args.rest
            if quit_early:
                break

            poses, lefts, rights, ts_list = [], [], [], []
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < args.seconds:
                result, _ = step(f"SIGN {sign} NOW", (0, 0, 255))
                if result is None:
                    continue
                pose, left, right = to_arrays(result)
                poses.append(pose)
                lefts.append(left)
                rights.append(right)
                ts_list.append(time.perf_counter() - t0)

            take = len(saved) + 1
            path = os.path.join(out_dir, f"{args.signer}_{args.session}_{date}_{take:02d}.npz")
            while os.path.exists(path):
                take += 1
                path = os.path.join(out_dir, f"{args.signer}_{args.session}_{date}_{take:02d}.npz")
            np.savez_compressed(path, pose=np.stack(poses), left=np.stack(lefts), right=np.stack(rights), t=np.array(ts_list, dtype=np.float32))
            saved.append(path)
            hands = np.mean([not (np.isnan(l).all() and np.isnan(r).all()) for l, r in zip(lefts, rights)])
            guess = ""
            if classifier:
                take_data = {"pose": np.stack(poses), "left": np.stack(lefts), "right": np.stack(rights), "t": np.array(ts_list)}
                label, prob = classifier.predict(take_data)
                mark = "OK" if label == sign else "WRONG"
                last_guess = f"Model: {label} ({prob:.0%}) {mark}"
                guess = f", model says {label} ({prob:.0%}) {mark}"
                if sign in classifier.signs:
                    live_true.append(classifier.signs.index(sign))
                    live_pred.append(classifier.signs.index(label))
            print(f"  {sign}: saved {os.path.basename(path)}: {len(poses)} frames, hand found in {hands:.0%}{guess}")

    landmarker.close()
    grabber.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Session finished.")
    if live_true:
        from train_pilot import report

        report(classifier.signs, np.array(live_true), np.array(live_pred), f"live test, session {args.session}")


if __name__ == "__main__":
    main()
