"""Task 2 step 1: MediaPipe Holistic tracking speed on the webcam.

Runs the Holistic Landmarker on live webcam frames and logs throughput and
detection rates. Gate: 24+ fps sustained for 2 minutes, hands found in 95%+
of frames, with both hands and arms in frame.

Run (from the repo root, inside .venv):
    python spike_tracking.py [--seconds 120] [--camera 0] [--show] [--width 1280 --height 720] [--delegate cpu|gpu]
        [--mode holistic|split]

Needs Camera permission for the terminal app, and the model at
models/holistic_landmarker.task (download URL in docs/HANDOFF.md).
"""

import argparse
import json
import statistics
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = "models/holistic_landmarker.task"
WINDOW = "signbridge tracking (q to quit)"
HAND_MODEL_PATH = "models/hand_landmarker.task"
POSE_MODEL_PATH = "models/pose_landmarker_lite.task"


def base_options(path, delegate):
    d = BaseOptions.Delegate.GPU if delegate == "gpu" else BaseOptions.Delegate.CPU
    return BaseOptions(model_asset_path=path, delegate=d)


def make_landmarker(delegate="cpu"):
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options(MODEL_PATH, delegate),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


POSE_WRISTS = (15, 16)


def visible_wrists(pose_landmarks, min_visibility=0.5, margin=0.02):
    """Count wrists the pose model sees inside the image, i.e. hands that should be detectable."""
    if not pose_landmarks:
        return 0
    count = 0
    for i in POSE_WRISTS:
        lm = pose_landmarks[i]
        inside = margin <= lm.x <= 1 - margin and margin <= lm.y <= 1 - margin
        if inside and (lm.visibility is None or lm.visibility >= min_visibility):
            count += 1
    return count


class LatestFrame:
    """Reads the camera on a background thread and keeps only the newest frame.

    Without this, a loop slower than the camera's 33 ms frame interval waits
    for the next frame after each inference and runs at about half speed.
    """

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.seq = 0
        self.cond = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.cond:
                    self.frame, self.seq = frame, self.seq + 1
                    self.cond.notify_all()

    def read(self, after_seq):
        """Block until a frame newer than after_seq exists; return (seq, frame)."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > after_seq, timeout=1)
            return self.seq, self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)


class SplitResult:
    def __init__(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks):
        self.pose_landmarks = pose_landmarks
        self.left_hand_landmarks = left_hand_landmarks
        self.right_hand_landmarks = right_hand_landmarks


class SplitLandmarker:
    """Hand Landmarker (2 hands) + Pose Landmarker lite, no face mesh.

    Exposes the same detect_for_video / result fields as HolisticLandmarker.
    """

    def __init__(self, delegate="cpu"):
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options(HAND_MODEL_PATH, delegate),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base_options(POSE_MODEL_PATH, delegate),
                running_mode=vision.RunningMode.VIDEO,
            )
        )

    def detect_for_video(self, image, ts):
        h = self.hands.detect_for_video(image, ts)
        p = self.pose.detect_for_video(image, ts)
        sides = {}
        for landmarks, handedness in zip(h.hand_landmarks, h.handedness):
            sides[handedness[0].category_name] = landmarks
        return SplitResult(p.pose_landmarks[0] if p.pose_landmarks else None, sides.get("Left"), sides.get("Right"))

    def close(self):
        self.hands.close()
        self.pose.close()


def draw(frame, result):
    h, w = frame.shape[:2]
    for name, color in (("pose_landmarks", (0, 200, 255)), ("left_hand_landmarks", (0, 255, 0)), ("right_hand_landmarks", (255, 0, 0))):
        for lm in getattr(result, name, None) or []:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, color, -1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--delegate", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--mode", choices=["holistic", "split"], default="holistic", help="split = hands + pose, no face")
    p.add_argument("--warmup", type=float, default=10, help="seconds of positioning preview before measuring (0 = none)")
    p.add_argument("--show", action="store_true", help="keep the preview window open while measuring (runs on the main thread, separate from tracking)")
    p.add_argument("--out", help="write the summary as JSON to this path")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Grant Camera permission to the terminal app and retry.")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera opened but returned no frame (permission prompt pending?).")
    actual = f"{frame.shape[1]}x{frame.shape[0]}"

    landmarker = SplitLandmarker(args.delegate) if args.mode == "split" else make_landmarker(args.delegate)
    grabber = LatestFrame(cap)
    seq = 0
    if args.warmup > 0:
        print(f"Get into position: {args.warmup:.0f}s before measuring starts.", flush=True)
        end = time.perf_counter() + args.warmup
        while time.perf_counter() < end:
            seq, frame = grabber.read(seq)
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            left = int(end - time.perf_counter()) + 1
            cv2.putText(frame, f"Get in position: {left}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow(WINDOW, frame)
            cv2.waitKey(1)
        if not args.show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        print("Measuring now." + ("" if args.show else " Preview closed; keep signing in the same spot."), flush=True)
    frames = any_hand = both_hands = pose = 0
    infer_ms, loop_ms, stamps = [], [], []
    wrists_visible = hands_matched = 0
    latest = {"result": None}
    stop = threading.Event()
    start = time.perf_counter()

    def track():
        # Runs on its own thread so the preview never slows tracking down.
        nonlocal seq, frames, any_hand, both_hands, pose, wrists_visible, hands_matched
        last_ts = -1
        last_print = start
        while not stop.is_set():
            now = time.perf_counter()
            if now - start >= args.seconds:
                break
            seq, frame = grabber.read(seq)
            if frame is None:
                continue
            t_loop = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = max(int((now - start) * 1000), last_ts + 1)
            last_ts = ts
            t0 = time.perf_counter()
            result = landmarker.detect_for_video(image, ts)
            infer_ms.append((time.perf_counter() - t0) * 1000)
            stamps.append(time.perf_counter())
            latest["result"] = result

            left = bool(result.left_hand_landmarks)
            right = bool(result.right_hand_landmarks)
            frames += 1
            any_hand += left or right
            both_hands += left and right
            pose += bool(result.pose_landmarks)
            visible = visible_wrists(result.pose_landmarks)
            wrists_visible += visible
            hands_matched += min(visible, left + right)

            loop_ms.append((time.perf_counter() - t_loop) * 1000)
            if stamps[-1] - last_print >= 5:
                last_print = stamps[-1]
                recent = [s for s in stamps if s >= last_print - 5]
                recall = hands_matched / wrists_visible if wrists_visible else 0
                print(f"{last_print - start:5.0f}s  fps={len(recent) / 5:5.1f}  any hand={any_hand / frames:6.1%}  both={both_hands / frames:6.1%}  recall={recall:6.1%}", flush=True)

    tracker = threading.Thread(target=track)
    tracker.start()
    # cv2.imshow must run on the main thread on macOS, so the preview lives here.
    preview_frames = 0
    shown_seq = seq
    while tracker.is_alive():
        if not args.show:
            tracker.join(timeout=0.2)
            continue
        shown_seq, frame = grabber.read(shown_seq)  # waits for a new camera frame
        if frame is None:
            continue
        frame = frame.copy()
        if latest["result"] is not None:
            draw(frame, latest["result"])
        cv2.imshow(WINDOW, cv2.flip(frame, 1))
        preview_frames += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop.set()
    tracker.join()

    elapsed = stamps[-1] - start if stamps else 0
    # fps over each 1-second window, to catch dips the overall average hides
    per_sec = {}
    for s in stamps:
        per_sec[int(s - start)] = per_sec.get(int(s - start), 0) + 1
    full_secs = [v for k, v in per_sec.items() if k < int(elapsed)]
    summary = {
        "resolution": actual,
        "seconds": round(elapsed, 1),
        "frames": frames,
        "fps_mean": round(frames / elapsed, 1) if elapsed else 0,
        "fps_min_1s": min(full_secs) if full_secs else None,
        "infer_ms_median": round(statistics.median(infer_ms), 1) if infer_ms else None,
        "infer_ms_p95": round(sorted(infer_ms)[int(len(infer_ms) * 0.95)], 1) if infer_ms else None,
        "preview": args.show,
        "preview_fps": round(preview_frames / (time.perf_counter() - start), 1) if args.show else None,
        "loop_ms_median": round(statistics.median(loop_ms), 1) if loop_ms else None,
        "hand_recall_visible_wrists": round(hands_matched / wrists_visible, 3) if wrists_visible else None,
        "any_hand_rate": round(any_hand / frames, 3) if frames else 0,
        "both_hands_rate": round(both_hands / frames, 3) if frames else 0,
        "pose_rate": round(pose / frames, 3) if frames else 0,
        "mode": args.mode,
        "delegate": args.delegate,
        "mediapipe": mp.__version__,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    landmarker.close()
    grabber.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
