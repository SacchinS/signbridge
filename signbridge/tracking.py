"""Camera capture and MediaPipe Holistic tracking."""

import threading

import cv2
from mediapipe.tasks.python import BaseOptions, vision

from signbridge.paths import resource

MODEL_PATH = "models/holistic_landmarker.task"
WINDOW = "signbridge tracking (q to quit)"


def base_options(path, delegate):
    d = BaseOptions.Delegate.GPU if delegate == "gpu" else BaseOptions.Delegate.CPU
    return BaseOptions(model_asset_path=path, delegate=d)


def make_landmarker(delegate="cpu"):
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options(resource(MODEL_PATH), delegate),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


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


def draw(frame, result):
    h, w = frame.shape[:2]
    for name, color in (("pose_landmarks", (0, 200, 255)), ("left_hand_landmarks", (0, 255, 0)), ("right_hand_landmarks", (255, 0, 0))):
        for lm in getattr(result, name, None) or []:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, color, -1)

