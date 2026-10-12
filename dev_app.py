"""Developer front end for the SignBridge runtime: an OpenCV window instead of the Mac app.

The Mac app (macos/) is the real UI; this is for quick testing from the terminal.

Run (from the repo root, inside .venv):
    python dev_app.py                 # camera + preview
    python dev_app.py --dry-run       # print the route instead of running it
    python dev_app.py --no-camera     # keyboard only: test Claude and the routes without signing

Keys: w = wake, c = confirm, x = cancel, 1-6 = fake sign phrases, q = quit.
"""

import argparse
import time

import cv2
import numpy as np

from signbridge.engine import CONFIRMING, IDLE, LISTENING, SENDING, THINKING
from signbridge.runtime import Runtime

WINDOW = "SignBridge (dev)"
FAKE_PHRASES = {
    "1": ["RAIN", "TOMORROW"],
    "2": ["HUNGRY", "PIZZA"],
    "3": ["HELLO"],
    "4": ["WAIT"],
    "5": ["CALLONPHONE", "MOM"],
    "6": ["TIME", "NOW"],
}
COLORS = {IDLE: (160, 160, 160), LISTENING: (0, 0, 255), THINKING: (0, 200, 255), CONFIRMING: (0, 255, 255),
          SENDING: (0, 255, 0)}


def render(frame, s):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.putText(frame, s["state"].upper(), (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLORS[s["state"]], 3)
    if s["glosses"] and s["state"] in (LISTENING, THINKING):
        cv2.putText(frame, " ".join(s["glosses"]), (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    if s["intent"] and s["state"] in (CONFIRMING, SENDING):
        cv2.putText(frame, f'"{s["intent"]["sentence"]}" -> {s["intent"]["route"]}', (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    gesture, progress = s["holding"]
    if gesture:
        cv2.rectangle(frame, (w - 170, 20), (w - 170 + int(150 * progress), 35), (0, 255, 0), -1)
        cv2.putText(frame, gesture, (w - 170, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for i, message in enumerate(s["log"]):
        cv2.putText(frame, message[:70], (15, h - 65 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-camera", action="store_true")
    args = p.parse_args()
    rt = Runtime(dry_run=args.dry_run)
    rt.preview = True
    if not args.no_camera:
        rt.set_mode("run")
    seq, image, printed = 0, np.zeros((480, 640, 3), np.uint8), 0
    try:
        while True:
            rt.tick()
            s = rt.status()
            for _, message in rt.engine.log[printed:]:
                print(f"[{s['state']}] {message}", flush=True)
            printed = len(rt.engine.log)
            seq, jpeg = rt.preview_jpeg(seq)
            if jpeg:
                image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow(WINDOW, render(image.copy(), s))
            key = cv2.waitKey(15) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("w"), ord("c"), ord("x")):
                rt.gesture({"w": "wake", "c": "confirm", "x": "cancel"}[chr(key)])
            elif key != 255 and chr(key) in FAKE_PHRASES:
                rt.fake_phrase(FAKE_PHRASES[chr(key)])
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    rt.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
