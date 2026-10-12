"""SignBridge: sign ASL to the webcam to drive Siri, text fields and voice calls on a Mac.

Flow: both open palms (wake) -> sign -> rest your hands -> Claude writes the sentence and picks the route
-> thumbs up sends it, thumbs down cancels. Nothing is ever sent without the thumbs up.

Run (from the repo root, inside .venv; run onboard.py once first):
    python app.py                 # camera + preview
    python app.py --dry-run       # print the route instead of running it
    python app.py --no-camera     # keyboard only: test Claude and the routes without signing
    python app.py --no-preview    # no window; status goes to the terminal

Keys in the preview: w = wake, c = confirm, x = cancel, 1-6 = fake sign phrases, q = quit.
Click into the target app (Messages, Notes, a call) before waking: text is pasted into the app that was in
front when you woke SignBridge.
"""

import argparse
import asyncio
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from AppKit import NSRunningApplication, NSWorkspace

from signbridge import config
from signbridge.engine import CONFIRMING, IDLE, LISTENING, SENDING, THINKING, Engine
from signbridge.features import Classifier, to_arrays
from signbridge.gestures import GestureDetector
from signbridge.intent import PRIMARY_MODEL, IntentClient, is_valid_intent
from signbridge.routes import accessibility_ok, route_siri, route_text, route_voice
from signbridge.segment import Segmenter, rest_line
from signbridge.tracking import LatestFrame, draw, make_landmarker

WINDOW = "SignBridge"
FAKE_PHRASES = {
    "1": ["RAIN", "TOMORROW"],
    "2": ["HUNGRY", "PIZZA"],
    "3": ["HELLO"],
    "4": ["WAIT"],
    "5": ["CALLONPHONE", "MOM"],
    "6": ["TIME", "NOW"],
}
STATE_COLORS = {IDLE: (160, 160, 160), LISTENING: (0, 0, 255), THINKING: (0, 200, 255),
                CONFIRMING: (0, 255, 255), SENDING: (0, 255, 0)}


def frontmost_app():
    """(name, bundle id) of the frontmost app, or (None, None) if it's SignBridge itself."""
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None or app.processIdentifier() == os.getpid():
        return None, None
    return app.localizedName(), app.bundleIdentifier()


def activate(bundle_id):
    apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id) if bundle_id else []
    if apps:
        apps[0].activateWithOptions_(0)
        time.sleep(0.3)


class IntentWorker:
    """Runs the Claude client on its own asyncio loop thread."""

    def __init__(self, model):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.client = IntentClient(model)
        self.ready = asyncio.run_coroutine_threadsafe(self.client.__aenter__(), self.loop)

    def interpret(self, glosses, frontmost, done):
        async def call():
            await asyncio.wrap_future(self.ready)  # connected (never block the loop itself)
            intent, t_intent, _, _ = await self.client.interpret(glosses, frontmost)
            return intent, t_intent

        def finish(future):
            try:
                intent, t_intent = future.result()
            except Exception as e:  # network or login trouble: report, don't crash
                print(f"  Claude error: {e}", flush=True)
                intent, t_intent = None, None
            done(intent if is_valid_intent(intent) else None, t_intent)

        asyncio.run_coroutine_threadsafe(call(), self.loop).add_done_callback(finish)


class App:
    def __init__(self, args, cfg):
        self.args, self.cfg = args, cfg
        self.lock = threading.RLock()
        self.target = (None, None)  # app in front at wake time
        self.intent_worker = IntentWorker(args.intent_model)
        self.engine = Engine(self._interpret, self._send, min_conf=cfg["min_conf"])
        self.printed = 0
        self.view = {"result": None, "line": None, "holding": (None, 0.0)}

    # ---- engine callbacks ----

    def _interpret(self, glosses, request):
        def done(intent, t_intent):
            with self.lock:
                if intent and t_intent:
                    self.engine.log.append((time.monotonic(), f"Claude answered in {t_intent:.1f}s"))
                self.engine.intent_ready(intent, request)

        self.intent_worker.interpret(glosses, self.target[0], done)

    def _send(self, intent):
        def run():
            route, sentence = intent["route"], intent["sentence"]
            ok = True
            if self.args.dry_run:
                print(f"  [dry run] would send to {route}: {sentence}", flush=True)
            elif route == "voice":
                ok = route_voice(sentence, self.cfg["voice_device"])
            elif not accessibility_ok():
                ok = False
                print("  Accessibility permission missing; run onboard.py", flush=True)
            elif route == "siri":
                route_siri(sentence)
            else:
                activate(self.target[1])
                route_text(sentence)
            with self.lock:
                self.engine.sent(ok)

        threading.Thread(target=run, daemon=True).start()

    # ---- inputs ----

    def gesture(self, name):
        with self.lock:
            if name == "wake" and self.engine.state == IDLE:
                self.target = frontmost_app()
                self.segmenter.reset()
            self.engine.gesture(name)

    def fake_phrase(self, glosses):
        with self.lock:
            if self.engine.state == IDLE:
                self.gesture("wake")
            for g in glosses:
                self.engine.sign(g, 1.0)
            self.engine.phrase_end()

    def track(self, grabber, stop):
        landmarker = make_landmarker()
        classifier = Classifier(self.cfg["model"])
        detector = GestureDetector()
        clock = time.perf_counter()
        seq, last_ts = 0, -1
        while not stop.is_set():
            seq, frame = grabber.read(seq)
            if frame is None:
                continue
            now = time.perf_counter() - clock
            last_ts = max(int(now * 1000), last_ts + 1)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), last_ts)
            pose, left, right = to_arrays(result)
            g = detector.update(now, left, right)
            if g:
                self.gesture(g)
            with self.lock:
                self.view = {"result": result, "line": rest_line(pose, self.cfg["below"]),
                             "holding": detector.holding(now)}
                if self.engine.state != LISTENING:
                    continue
                for event in self.segmenter.update((now, pose, left, right)):
                    if event[0] == "sign":
                        label, prob = classifier.predict(event[1])
                        self.engine.log.append((time.monotonic(), f"saw {label} ({prob:.0%})"))
                        self.engine.sign(label, prob)
                    elif event[0] == "phrase":
                        self.engine.phrase_end()
        landmarker.close()

    # ---- output ----

    def print_log(self):
        log = self.engine.log
        for _, message in log[self.printed:]:
            print(f"[{self.engine.state}] {message}", flush=True)
        self.printed = len(log)

    def render(self, frame):
        with self.lock:
            status, view = self.engine.status(), dict(self.view)
        if view["result"] is not None:
            draw(frame, view["result"])
        h, w = frame.shape[:2]
        if view["line"] is not None:
            cv2.line(frame, (0, int(view["line"] * h)), (w, int(view["line"] * h)), (0, 200, 255), 1)
        frame = cv2.flip(frame, 1)
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.rectangle(panel, (0, h - 90), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)
        state = status["state"]
        cv2.putText(frame, state.upper(), (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, STATE_COLORS[state], 3)
        if status["glosses"] and state in (LISTENING, THINKING):
            cv2.putText(frame, " ".join(status["glosses"]), (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        if status["intent"] and state in (CONFIRMING, SENDING):
            text = f'"{status["intent"]["sentence"]}"  -> {status["intent"]["route"]}'
            cv2.putText(frame, text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "thumbs up = send   thumbs down = cancel", (15, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 1)
        gesture, progress = view["holding"]
        if gesture:
            cv2.rectangle(frame, (w - 170, 20), (w - 170 + int(150 * progress), 35), (0, 255, 0), -1)
            cv2.putText(frame, gesture, (w - 170, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, message in enumerate(status["log"]):
            cv2.putText(frame, message[:70], (15, h - 65 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        return frame

    def handle_key(self, key):
        if key in (ord("w"), ord("c"), ord("x")):
            self.gesture({"w": "wake", "c": "confirm", "x": "cancel"}[chr(key)])
        elif chr(key) in FAKE_PHRASES:
            self.fake_phrase(FAKE_PHRASES[chr(key)])

    def run(self):
        self.segmenter = Segmenter(below=self.cfg["below"], still=self.cfg["still"],
                                   phrase_rest=self.cfg["phrase_rest"])
        stop = threading.Event()
        grabber = cap = None
        if not self.args.no_camera:
            cap = cv2.VideoCapture(self.cfg["camera"])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cap.isOpened():
                raise SystemExit("Could not open the camera. Run onboard.py to check permissions.")
            grabber = LatestFrame(cap)
            threading.Thread(target=self.track, args=(grabber, stop), daemon=True).start()
        print(__doc__.split("\n\n")[1], flush=True)
        shown = 0
        blank = np.zeros((480, 640, 3), np.uint8)
        try:
            while True:
                with self.lock:
                    self.engine.tick()
                    self.print_log()
                if self.args.no_preview:
                    time.sleep(0.05)
                    continue
                if grabber:
                    shown, frame = grabber.read(shown)
                    frame = blank.copy() if frame is None else frame.copy()
                else:
                    frame = blank.copy()
                    time.sleep(1 / 30)
                cv2.imshow(WINDOW, self.render(frame))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key != 255:
                    self.handle_key(key)
        except KeyboardInterrupt:
            pass
        stop.set()
        if grabber:
            grabber.stop()
            cap.release()
        cv2.destroyAllWindows()


def main():
    cfg = config.load()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print the route instead of running it")
    p.add_argument("--no-camera", action="store_true", help="keyboard only (w/c/x/1-6)")
    p.add_argument("--no-preview", action="store_true", help="no window; status in the terminal")
    p.add_argument("--intent-model", default=PRIMARY_MODEL)
    args = p.parse_args()
    if args.no_camera and args.no_preview:
        p.error("--no-camera needs the preview window for its keyboard controls")
    if not cfg["onboarded"]:
        print("Tip: run `python onboard.py` once to check permissions and calibrate the camera.")
    App(args, cfg).run()


if __name__ == "__main__":
    main()
