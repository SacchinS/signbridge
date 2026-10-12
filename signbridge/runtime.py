"""Everything the app does, without any UI: camera, tracking, gestures, segmenter, classifier, Claude, routes.

A UI (the SwiftUI app through server.py, or the OpenCV dev window in app.py) drives it with a few calls and
reads status() and preview_jpeg(). Modes:
  off       camera closed
  run       the real thing: wake -> signs -> Claude -> confirm -> send
  framing   setup: report where the user sits; calibrate() saves the rest line
  gestures  setup: practice wake, confirm and cancel
"""

import asyncio
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from AppKit import NSRunningApplication, NSWorkspace

from signbridge import config
from signbridge.engine import IDLE, LISTENING, Engine
from signbridge.features import Classifier, to_arrays
from signbridge.gestures import GestureDetector
from signbridge.intent import PRIMARY_MODEL, IntentClient, is_valid_intent
from signbridge.paths import resource
from signbridge.routes import accessibility_ok, route_siri, route_text, route_voice
from signbridge.segment import L_SHOULDER, R_SHOULDER, Segmenter, rest_line
from signbridge.tracking import LatestFrame, draw, make_landmarker

MODES = ("off", "run", "framing", "gestures")
PRACTICE = ["wake", "confirm", "cancel"]


def frontmost_app():
    """(name, bundle id) of the frontmost app, or (None, None) if it's SignBridge itself."""
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None or app.processIdentifier() in (os.getpid(), os.getppid()):
        return None, None
    return app.localizedName(), app.bundleIdentifier()


def activate(bundle_id):
    apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id) if bundle_id else []
    if apps:
        apps[0].activateWithOptions_(0)
        time.sleep(0.3)


class IntentWorker:
    """Runs the Claude client on its own asyncio loop thread."""

    def __init__(self, model=PRIMARY_MODEL):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.client = IntentClient(model)
        self.ready = asyncio.run_coroutine_threadsafe(self.client.__aenter__(), self.loop)

    def interpret(self, glosses, frontmost, done):
        """Calls done(intent or None, seconds, error or None) on the loop thread."""
        async def call():
            await asyncio.wrap_future(self.ready)  # connected (never block the loop itself)
            return await self.client.interpret(glosses, frontmost)

        def finish(future):
            try:
                intent, t_intent, _, _ = future.result()
                done(intent if is_valid_intent(intent) else None, t_intent, None)
            except Exception as e:  # network or login trouble: report, don't crash
                done(None, None, str(e))

        asyncio.run_coroutine_threadsafe(call(), self.loop).add_done_callback(finish)


class Runtime:
    def __init__(self, cfg=None, dry_run=False, intent_model=PRIMARY_MODEL):
        self.cfg = cfg or config.load()
        self.dry_run = dry_run
        self.lock = threading.RLock()
        self.engine = Engine(self._interpret, self._send, min_conf=self.cfg["min_conf"])
        self.intent_model = intent_model
        self.intent_worker = None  # started on first use: connecting takes a few seconds
        self.segmenter = self._new_segmenter()
        self.classifier = None
        self.mode = "off"
        self.preview = False
        self.target = (None, None)
        self._camera = None  # (cap, grabber, stop event, thread)
        self._jpeg, self._jpeg_seq = None, 0
        self._view = {"holding": (None, 0.0), "fps": 0.0, "hands": 0}
        self._framing = []  # (shoulder y, shoulder width, both hands visible)
        self._practice = []
        self.error = None

    def _new_segmenter(self):
        return Segmenter(below=self.cfg["below"], still=self.cfg["still"], phrase_rest=self.cfg["phrase_rest"])

    # ---- engine callbacks ----

    def _interpret(self, glosses, request):
        if self.intent_worker is None:
            self.intent_worker = IntentWorker(self.intent_model)

        def done(intent, t_intent, error):
            with self.lock:
                if error:
                    self.engine.log.append((time.monotonic(), f"Claude error: {error}"))
                elif intent and t_intent:
                    self.engine.log.append((time.monotonic(), f"Claude answered in {t_intent:.1f}s"))
                self.engine.intent_ready(intent, request)

        self.intent_worker.interpret(glosses, self.target[0], done)

    def _send(self, intent):
        def run():
            ok, detail = self.send_now(intent["route"], intent["sentence"], self.target[1])
            with self.lock:
                self.engine.sent(ok, detail)

        threading.Thread(target=run, daemon=True).start()

    def send_now(self, route, sentence, bundle_id=None):
        """Run one output route. Returns (ok, detail)."""
        if self.dry_run:
            return True, f"(dry run) {route}: {sentence}"
        if route == "voice":
            return route_voice(sentence, self.cfg["voice_device"]), ""
        if not accessibility_ok():
            return False, "Accessibility permission is off"
        if route == "siri":
            route_siri(sentence)
        else:
            activate(bundle_id)
            route_text(sentence)
        return True, ""

    # ---- controls ----

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(mode)
        with self.lock:
            self.mode = mode
            self._framing, self._practice = [], list(PRACTICE)
            if mode != "run":
                self.engine.gesture("cancel")
        if mode == "off":
            self._close_camera()
        else:
            self._open_camera()

    def gesture(self, name):
        with self.lock:
            if self.mode == "framing":
                return
            if self.mode == "gestures":
                if self._practice and self._practice[0] == name:
                    self._practice.pop(0)
                return
            if name == "wake" and self.engine.state == IDLE:
                self.target = frontmost_app()
                self.segmenter.reset()
            self.engine.gesture(name)

    def fake_phrase(self, glosses):
        """Inject signs as if recognized (testing and the keyboard fallback)."""
        with self.lock:
            if self.engine.state == IDLE:
                self.gesture("wake")
            for g in glosses:
                self.engine.sign(g, 1.0)
            self.engine.phrase_end()

    def tick(self):
        with self.lock:
            self.engine.tick()

    def calibrate(self):
        """From the framing samples, save the rest line. Returns a result dict for the UI."""
        with self.lock:
            samples = [s for s in self._framing[-150:] if s[0] is not None]
        if len(samples) < 20:
            return {"ok": False, "message": "Couldn't see your shoulders. Sit so your head and shoulders show."}
        y, width = np.median([(s[0], s[1]) for s in samples], axis=0)
        line = min(0.95, y + 0.5 * (1 - y))  # halfway from the shoulders to the bottom edge
        self.cfg["below"] = round(float((line - y) / width), 2)
        config.save(self.cfg)
        with self.lock:
            self.segmenter = self._new_segmenter()
        return {"ok": True, "below": self.cfg["below"], "shoulder_y": round(float(y), 2),
                "message": "Saved. Your rest line is the orange line."}

    def save_config(self, values):
        self.cfg.update(values)
        config.save(self.cfg)
        with self.lock:
            self.segmenter = self._new_segmenter()
            self.engine.min_conf = self.cfg["min_conf"]

    # ---- status for the UI ----

    def status(self):
        with self.lock:
            s = self.engine.status()
            s.update(mode=self.mode, holding=list(self._view["holding"]), fps=round(self._view["fps"], 1),
                     target=self.target[0], error=self.error)
            if self.mode == "framing":
                s["framing"] = self._framing_hint()
            if self.mode == "gestures":
                s["practice"] = {"todo": list(self._practice), "done": [g for g in PRACTICE if g not in self._practice]}
            return s

    def _framing_hint(self):
        recent = self._framing[-30:]
        if len(recent) < 10:
            return {"ok": False, "hint": "Starting the camera..."}
        seen = [r for r in recent if r[0] is not None]
        if len(seen) < len(recent) / 2:
            return {"ok": False, "hint": "I can't see your shoulders. Sit back so your head and shoulders show."}
        y = float(np.median([r[0] for r in seen]))
        hands = np.mean([r[2] for r in recent])
        if y > 0.85:
            return {"ok": False, "hint": "Sit a little farther back or tilt the screen down: show your chest."}
        if y < 0.35:
            return {"ok": False, "hint": "Move closer or tilt the screen up: you're small in the frame."}
        if hands < 0.5:
            return {"ok": True, "hint": "Good position. Now raise both hands to chest height."}
        return {"ok": True, "hint": "Looks good. Both hands are tracked."}

    def preview_jpeg(self, after_seq=0):
        """(seq, jpeg bytes) of the newest preview frame, or (seq, None) if nothing newer."""
        with self.lock:
            if self._jpeg_seq <= after_seq:
                return self._jpeg_seq, None
            return self._jpeg_seq, self._jpeg

    # ---- camera and tracking ----

    def _open_camera(self):
        if self._camera:
            return
        cap = cv2.VideoCapture(self.cfg["camera"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            self.error = "Could not open the camera. Check Camera permission for SignBridge."
            return
        self.error = None
        grabber, stop = LatestFrame(cap), threading.Event()
        thread = threading.Thread(target=self._track, args=(grabber, stop), daemon=True)
        self._camera = (cap, grabber, stop, thread)
        thread.start()

    def _close_camera(self):
        if not self._camera:
            return
        cap, grabber, stop, thread = self._camera
        stop.set()
        thread.join(timeout=2)
        grabber.stop()
        cap.release()
        self._camera = None

    def _track(self, grabber, stop):
        landmarker = make_landmarker()
        if self.classifier is None:
            self.classifier = Classifier(resource(self.cfg["model"]))
        detector = GestureDetector()
        clock = time.perf_counter()
        seq, last_ts, stamps = 0, -1, []
        while not stop.is_set():
            seq, frame = grabber.read(seq)
            if frame is None:
                continue
            now = time.perf_counter() - clock
            last_ts = max(int(now * 1000), last_ts + 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ts)
            pose, left, right = to_arrays(result)
            stamps = [s for s in stamps if now - s < 1] + [now]
            g = detector.update(now, left, right)
            if g:
                self.gesture(g)
            with self.lock:
                self._view = {"holding": detector.holding(now), "fps": len(stamps), "hands": 0}
                if self.mode == "framing":
                    s = pose[[L_SHOULDER, R_SHOULDER], :2]
                    both = not np.isnan(left).all() and not np.isnan(right).all()
                    if np.isnan(s).any():
                        self._framing.append((None, None, both))
                    else:
                        self._framing.append((s[:, 1].mean(), abs(s[0, 0] - s[1, 0]), both))
                elif self.mode == "run" and self.engine.state == LISTENING:
                    for event in self.segmenter.update((now, pose, left, right)):
                        if event[0] == "sign":
                            label, prob = self.classifier.predict(event[1])
                            self.engine.log.append((time.monotonic(), f"saw {label} ({prob:.0%})"))
                            self.engine.sign(label, prob)
                        elif event[0] == "phrase":
                            self.engine.phrase_end()
                want_preview = self.preview
                line = rest_line(pose, self.cfg["below"]) if self.mode in ("run", "framing") else None
            if want_preview:
                self._encode(frame, result, line)
        landmarker.close()

    def _encode(self, frame, result, line):
        draw(frame, result)
        h, w = frame.shape[:2]
        if line is not None:
            cv2.line(frame, (0, int(line * h)), (w, int(line * h)), (0, 165, 255), 2)
        ok, jpeg = cv2.imencode(".jpg", cv2.flip(frame, 1), [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.lock:
                self._jpeg, self._jpeg_seq = jpeg.tobytes(), self._jpeg_seq + 1

    def close(self):
        self._close_camera()
