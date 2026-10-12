"""First-run setup for SignBridge: permissions, routes, camera framing and the control gestures.

Walks through each step, says how to fix anything that fails, and saves the camera calibration to
~/Library/Application Support/SignBridge/config.json. Safe to re-run; each step can be run alone.

Run (from the repo root, inside .venv):
    python onboard.py               # all steps
    python onboard.py --check       # status of the automatic checks only, no prompts or windows
    python onboard.py --step camera # one step: claude, accessibility, siri, voice, camera, gestures
"""

import argparse
import asyncio
import subprocess
import time

import cv2
import mediapipe as mp
import numpy as np
from ApplicationServices import AXIsProcessTrusted, AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt

from signbridge import config
from signbridge.features import to_arrays
from signbridge.gestures import GestureDetector
from signbridge.intent import IntentClient, is_valid_intent
from signbridge.routes import route_siri, route_voice
from signbridge.tracking import LatestFrame, draw, make_landmarker

WINDOW = "SignBridge setup"
L_SHOULDER, R_SHOULDER = 11, 12


def ok(msg):
    print(f"  OK    {msg}", flush=True)
    return True


def fail(msg, fix):
    print(f"  FIX   {msg}\n        {fix}", flush=True)
    return False


def ask(question):
    return input(f"  ?     {question} [y/n] ").strip().lower().startswith("y")


# ---- automatic checks ----

def check_claude():
    async def one_call():
        async with IntentClient() as client:
            return await client.interpret(["HELLO"])

    try:
        intent, t_intent, _, model = asyncio.run(one_call())
    except Exception as e:
        return fail(f"Claude call failed: {e}", "Install Claude Code and log in (`claude` then /login).")
    if not is_valid_intent(intent):
        return fail("Claude answered without a valid intent.", "Re-run; if it persists, check `claude` works.")
    return ok(f"Claude ({model}) answered in {t_intent:.1f}s")


def check_accessibility(prompt=False):
    if AXIsProcessTrusted():
        return ok("Accessibility permission (needed to type into Siri and text fields)")
    if prompt:
        AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
    return fail("Accessibility permission is off for this terminal app.",
                "System Settings > Privacy & Security > Accessibility: turn on your terminal, then restart it.")


def voice_devices():
    out = subprocess.run(["say", "-a", "?"], capture_output=True, text=True).stdout
    return [line.split(None, 1)[1].strip() for line in out.splitlines() if line.strip()]


def check_voice(cfg):
    if cfg["voice_device"] in voice_devices():
        return ok(f"{cfg['voice_device']} found (virtual mic for the voice route)")
    return fail(f"{cfg['voice_device']} not found.",
                "Install it in your own Terminal (needs your password): brew install blackhole-2ch")


def check_camera(cfg):
    cap = cv2.VideoCapture(cfg["camera"])
    good = cap.isOpened() and cap.read()[0]
    cap.release()
    if good:
        return ok("camera opens and returns frames")
    return fail("camera did not return a frame.",
                "System Settings > Privacy & Security > Camera: turn on your terminal app, then restart it.")


# ---- guided steps ----

def step_siri():
    print("\nSiri route. Needs Type to Siri and the 'Hold Command Space' shortcut:")
    print("  System Settings > Apple Intelligence & Siri: Keyboard shortcut = Hold Command Space")
    print("  System Settings > Accessibility > Siri: Type to Siri = on")
    if not ask("Send a test question to Siri now?"):
        return None
    route_siri("What time is it?")
    return ok("Siri answered") if ask("Did Siri answer?") else fail(
        "Siri did not answer.", "Check both settings above, then re-run: python onboard.py --step siri")


def step_voice(cfg):
    print("\nVoice route. In your call app (FaceTime, Zoom, Discord), set the microphone to "
          f"{cfg['voice_device']}. SignBridge speaks into it; people in the call hear it.")
    if not check_voice(cfg) or not ask("Play a test phrase into it now (you won't hear it locally)?"):
        return None
    route_voice("SignBridge voice test.", cfg["voice_device"])
    return ok("test phrase sent; ask someone in the call whether they heard it")


def open_camera(cfg):
    cap = cv2.VideoCapture(cfg["camera"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        return None, None
    return cap, LatestFrame(cap)


def live_loop(cfg, seconds, on_frame):
    """Track for up to `seconds`; on_frame(t, pose, left, right, frame) returns (done, banner text)."""
    cap, grabber = open_camera(cfg)
    if cap is None:
        return False
    landmarker = make_landmarker()
    start, seq, last_ts, done = time.perf_counter(), 0, -1, False
    while time.perf_counter() - start < seconds and not done:
        seq, frame = grabber.read(seq)
        if frame is None:
            continue
        t = time.perf_counter() - start
        last_ts = max(int(t * 1000), last_ts + 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ts)
        pose, left, right = to_arrays(result)
        done, text = on_frame(t, pose, left, right, frame)
        draw(frame, result)
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    landmarker.close()
    grabber.stop()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return done


def step_camera(cfg):
    """Check framing (shoulders, hands) and set the rest line between the shoulders and the frame bottom."""
    print("\nCamera framing. Sit where you will sign. Keep your shoulders in view; the window shows tips.")
    if not ask("Open the camera now?"):
        return None
    shoulders, hands_up = [], 0

    def on_frame(t, pose, left, right, frame):
        nonlocal hands_up
        s = pose[[L_SHOULDER, R_SHOULDER], :2]
        if not np.isnan(s).any():
            shoulders.append((s[:, 1].mean(), abs(s[0, 0] - s[1, 0])))
        both = not np.isnan(left).all() and not np.isnan(right).all()
        if t > 4:
            hands_up += both
        if t < 4:
            return False, "Sit still, hands down..."
        return hands_up >= 20, "Now raise both hands to chest height"

    raised = live_loop(cfg, 15, on_frame)
    if len(shoulders) < 20:
        return fail("couldn't see your shoulders.", "Move back or tilt the screen so your head and shoulders show.")
    y, width = np.median(shoulders, axis=0)
    line = min(0.95, y + 0.5 * (1 - y))  # halfway from the shoulders to the bottom edge
    cfg["below"] = round(float((line - y) / width), 2)
    if y > 0.8:
        print("  NOTE  Your shoulders are near the bottom of the frame. Signing space is better if your chest shows:"
              " sit a little farther back or tilt the screen down.")
    config.save(cfg)
    ok(f"shoulders at {y:.2f} of the frame height; rest line saved (below={cfg['below']})")
    return ok("both hands tracked when raised") if raised else fail(
        "both hands weren't tracked together.", "Improve lighting on your hands and keep them inside the frame.")


def step_gestures(cfg):
    print("\nControl gestures:")
    print("  wake    = both open palms to the camera, apart, held still")
    print("  confirm = thumbs up, held still (other hand down)")
    print("  cancel  = thumbs down, held still")
    if not ask("Practice them now?"):
        return None
    detector, todo = GestureDetector(), ["wake", "confirm", "cancel"]

    def on_frame(t, pose, left, right, frame):
        g = detector.update(t, left, right)
        if g and todo and g == todo[0]:
            print(f"  OK    {todo.pop(0)}", flush=True)
        return not todo, (f"Show: {todo[0].upper()}" if todo else "All done!")

    done = live_loop(cfg, 60, on_frame)
    return ok("all three gestures recognized") if done else fail(
        f"not recognized: {', '.join(todo)}", "Hold the pose still for a second, inside the frame, then retry.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="automatic checks only")
    p.add_argument("--step", choices=["claude", "accessibility", "siri", "voice", "camera", "gestures"])
    args = p.parse_args()
    cfg = config.load()

    if args.check:
        print("SignBridge status:")
        results = [check_claude(), check_accessibility(), check_voice(cfg), check_camera(cfg)]
        print(f"\n{sum(results)}/{len(results)} automatic checks passed. Calibrated: {cfg['onboarded']}.")
        return

    steps = {
        "claude": lambda: (print("\nClaude (writes the sentences):"), check_claude())[1],
        "accessibility": lambda: (print("\nAccessibility:"), check_accessibility(prompt=True))[1],
        "siri": step_siri,
        "voice": lambda: step_voice(cfg),
        "camera": lambda: (print("\nCamera:"), check_camera(cfg))[1] and step_camera(cfg),
        "gestures": lambda: step_gestures(cfg),
    }
    names = [args.step] if args.step else list(steps)
    results = {name: steps[name]() for name in names}
    print("\nSummary:")
    for name, result in results.items():
        print(f"  {name:14s} {'OK' if result else 'skipped' if result is None else 'needs fixing'}")
    if not args.step and results["claude"] and results["accessibility"] and results.get("camera"):
        cfg["onboarded"] = True
        config.save(cfg)
        print("\nSetup saved. Start the app with: python app.py")


if __name__ == "__main__":
    main()
