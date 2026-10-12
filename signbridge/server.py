"""The engine process the SignBridge app launches: JSON lines on stdin (commands) and stdout (events).

Commands (one JSON object per line):
  {"cmd": "mode", "mode": "off|run|framing|gestures"}
  {"cmd": "preview", "on": true}
  {"cmd": "gesture", "name": "wake|confirm|cancel"}       on-screen buttons, same as the hand gestures
  {"cmd": "fake", "glosses": ["RAIN", "TOMORROW"]}          testing without signing
  {"cmd": "check", "name": "claude|voice"}                 setup checks the engine can do
  {"cmd": "test_route", "route": "siri|voice", "text": "What time is it?"}
  {"cmd": "calibrate"}
  {"cmd": "config", "values": {...}}                       saves settings
  {"cmd": "quit"}

Events:
  {"type": "ready", "config": {...}}
  {"type": "status", ...}                                  ~10 per second while something changes
  {"type": "frame", "seq": n, "jpeg": "<base64>"}          while the preview is on
  {"type": "check", "name": ..., "ok": bool, "detail": "..."}
  {"type": "calibrated", ...}, {"type": "route_tested", ...}, {"type": "error", "message": ...}

Run by hand to try it: python -m signbridge.server [--dry-run]
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import threading
import time

from signbridge.intent import IntentClient, is_valid_intent
from signbridge.routes import route_voice


def make_writer():
    """Protocol output on the real stdout; anything libraries print goes to stderr instead."""
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def emit(event):
        line = json.dumps(event, separators=(",", ":"))
        with lock:
            out.write(line + "\n")
            out.flush()

    return emit


def check_claude():
    async def one_call():
        async with IntentClient() as client:
            return await client.interpret(["HELLO"])

    try:
        intent, t_intent, _, model = asyncio.run(one_call())
    except Exception as e:
        return False, f"Claude call failed: {e}"
    if not is_valid_intent(intent):
        return False, "Claude answered without a valid result."
    return True, f"{model} answered in {t_intent:.1f}s"


def check_voice(cfg):
    import subprocess

    out = subprocess.run(["say", "-a", "?"], capture_output=True, text=True).stdout
    devices = [line.split(None, 1)[1].strip() for line in out.splitlines() if line.strip()]
    if cfg["voice_device"] in devices:
        return True, f"{cfg['voice_device']} is installed"
    return False, f"{cfg['voice_device']} is not installed"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    emit = make_writer()

    from signbridge.runtime import Runtime  # heavy imports after stdout is protected

    rt = Runtime(dry_run=args.dry_run)
    emit({"type": "ready", "config": rt.cfg})
    stop = threading.Event()

    def in_background(fn, *a):
        threading.Thread(target=fn, args=a, daemon=True).start()

    def run_check(name):
        ok, detail = check_claude() if name == "claude" else check_voice(rt.cfg)
        emit({"type": "check", "name": name, "ok": ok, "detail": detail})

    def test_route(route, text):
        ok, detail = (route_voice(text, rt.cfg["voice_device"]), "") if route == "voice" else rt.send_now(route, text)
        emit({"type": "route_tested", "route": route, "ok": ok, "detail": detail})

    def handle(msg):
        cmd = msg.get("cmd")
        if cmd == "mode":
            rt.set_mode(msg["mode"])
        elif cmd == "preview":
            rt.preview = bool(msg.get("on"))
        elif cmd == "gesture":
            rt.gesture(msg["name"])
        elif cmd == "fake":
            rt.fake_phrase([str(g) for g in msg["glosses"]])
        elif cmd == "check":
            in_background(run_check, msg["name"])
        elif cmd == "test_route":
            in_background(test_route, msg["route"], msg.get("text", "What time is it?"))
        elif cmd == "calibrate":
            emit({"type": "calibrated", **rt.calibrate()})
        elif cmd == "config":
            rt.save_config(msg.get("values", {}))
            emit({"type": "config", "config": rt.cfg})
        elif cmd == "quit":
            stop.set()
        else:
            emit({"type": "error", "message": f"unknown command {cmd!r}"})

    def read_commands():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                handle(json.loads(line))
            except Exception as e:
                emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        stop.set()  # the app closed our stdin: it quit

    threading.Thread(target=read_commands, daemon=True).start()
    last_status, last_sent, frame_seq = None, 0.0, 0
    while not stop.is_set():
        rt.tick()
        status = rt.status()
        now = time.monotonic()
        if status != last_status or now - last_sent > 1:
            emit({"type": "status", **status})
            last_status, last_sent = status, now
        if rt.preview:
            frame_seq, jpeg = rt.preview_jpeg(frame_seq)
            if jpeg:
                emit({"type": "frame", "seq": frame_seq, "jpeg": base64.b64encode(jpeg).decode()})
        time.sleep(1 / 30)
    rt.close()


if __name__ == "__main__":
    main()
