# SignBridge

SignBridge is a Mac menu-bar app that lets a Deaf or hard-of-hearing user control any voice-driven app by signing ASL to the webcam. Hand and body landmarks are tracked on-device with MediaPipe, a small classifier recognizes a set of command signs, and Claude turns the recognized glosses into a sentence and picks an output route: Siri, a pasted text field, or a synthesized voice on a virtual microphone. Anything a hearing person can do with their voice on a Mac, you can do by signing.

## Pipeline

```
webcam -> MediaPipe Holistic landmarks (on-device)
       -> small PyTorch classifier (~40 command signs)
       -> Claude intent layer (tool call: sentence, route, confidence)
       -> confirm / cancel sign
       -> output route: Siri | text paste | voice (virtual mic)
```

A distinct two-part wake-sign with a timeout gates recognition, so random hand movement does nothing.

## How to run

Requirements: Apple silicon Mac, Python 3.11+, Camera and Accessibility permission for your terminal.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install mediapipe opencv-python pyobjc rumps anthropic torch numpy
cp .env.example .env   # then set ANTHROPIC_API_KEY in your shell
```

Spike scripts live at the repo root; each has a header comment explaining how to run it.

## Status

- [x] Task 0: repository setup
- [ ] Task 1: output-route spike (Siri, text paste, voice) with no ML
- [ ] Task 2: tracking speed and 5-sign pilot

Measured results are recorded in [`docs/results/`](docs/results/). The original build plan is in [`docs/HANDOFF.md`](docs/HANDOFF.md).

Built for CalHacks 2026.
