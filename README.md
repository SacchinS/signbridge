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

Requirements: Apple silicon Mac, Python 3.12 via [uv](https://docs.astral.sh/uv/), Claude Code logged in, Camera and Accessibility permission for your terminal.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Keep the repo outside iCloud-synced folders (`~/Documents`, `~/Desktop` with "Desktop & Documents" on): macOS
evicts files there when the disk is low, and imports then hang waiting for iCloud.

The intent layer calls Claude through the [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/), which uses your local Claude Code login, so no API key is needed for testing.

Spike scripts live at the repo root; each has a header comment explaining how to run it.

## Status

- [x] Task 0: repository setup
- [x] Task 1: output-route spike. Siri 10/10, text 30/30, voice 5/5, Haiku 5.5 median 1.08 s ([results](docs/results/task1-routes.md))
- [x] Task 2: tracking 29.8 fps with preview, 97.5% hand recall; 5-sign pilot ~90% on held-out sessions (97% to 79% across seeds; ~98% with PopSign pretraining) ([tracking](docs/results/task2-tracking.md), [pilot](docs/results/task2-pilot.md))

Measured results are recorded in [`docs/results/`](docs/results/). The original build plan is in [`docs/HANDOFF.md`](docs/HANDOFF.md).

Built for CalHacks 2026.
