# SignBridge

SignBridge is a Mac menu-bar app that lets a Deaf or hard-of-hearing user control any voice-driven app by signing ASL to the webcam. Hand and body landmarks are tracked on-device with MediaPipe, a small classifier recognizes a set of command signs, and Claude turns the recognized glosses into a sentence and picks an output route: Siri, a pasted text field, or a synthesized voice on a virtual microphone. Anything a hearing person can do with their voice on a Mac, you can do by signing.

## Pipeline

```
webcam -> MediaPipe Holistic landmarks (on-device)
       -> small PyTorch classifier (~40 command signs)
       -> Claude intent layer (tool call: sentence, route, confidence)
       -> confirm / cancel gesture
       -> output route: Siri | text paste | voice (virtual mic)
```

Rule-based control gestures gate everything: both open palms held still to wake, thumbs up to send,
thumbs down to cancel. Nothing is sent without the thumbs up, and random hand movement does nothing.

## How to run

Requirements: Apple silicon Mac, Python 3.12 via [uv](https://docs.astral.sh/uv/), Claude Code logged in, Camera and Accessibility permission for your terminal.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Keep the repo outside iCloud-synced folders (`~/Documents`, `~/Desktop` with "Desktop & Documents" on): macOS
evicts files there when the disk is low, and imports then hang waiting for iCloud.

The intent layer calls Claude through the [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/), which uses your local Claude Code login, so no API key is needed for testing.

**The Mac app.** Build, install to `~/Applications` and open:

```sh
macos/build.sh && open ~/Applications/SignBridge.app
```

It lives in the menu bar (hand icon). First launch opens a setup wizard: camera and Accessibility
permissions, Claude sign-in check, Siri and voice setup, sitting position, and gesture practice. The
preview window shows the camera with tracking, what you signed, and Claude's sentence and route.
**Demo Mode** (menu bar, or Demo in the preview) is a full-screen stage for an audience with a history of
recent phrases; Esc leaves it. Backup keys there: W start, C send, X cancel, 1-6 test phrases.
**Sign Guide** shows animated diagrams of each sign the model knows, drawn from recorded landmarks
(`python export_sign_guide.py` regenerates `macos/Assets/signs.json`).
Engine log: `~/Library/Logs/SignBridge/engine.log`.

**Developer tools:**

```sh
.venv/bin/python dev_app.py --no-camera --dry-run   # OpenCV front end; keys w/c/x and 1-6, nothing sent
.venv/bin/python onboard.py --check                 # terminal setup checks
.venv/bin/python -m pytest tests                    # gesture and state-machine tests
```

Code: `macos/` is the SwiftUI app; it launches the Python engine (`signbridge/server.py`) and talks JSON
lines with it. `signbridge/` holds the engine (runtime, tracking, numpy classifier, segmenter, gestures,
state machine, Claude intent, routes). Spike, recording and training scripts live at the repo root; each has a
header comment explaining how to run it. Product drafts: [signs](docs/product/signs.md),
[demo](docs/product/demo.md).

## Status

- [x] Task 0: repository setup
- [x] Task 1: output-route spike. Siri 10/10, text 30/30, voice 5/5, Haiku 5.5 median 1.08 s ([results](docs/results/task1-routes.md))
- [x] Task 2: tracking 29.8 fps with preview, 97.5% hand recall; 5-sign pilot ~90% on held-out sessions (97% to 79% across seeds; ~98% with PopSign pretraining) ([tracking](docs/results/task2-tracking.md), [pilot](docs/results/task2-pilot.md))

- [ ] Mac app: builds, installs and launches; bundled engine passes Claude and fake-phrase checks. Not yet
  walked through by a user; gestures and segmentation untested on camera.
- [ ] "Not a command" class: code ready, PopSign downloads rate-limited by Kaggle.

Measured results are recorded in [`docs/results/`](docs/results/). The original build plan is in [`docs/HANDOFF.md`](docs/HANDOFF.md).

Built for CalHacks 2026.
