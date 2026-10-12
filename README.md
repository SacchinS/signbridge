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

```sh
.venv/bin/python onboard.py            # once: permissions, Siri/voice setup, camera framing, gestures
.venv/bin/python app.py                # the app, with the live preview
.venv/bin/python app.py --no-camera --dry-run   # keyboard stand-ins (w/c/x, 1-6), nothing is sent
.venv/bin/python -m pytest tests       # gesture and state-machine tests
```

Code: `signbridge/` is the app's package (tracking, features and classifier, segmenter, gestures, state
machine, Claude intent, routes). Spike, recording and training scripts live at the repo root; each has a
header comment explaining how to run it. Product drafts: [signs](docs/product/signs.md),
[demo](docs/product/demo.md).

## Status

- [x] Task 0: repository setup
- [x] Task 1: output-route spike. Siri 10/10, text 30/30, voice 5/5, Haiku 5.5 median 1.08 s ([results](docs/results/task1-routes.md))
- [x] Task 2: tracking 29.8 fps with preview, 97.5% hand recall; 5-sign pilot ~90% on held-out sessions (97% to 79% across seeds; ~98% with PopSign pretraining) ([tracking](docs/results/task2-tracking.md), [pilot](docs/results/task2-pilot.md))

- [ ] Core app: pipeline runs end to end with keyboard stand-ins and real Claude calls (dry run). Untested
  live: gestures on camera, segmentation with the new pause rule, real routes from the app. Menu bar not built
  yet (the preview window is the UI).
- [ ] Onboarding: written; automatic checks pass except the camera (needs the user's terminal).
- [ ] "Not a command" class: code ready, PopSign downloads rate-limited by Kaggle.

Measured results are recorded in [`docs/results/`](docs/results/). The original build plan is in [`docs/HANDOFF.md`](docs/HANDOFF.md).

Built for CalHacks 2026.
