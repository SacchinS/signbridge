"""Per-user settings saved by onboarding, at ~/Library/Application Support/SignBridge/config.json."""

import json
import os

PATH = os.path.expanduser("~/Library/Application Support/SignBridge/config.json")

DEFAULTS = {
    "model": "models/popsign.pt",
    "camera": 0,
    "below": 1.0,  # rest line, in shoulder widths below the shoulders (calibrated in onboarding)
    "still": 0.5,  # seconds of holding still that end a sign
    "phrase_rest": 1.5,  # seconds with hands down that end a phrase
    "min_conf": 0.6,  # signs below this confidence are ignored
    "voice_device": "BlackHole 2ch",
    "onboarded": False,
}


def load():
    cfg = dict(DEFAULTS)
    try:
        with open(PATH) as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    return cfg


def save(cfg):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w") as f:
        json.dump(cfg, f, indent=2)
