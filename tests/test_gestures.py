"""Synthetic-hand checks for the control gestures. Real-data false-positive check: see docs/results."""

import numpy as np

from signbridge.gestures import GestureDetector, hand_shape

NAN_HAND = np.full((21, 3), np.nan, np.float32)


def make_hand(kind, wrist=(0.5, 0.6), size=0.08):
    """Rough 21-point hand. kind: open (fingers up), fist, thumbs_up, thumbs_down (sideways fists)."""
    w = np.array(wrist)
    pts = np.zeros((21, 3), np.float32)
    pts[0, :2] = w
    flip = -1 if kind == "thumbs_down" else 1
    if kind == "open":
        pts[1:5, :2] = w + size * np.array([[-0.3, -0.3], [-0.5, -0.55], [-0.7, -0.75], [-0.9, -0.9]])
        for i, dx in enumerate((-0.3, -0.1, 0.1, 0.3)):
            mcp = w + size * np.array([dx, -1.0])
            pts[5 + 4 * i, :2] = mcp
            pts[6 + 4 * i, :2] = mcp + size * np.array([0, -0.4])
            pts[7 + 4 * i, :2] = mcp + size * np.array([0, -0.65])
            pts[8 + 4 * i, :2] = mcp + size * np.array([0, -0.9])
        return pts
    # sideways fist: knuckles stacked vertically, fingers curled back toward the wrist side
    for i, dy in enumerate((-0.3, -0.1, 0.1, 0.3)):
        mcp = w + size * np.array([0.9, flip * dy])
        pts[5 + 4 * i, :2] = mcp
        pts[6 + 4 * i, :2] = mcp + size * np.array([0.35, 0])
        pts[7 + 4 * i, :2] = mcp + size * np.array([0.25, 0])
        pts[8 + 4 * i, :2] = mcp + size * np.array([0.05, 0])
    # middle MCP (9) sets the hand size: keep it ~1 size from the wrist
    thumb_base = w + size * np.array([0.6, flip * -0.45])
    pts[1, :2] = w + size * np.array([0.3, flip * -0.3])
    pts[2, :2] = thumb_base
    if kind == "fist":
        pts[3, :2] = thumb_base + size * np.array([0.3, 0.1 * flip])
        pts[4, :2] = thumb_base + size * np.array([0.45, 0.3 * flip])
    else:
        pts[3, :2] = thumb_base + size * np.array([0.05, flip * -0.45])
        pts[4, :2] = thumb_base + size * np.array([0.1, flip * -0.85])
    return pts


def test_shapes():
    assert hand_shape(make_hand("open")) == "open"
    assert hand_shape(make_hand("thumbs_up")) == "thumbs_up"
    assert hand_shape(make_hand("thumbs_down")) == "thumbs_down"
    assert hand_shape(make_hand("fist")) is None
    assert hand_shape(NAN_HAND) is None


def run(frames, fps=25):
    det, out = GestureDetector(), []
    for i, (l, r) in enumerate(frames):
        e = det.update(i / fps, l, r)
        if e:
            out.append(e)
    return out


def test_wake_needs_both_hands_apart_and_held():
    l, r = make_hand("open", (0.3, 0.6)), make_hand("open", (0.7, 0.6))
    assert run([(l, r)] * 25) == ["wake"]  # 1 s hold fires once
    assert run([(l, r)] * 10) == []  # 0.4 s is too short
    assert run([(l, NAN_HAND)] * 25) == []  # one hand is not enough
    close = make_hand("open", (0.32, 0.6))
    assert run([(l, close)] * 25) == []  # hands together (e.g. a two-handed sign)


def test_wake_must_be_still():
    frames = [(make_hand("open", (0.3 + 0.01 * i, 0.6)), make_hand("open", (0.7 + 0.01 * i, 0.6))) for i in range(40)]
    assert run(frames) == []


def test_confirm_and_cancel():
    up, down = make_hand("thumbs_up"), make_hand("thumbs_down")
    assert run([(NAN_HAND, up)] * 20) == ["confirm"]
    assert run([(NAN_HAND, down)] * 20) == ["cancel"]
    base = make_hand("open", (0.52, 0.65))  # other hand right under the thumb, like HELP
    assert run([(base, up)] * 20) == []


def test_flicker_is_forgiven_and_refire_needs_release():
    up = make_hand("thumbs_up")
    frames = [(NAN_HAND, up)] * 8 + [(NAN_HAND, NAN_HAND)] * 2 + [(NAN_HAND, up)] * 8
    assert run(frames) == ["confirm"]
    assert run([(NAN_HAND, up)] * 60) == ["confirm"]  # held long: still fires once
    assert run([(NAN_HAND, up)] * 20 + [(NAN_HAND, NAN_HAND)] * 10 + [(NAN_HAND, up)] * 20) == ["confirm", "confirm"]
