"""Rule-based control gestures from hand landmarks. No ML, so they need no training data.

wake    = both hands open, fingers up, held apart and still for `wake` seconds
confirm = thumbs up (other fingers curled), held still for `hold` seconds, other hand away from it
cancel  = thumbs down, same rules as confirm

Shapes are judged in 2D image coordinates, scaled by the hand's own size, so distance from the camera
doesn't matter. Gestures must be held still: signs move, which is what keeps THANK-YOU (flat hands) and
HELP (a thumbs-up on the other palm) from triggering them. Each gesture fires once per hold; the hand
must change shape before it can fire again.
"""

import numpy as np

WRIST, THUMB_MCP, THUMB_TIP, MIDDLE_MCP = 0, 2, 4, 9
FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]  # (mcp, pip, tip) for index..pinky


def hand_shape(hand):
    """One hand (21, 3) -> "open", "thumbs_up", "thumbs_down", or None."""
    if hand is None or np.isnan(hand).any():
        return None
    p = hand[:, :2]
    size = np.linalg.norm(p[MIDDLE_MCP] - p[WRIST])
    if size < 1e-3:
        return None
    dist = lambda a, b: np.linalg.norm(p[a] - p[b]) / size  # noqa: E731
    extended = [dist(tip, WRIST) > 1.15 * dist(pip, WRIST) for _, pip, tip in FINGERS]
    curled = [dist(tip, WRIST) < dist(pip, WRIST) for _, pip, tip in FINGERS]
    thumb = (p[THUMB_TIP] - p[THUMB_MCP]) / size
    thumb_out = dist(THUMB_TIP, 5) > 0.5  # thumb tip away from the index knuckle

    if all(extended) and (p[12, 1] - p[WRIST, 1]) / size < -1.2:  # middle fingertip well above the wrist
        return "open"
    if all(curled) and thumb_out:
        if thumb[1] < -0.5 and abs(thumb[0]) < abs(thumb[1]):
            return "thumbs_up"
        if thumb[1] > 0.5 and abs(thumb[0]) < abs(thumb[1]):
            return "thumbs_down"
    return None


def _size(hand):
    return np.linalg.norm(hand[MIDDLE_MCP, :2] - hand[WRIST, :2])


def frame_gesture(left, right, apart=2.0):
    """(gesture, anchor) for a single frame: anchor is the wrist position(s) that must stay still, in
    hand sizes. (None, None) when the frame shows no control gesture."""
    a, b = hand_shape(left), hand_shape(right)
    if a == b == "open":
        size = (_size(left) + _size(right)) / 2
        if np.linalg.norm(left[WRIST, :2] - right[WRIST, :2]) / size < apart:
            return None, None
        return "wake", np.concatenate([left[WRIST, :2], right[WRIST, :2]]) / size
    for shape, gesture in (("thumbs_up", "confirm"), ("thumbs_down", "cancel")):
        hands = [h for h, sh in ((left, a), (right, b)) if sh == shape]
        if len(hands) != 1 or {a, b} == {"thumbs_up", "thumbs_down"}:
            continue
        hand = hands[0]
        other = right if hand is left else left
        size = _size(hand)
        if not np.isnan(other).all() and np.nanmin(np.linalg.norm(other[:, :2] - hand[WRIST, :2], axis=1)) / size < apart:
            return None, None  # touching the other hand: a sign like HELP, not a gesture
        return gesture, hand[WRIST, :2] / size
    return None, None


class GestureDetector:
    """Feeds on one frame at a time; returns "wake", "confirm" or "cancel" once per held gesture."""

    def __init__(self, wake=0.7, hold=0.5, slack=0.15, drift=0.5):
        self.need = {"wake": wake, "confirm": hold, "cancel": hold}
        self.slack = slack  # brief tracking flickers inside a hold are forgiven
        self.drift = drift  # how far (in hand sizes) the wrists may move during a hold
        self.current, self.since, self.last_seen, self.fired, self.anchor = None, None, None, False, None

    def update(self, t, left, right):
        g, anchor = frame_gesture(left, right)
        if g is not None and g == self.current and np.abs(anchor - self.anchor).max() <= self.drift:
            self.last_seen = t
        elif g is not None:  # new gesture, or the hand moved: start the hold again
            self.current, self.since, self.last_seen, self.fired, self.anchor = g, t, t, False, anchor
        elif self.current is not None and t - self.last_seen > self.slack:
            self.current, self.fired = None, False
        if self.current and not self.fired and t - self.since >= self.need[self.current]:
            self.fired = True
            return self.current
        return None

    def holding(self, t):
        """(gesture, progress 0-1) for the preview, or (None, 0)."""
        if not self.current or self.fired:
            return None, 0.0
        return self.current, min((t - self.since) / self.need[self.current], 1.0)
