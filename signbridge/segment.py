"""Cut a live landmark stream into sign clips and phrases (pauses, rest position, tracking dropouts)."""

import collections

import numpy as np

L_SHOULDER, R_SHOULDER = 11, 12


def rest_line(pose, below, floor=0.95):
    """Image y of the rest line: `below` shoulder-widths under the shoulders, but never lower than `floor`
    (sitting close to the camera would otherwise put it off-screen). None without a pose."""
    s = pose[[L_SHOULDER, R_SHOULDER], :2]
    if np.isnan(s).any():
        return None
    return min(s[:, 1].mean() + below * abs(s[0, 0] - s[1, 0]), floor)


def hands_up(pose, left, right, below):
    """True when the top of a tracked hand is above the rest line."""
    line = rest_line(pose, below)
    for hand in (left, right):
        if not np.isnan(hand).all() and (line is None or np.nanmin(hand[:, 1]) < line):
            return True
    return False


class Segmenter:
    """Feeds on one frame at a time; returns sign starts, finished sign clips and phrase ends."""

    def __init__(self, below=1.0, still=0.5, still_speed=0.008, gap=0.3, start_frames=3, pad=3, min_frames=8,
                 max_seconds=6.0, phrase_rest=1.5, smooth=5):
        self.below, self.still, self.still_speed, self.gap = below, still, still_speed, gap
        self.start_frames, self.pad, self.min_frames = start_frames, pad, min_frames
        self.max_seconds, self.phrase_rest, self.smooth = max_seconds, phrase_rest, smooth
        self.reset()

    def reset(self):
        self.active = False
        self.history = collections.deque(maxlen=self.pad + self.start_frames)
        self.clip, self.run = [], 0
        self.gap_start = self.still_start = self.down_since = None
        self.speeds, self.prev, self.width = collections.deque(maxlen=self.smooth), None, None
        self.signs_in_phrase = 0

    def _speed(self, pose, left, right):
        """Fastest wrist speed in shoulder widths per frame, averaged over `smooth` frames; None if unknown."""
        s = pose[[L_SHOULDER, R_SHOULDER], :2]
        if not np.isnan(s).any() and abs(s[0, 0] - s[1, 0]) > 1e-3:
            self.width = abs(s[0, 0] - s[1, 0])
        cur = (left[0, :2], right[0, :2])
        moves = []
        if self.prev is not None and self.width:
            moves = [np.linalg.norm(b - a) / self.width for a, b in zip(self.prev, cur)
                     if not (np.isnan(a).any() or np.isnan(b).any())]
        self.prev = cur
        if not moves:
            self.speeds.clear()
            return None
        self.speeds.append(max(moves))
        return float(np.mean(self.speeds)) if len(self.speeds) == self.smooth else None

    def update(self, frame):
        """frame = (t, pose, left, right). Returns a list of events: ("start",), ("sign", take), ("phrase",)."""
        t, pose, left, right = frame
        up = hands_up(pose, left, right, self.below)
        speed = self._speed(pose, left, right)
        still = speed is not None and speed < self.still_speed
        events = []
        if not self.active:
            self.history.append(frame)
            self.down_since = None if up else (t if self.down_since is None else self.down_since)
            self.run = self.run + 1 if up and not still else 0
            if self.run >= self.start_frames:
                self.active, self.clip = True, list(self.history)
                self.gap_start = self.still_start = None
                events.append(("start",))
            elif self.signs_in_phrase and self.down_since is not None and t - self.down_since >= self.phrase_rest:
                self.signs_in_phrase = 0
                events.append(("phrase",))
            return events

        self.clip.append(frame)
        n = len(self.clip) - 1
        self.gap_start = None if up else (n if self.gap_start is None else self.gap_start)
        self.still_start = (n if self.still_start is None else self.still_start) if up and still else None
        if self.gap_start is not None and t - self.clip[self.gap_start][0] >= self.gap:
            end = self.gap_start
        elif self.still_start is not None and t - self.clip[self.still_start][0] >= self.still:
            end = self.still_start
        elif t - self.clip[0][0] >= self.max_seconds:
            end = n + 1
        else:
            return events
        clip = self.clip[: min(end + self.pad, len(self.clip))]
        self.down_since = self.clip[self.gap_start][0] if self.gap_start is not None else None
        self.active, self.run = False, 0
        self.history.clear()
        if len(clip) >= self.min_frames:
            self.signs_in_phrase += 1
            events.append(("sign", {
                "pose": np.stack([f[1] for f in clip]), "left": np.stack([f[2] for f in clip]),
                "right": np.stack([f[3] for f in clip]), "t": np.array([f[0] for f in clip]) - clip[0][0],
            }))
        return events

