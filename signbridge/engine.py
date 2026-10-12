"""The app's state machine: wake -> listen to signs -> Claude -> confirm -> send.

IDLE        nothing happens except the wake gesture
LISTENING   signs are collected; the phrase ends when the hands rest (segmenter), or cancel
THINKING    Claude turns the glosses into a sentence and a route
CONFIRMING  the sentence is shown; only confirm or cancel count; times out to IDLE (never sends)
SENDING     the route runs, then back to IDLE

Pure logic: no camera, threads or I/O. The app feeds it events and gives it two callbacks:
interpret(glosses) starts a Claude call whose result comes back via intent_ready(), and
send(intent) runs the output route. Every action requires an explicit confirm.
"""

import time

IDLE, LISTENING, THINKING, CONFIRMING, SENDING = "idle", "listening", "thinking", "confirming", "sending"


class Engine:
    def __init__(self, interpret, send, clock=time.monotonic, listen_timeout=10.0, confirm_timeout=15.0,
                 min_conf=0.6, other="OTHER"):
        self.interpret, self.send, self.clock = interpret, send, clock
        self.listen_timeout, self.confirm_timeout = listen_timeout, confirm_timeout
        self.min_conf, self.other = min_conf, other
        self.log = []  # (time, message) for the preview and the terminal
        self._to(IDLE, "ready: show both open palms to start")

    def _to(self, state, message=""):
        self.state, self.since = state, self.clock()
        if state in (IDLE, LISTENING):
            self.glosses, self.intent, self.request = [], None, None
        if message:
            self.log.append((self.since, message))

    # ---- events from the app ----

    def gesture(self, name):
        if name == "wake" and self.state == IDLE:
            self._to(LISTENING, "listening: sign, then rest your hands")
        elif name == "cancel" and self.state in (LISTENING, THINKING, CONFIRMING):
            self._to(IDLE, "cancelled")
        elif name == "confirm" and self.state == CONFIRMING:
            intent = self.intent
            self._to(SENDING, f"sending to {intent['route']}")
            self.send(intent)
        elif name == "confirm" and self.state == LISTENING and self.glosses:
            self.phrase_end()  # thumbs up while listening also means "done signing"

    def sign(self, label, prob):
        if self.state != LISTENING:
            return
        self.since = self.clock()  # each sign extends the listening window
        if label == self.other:
            return
        if prob < self.min_conf:
            self.log.append((self.since, f"unsure sign ({label} {prob:.0%}), ignored"))
            return
        self.glosses.append(label)

    def phrase_end(self):
        if self.state != LISTENING or not self.glosses:
            return
        self._to(THINKING, f"thinking: {' '.join(self.glosses)}")
        self.request = object()  # identifies this call; a late reply to a cancelled one is dropped
        self.interpret(list(self.glosses), self.request)

    def intent_ready(self, intent, request):
        if self.state != THINKING or request is not self.request:
            return
        if intent is None:
            self._to(IDLE, "Claude didn't return a sentence; try again")
            return
        self.intent = intent
        self._to(CONFIRMING, f'"{intent["sentence"]}" -> {intent["route"]}. Thumbs up to send, thumbs down to cancel')

    def sent(self, ok, detail=""):
        if self.state == SENDING:
            self._to(IDLE, "sent" if ok else f"send failed {detail}".strip())

    def tick(self):
        """Call regularly: times out listening and confirming. A confirm timeout never sends."""
        age = self.clock() - self.since
        if self.state == LISTENING and age > self.listen_timeout:
            self._to(IDLE, "timed out")
        elif self.state == CONFIRMING and age > self.confirm_timeout:
            self._to(IDLE, "not confirmed in time; nothing sent")

    def status(self):
        """Snapshot for the preview: state, glosses so far, pending intent, last messages."""
        return {"state": self.state, "glosses": list(self.glosses), "intent": self.intent,
                "log": [m for _, m in self.log[-3:]]}
