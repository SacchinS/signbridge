"""State-machine checks, including the rule that nothing is sent without a confirm."""

from signbridge.engine import CONFIRMING, IDLE, LISTENING, SENDING, THINKING, Engine

INTENT = {"sentence": "Will it rain tomorrow?", "route": "siri", "confidence": 0.9}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make():
    calls, sends, clock = [], [], Clock()
    e = Engine(lambda g, req: calls.append((g, req)), sends.append, clock=clock)
    return e, calls, sends, clock


def to_confirming(e, calls):
    e.gesture("wake")
    e.sign("RAIN", 0.9)
    e.sign("TOMORROW", 0.95)
    e.phrase_end()
    e.intent_ready(INTENT, calls[-1][1])


def test_happy_path():
    e, calls, sends, _ = make()
    to_confirming(e, calls)
    assert calls[0][0] == ["RAIN", "TOMORROW"]
    assert e.state == CONFIRMING and sends == []
    e.gesture("confirm")
    assert e.state == SENDING and sends == [INTENT]
    e.sent(True)
    assert e.state == IDLE


def test_nothing_happens_without_wake():
    e, calls, sends, _ = make()
    e.sign("RAIN", 0.99)
    e.phrase_end()
    e.gesture("confirm")
    assert e.state == IDLE and calls == [] and sends == []


def test_cancel_and_timeout_never_send():
    e, calls, sends, clock = make()
    to_confirming(e, calls)
    e.gesture("cancel")
    assert e.state == IDLE and sends == []
    to_confirming(e, calls)
    clock.t += 60
    e.tick()
    assert e.state == IDLE and sends == []


def test_late_reply_after_cancel_is_dropped():
    e, calls, sends, _ = make()
    e.gesture("wake")
    e.sign("HELLO", 0.9)
    e.phrase_end()
    assert e.state == THINKING
    stale = calls[-1][1]
    e.gesture("cancel")
    e.gesture("wake")
    e.intent_ready(INTENT, stale)
    assert e.state == LISTENING and e.intent is None


def test_other_and_unsure_signs_are_ignored():
    e, calls, _, _ = make()
    e.gesture("wake")
    e.sign("OTHER", 0.99)
    e.sign("YES", 0.3)
    e.phrase_end()
    assert e.state == LISTENING and calls == []  # nothing to interpret yet
    e.sign("YES", 0.9)
    e.phrase_end()
    assert calls[0][0] == ["YES"]


def test_thumbs_up_while_listening_ends_the_phrase():
    e, calls, sends, _ = make()
    e.gesture("wake")
    e.sign("HELLO", 0.9)
    e.gesture("confirm")
    assert e.state == THINKING and sends == []


def test_listening_times_out():
    e, _, _, clock = make()
    e.gesture("wake")
    clock.t += 5
    e.sign("HELLO", 0.9)  # signing keeps it alive
    clock.t += 9
    e.tick()
    assert e.state == LISTENING
    clock.t += 2
    e.tick()
    assert e.state == IDLE
