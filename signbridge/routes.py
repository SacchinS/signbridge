"""Output routes: Type to Siri, paste into the focused text field, or speak into a virtual mic.

Siri and text post synthetic keystrokes, which needs Accessibility permission. Voice needs BlackHole 2ch.
"""

import subprocess
import time

import Quartz
from AppKit import NSPasteboard, NSPasteboardTypeString
from ApplicationServices import AXIsProcessTrusted

KEY_V, KEY_SPACE, KEY_RETURN, KEY_ESCAPE = 9, 49, 36, 53


def post_key(keycode, flags=0):
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, keycode, down)
        Quartz.CGEventSetFlags(event, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.02)


def accessibility_ok():
    return bool(AXIsProcessTrusted())


def read_clipboard():
    return NSPasteboard.generalPasteboard().stringForType_(NSPasteboardTypeString)


def write_clipboard(text):
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)


def paste_text(text, restore_delay=0.3):
    """Paste via Cmd-V, then restore the previous clipboard. Returns True if restored."""
    saved = read_clipboard()
    write_clipboard(text)
    time.sleep(0.05)
    post_key(KEY_V, Quartz.kCGEventFlagMaskCommand)
    time.sleep(restore_delay)
    if saved is not None:
        write_clipboard(saved)
    return read_clipboard() == saved


def route_text(sentence):
    return paste_text(sentence)


def route_siri(sentence, open_delay=0.4, hold=0.8):
    """macOS 26 flow: open Type to Siri with its keyboard shortcut, paste, Return.

    Assumes Type to Siri is on and the Siri shortcut is "Hold Command Space".
    """
    open_siri(hold)
    time.sleep(open_delay)
    restored = paste_text(sentence)
    post_key(KEY_RETURN)
    return restored


KEY_COMMAND = 55


def open_siri(hold):
    """Hold Cmd-Space for `hold` seconds, sending real modifier key events."""
    cmd = Quartz.kCGEventFlagMaskCommand
    for keycode, down, flags in ((KEY_COMMAND, True, cmd), (KEY_SPACE, True, cmd)):
        event = Quartz.CGEventCreateKeyboardEvent(None, keycode, down)
        Quartz.CGEventSetFlags(event, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.02)
    time.sleep(hold)
    for keycode, down, flags in ((KEY_SPACE, False, cmd), (KEY_COMMAND, False, 0)):
        event = Quartz.CGEventCreateKeyboardEvent(None, keycode, down)
        Quartz.CGEventSetFlags(event, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.02)


def route_voice(sentence, device="BlackHole 2ch"):
    return subprocess.run(["say", "-a", device, sentence]).returncode == 0

