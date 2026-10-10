"""Task 1 output-route spike: fake sign -> Claude intent -> Siri / text / voice.

No ML here. A keypress (1, 2, 3) stands in for a recognized sign sequence.
Claude is reached through the Claude Agent SDK, which uses the local Claude
Code login (a Claude Pro subscription), so no ANTHROPIC_API_KEY is needed.

Run (from the repo root, inside .venv):
    python spike_routes.py bench [--runs 20] [--model claude-haiku-5-5]
    python spike_routes.py text  [--runs 10]   # pastes into the frontmost app
    python spike_routes.py siri  [--runs 10]
    python spike_routes.py voice [--runs 5]
    python spike_routes.py interactive         # press 1/2/3, then pick a route

The text and Siri routes post synthetic keystrokes, which needs Accessibility
permission for the terminal app (System Settings > Privacy & Security >
Accessibility). The voice route needs BlackHole 2ch installed.
"""

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time

import Quartz
from ApplicationServices import AXIsProcessTrusted
from AppKit import NSPasteboard, NSPasteboardTypeString
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

FAKE_SIGNS = {
    "1": ["WEATHER", "TOMORROW"],
    "2": ["ME", "LATE", "TEN", "MINUTE"],
    "3": ["HELLO", "EVERYONE", "ME", "READY", "START"],
}

PRIMARY_MODEL = "claude-haiku-5-5"
FALLBACK_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You turn ASL sign glosses into one natural English sentence for a Deaf user's Mac.
Glosses are uppercase words in signing order and omit articles and tense; rebuild a fluent sentence.
Pick a route:
- siri: a question or command for the voice assistant (weather, timers, opening apps, facts).
- text: a message the user wants typed into the focused text field.
- voice: something the user wants spoken aloud to people in a voice call.
Confidence is 0-1: how sure you are the sentence and route match the user's intent.
Every request is independent. Always answer by calling report_intent exactly once and say nothing else."""

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentence": {"type": "string"},
        "route": {"type": "string", "enum": ["siri", "text", "voice"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["sentence", "route", "confidence"],
}


@tool("report_intent", "Report the sentence, output route and confidence.", INTENT_SCHEMA)
async def report_intent(args):
    return {"content": [{"type": "text", "text": "ok"}]}


def is_valid_intent(intent):
    return (
        isinstance(intent, dict)
        and isinstance(intent.get("sentence"), str)
        and intent["sentence"].strip() != ""
        and intent.get("route") in ("siri", "text", "voice")
        and isinstance(intent.get("confidence"), (int, float))
        and 0 <= intent["confidence"] <= 1
    )


class IntentClient:
    """Keeps one Claude Code subprocess alive so each call skips startup cost."""

    def __init__(self, model=PRIMARY_MODEL):
        server = create_sdk_mcp_server("signbridge", tools=[report_intent])
        self.options = ClaudeAgentOptions(
            model=model,
            fallback_model=FALLBACK_MODEL,
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={"signbridge": server},
            tools=[],
            allowed_tools=["mcp__signbridge__report_intent"],
            setting_sources=[],
            max_turns=2,
        )
        self.client = ClaudeSDKClient(self.options)

    async def __aenter__(self):
        await self.client.connect()
        return self

    async def __aexit__(self, *exc):
        await self.client.disconnect()

    async def interpret(self, glosses):
        """Returns (intent or None, seconds until the tool call, seconds for the full turn, model)."""
        start = time.perf_counter()
        intent, t_intent, model = None, None, None
        await self.client.query(json.dumps({"glosses": glosses}))
        async for msg in self.client.receive_response():
            if isinstance(msg, AssistantMessage):
                model = msg.model
                for block in msg.content:
                    if isinstance(block, ToolUseBlock) and block.name.endswith("report_intent") and intent is None:
                        intent, t_intent = block.input, time.perf_counter() - start
            elif isinstance(msg, ResultMessage):
                break
        return intent, t_intent, time.perf_counter() - start, model


# ---------- keystrokes and clipboard ----------

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


# ---------- routes ----------


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


# ---------- commands ----------


async def cmd_bench(args):
    rows = []
    async with IntentClient(args.model) as ic:
        for i in range(args.runs):
            glosses = FAKE_SIGNS[str(i % 3 + 1)]
            intent, t_intent, t_total, model = await ic.interpret(glosses)
            ok = is_valid_intent(intent)
            rows.append((t_intent, t_total, ok))
            t_i = f"{t_intent:.2f}s" if t_intent else "  -  "
            print(f"{i + 1:2d} {model} intent={t_i} turn={t_total:.2f}s valid={ok} {json.dumps(intent)}")
    intent_times = [r[0] for r in rows if r[0] is not None]
    valid = sum(r[2] for r in rows)
    print(f"\nmodel: {args.model}")
    if intent_times:
        print(f"median time to tool call: {statistics.median(intent_times):.2f}s")
    print(f"median full turn: {statistics.median(r[1] for r in rows):.2f}s")
    print(f"valid tool output: {valid}/{len(rows)}")


def countdown(label, seconds):
    print(f"{label}: focus the target now, starting in {seconds}s", flush=True)
    for s in range(seconds, 0, -1):
        print(f"  {s}...", flush=True)
        time.sleep(1)


def cmd_route(args):
    if args.route in ("text", "siri") and not accessibility_ok():
        sys.exit("Accessibility permission missing for this terminal; grant it and rerun.")
    sentences = {
        "text": "I am running about ten minutes late.",
        "siri": "What is the weather tomorrow?",
        "voice": "Hello everyone, I am ready to start.",
    }
    sentence = args.sentence or sentences[args.route]
    countdown(f"{args.route} route", args.delay)
    for i in range(args.runs):
        if args.route == "text":
            restored = route_text(sentence + (f" ({i + 1})" if args.runs > 1 else ""))
            if args.newline:
                post_key(KEY_RETURN)
            print(f"run {i + 1}: pasted, clipboard restored={restored}")
            time.sleep(0.5)
        elif args.route == "siri":
            restored = route_siri(sentence, open_delay=args.open_delay, hold=args.hold)
            print(f"run {i + 1}: fired, clipboard restored={restored}. Did Siri answer?", flush=True)
            time.sleep(args.gap)
            if not args.no_dismiss:
                post_key(KEY_ESCAPE)  # dismiss Siri before the next run
            time.sleep(1)
        else:
            print(f"run {i + 1}: say exit ok={route_voice(sentence)}", flush=True)
            time.sleep(args.gap)


async def cmd_interactive(args):
    async with IntentClient(args.model) as ic:
        while True:
            key = input("\nfake sign [1/2/3, q to quit]: ").strip()
            if key == "q":
                return
            if key not in FAKE_SIGNS:
                continue
            intent, t_intent, _, _ = await ic.interpret(FAKE_SIGNS[key])
            print(f"{FAKE_SIGNS[key]} -> {intent} ({t_intent:.2f}s)")
            if not is_valid_intent(intent):
                continue
            if input("confirm? [y/N]: ").strip().lower() != "y":
                print("cancelled")
                continue
            route = intent["route"]
            if route == "text":
                countdown("Click into a text field", 3)
                route_text(intent["sentence"])
            elif route == "siri":
                route_siri(intent["sentence"])
            else:
                route_voice(intent["sentence"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--runs", type=int, default=20)
    b.add_argument("--model", default=PRIMARY_MODEL)
    for name in ("text", "siri", "voice"):
        r = sub.add_parser(name)
        r.add_argument("--runs", type=int, default=5 if name == "voice" else 10)
        r.add_argument("--sentence")
        r.add_argument("--delay", type=int, default=5)
        r.add_argument("--newline", action="store_true", help="press Return after each paste")
        r.add_argument("--hold", type=float, default=0.8, help="seconds to hold the Siri shortcut")
        r.add_argument("--open-delay", type=float, default=0.4, help="seconds between opening Siri and pasting")
        r.add_argument("--no-dismiss", action="store_true", help="leave Siri open after each run")
        r.add_argument("--gap", type=float, default=8, help="seconds to wait between Siri/voice runs")
        r.set_defaults(route=name)
    i = sub.add_parser("interactive")
    i.add_argument("--model", default=PRIMARY_MODEL)
    args = p.parse_args()

    if args.cmd == "bench":
        asyncio.run(cmd_bench(args))
    elif args.cmd == "interactive":
        asyncio.run(cmd_interactive(args))
    else:
        cmd_route(args)


if __name__ == "__main__":
    main()
