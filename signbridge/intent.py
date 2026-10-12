"""Claude intent layer: sign glosses -> one English sentence, an output route and a confidence.

Uses the Claude Agent SDK, which runs on the local Claude Code login (no ANTHROPIC_API_KEY needed).
"""

import json
import time

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

PRIMARY_MODEL = "claude-haiku-5-5"


FALLBACK_MODEL = "claude-haiku-4-5-20251001"


SYSTEM_PROMPT = """You turn ASL sign glosses into one natural English sentence for a Deaf user's Mac.
Glosses are uppercase words in signing order and omit articles and tense; rebuild a fluent sentence.
Some glosses are compounds: CALLONPHONE = phone call, THANKYOU = thank you, RADIO = music/radio,
LOUD/QUIET about media = volume up/down.
Pick a route:
- siri: a question or command for the voice assistant (weather, timers, opening apps, facts).
- text: a message the user wants typed into the focused text field.
- voice: something the user wants spoken aloud to people in a voice call.
frontmost_app, if given, is the app the user was in when they started signing. Use it as a hint: a message
while in Messages, Mail, Notes, Slack or a browser is text; while in a call app (FaceTime, Zoom, Discord,
Meet) prefer voice; questions and device commands are siri wherever the user is.
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

    async def interpret(self, glosses, frontmost_app=None):
        """Returns (intent or None, seconds until the tool call, seconds for the full turn, model)."""
        start = time.perf_counter()
        intent, t_intent, model = None, None, None
        request = {"glosses": glosses}
        if frontmost_app:
            request["frontmost_app"] = frontmost_app
        await self.client.query(json.dumps(request))
        async for msg in self.client.receive_response():
            if isinstance(msg, AssistantMessage):
                model = msg.model
                for block in msg.content:
                    if isinstance(block, ToolUseBlock) and block.name.endswith("report_intent") and intent is None:
                        intent, t_intent = block.input, time.perf_counter() - start
            elif isinstance(msg, ResultMessage):
                break
        return intent, t_intent, time.perf_counter() - start, model

