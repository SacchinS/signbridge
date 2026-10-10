# Task 1: output-route spike results

Script: `spike_routes.py`. Measured 2026-10-09.

```
macOS version / chip: macOS 26.2 (25C56) / Apple silicon (arm64)
Siri route: pending
Text route: pending
Voice route: pending
Haiku 5.5 median latency / valid output rate: 1.08 s to tool call (1.82 s full turn) / 20/20
Surprises:
Recommended cut or fallback:
```

## Intent latency (`python spike_routes.py bench --runs 20`)

- Claude reached through the Claude Agent SDK on a Claude Pro login (`apiKeySource: none`), not the direct API.
- One `ClaudeSDKClient` is kept open for all 20 calls. A cold one-shot `query()` took 7.2 s end to end (1.6 s of API time), so subprocess startup dominates without a persistent client.
- "Time to tool call" is when the `report_intent` tool call arrives; the app can act then. "Full turn" also waits for the model's short reply to the tool result.
- The first 3 calls took 1.36-1.71 s; calls 4-20 settled at 0.98-1.17 s.
- Median time to tool call 1.08 s (gate: under 1.2 s). Valid tool output 20/20 (gate: 95%+).
- Routes chosen: WEATHER TOMORROW -> siri (0.95), ME LATE TEN MINUTE -> text (0.6), HELLO EVERYONE ME READY START -> voice (0.75). Output was identical across repeats.
- Caveat: all 20 calls share one session, so later calls see earlier ones in context. The app should reset the session periodically.
