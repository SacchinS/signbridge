# SignBridge: Claude Code Handoff

Read this whole file first. Then do **Task 0 (GitHub repo)**, then **Task 1**, and stop to report. Do not build ahead.

## 1. What we are building

A Mac menu-bar app that lets a Deaf or hard-of-hearing user control any voice-driven app by signing ASL to the webcam.

Pitch line: "Anything a hearing person can do with their voice on a Mac, you can do by signing."

Target: CalHacks demo, Oct 23-25, 2026, San Francisco. Solo builder. Goal is a prize, a strong live demo, and networking. Today is Oct 9, so about 14 days remain.

Why it is novel: Google's SL2T (Aug 2026) is ASL-to-text dictation on Pixel 11 only, in Gboard and Live Transcribe, with no desktop version or API. Ours is a cross-app layer on a Mac: Siri, text fields, and voice chat.

## 2. Pipeline

```
webcam -> MediaPipe Holistic landmarks (on-device)
       -> small PyTorch classifier (~40 command signs)
       -> Claude intent layer (tool call: sentence, route, confidence)
       -> confirm / cancel sign
       -> output route: Siri | text paste | voice (virtual mic)
```

Wake-sign: a distinct two-part sign with a timeout, so random hand movement does nothing.

## 3. Stack (verified Oct 9, 2026)

| Layer | Choice | Notes |
|---|---|---|
| App shell | Python 3.11+, `rumps`, `pyobjc` | Needs Accessibility permission. **Do not sandbox**: CGEvent.post is blocked in the App Sandbox |
| Tracking | MediaPipe Holistic Landmarker, Python Tasks API | Model: `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`. Tasks supports Python 3.9+ |
| Recognizer | PyTorch Conv1D or small transformer on landmark windows | Reported ceilings: 74.7% top-1 unseen signers (asl-realtime), 84.2% (PopSign LSTM), 63% (ASL Citizen, 2,731 signs). Measure our own |
| Intent | `claude-haiku-5-5` (primary), `claude-haiku-4-5-20251001` (fallback), `claude-sonnet-5-5` on low confidence | Do **not** pass `temperature`, `top_p` or `top_k`: they return 400 on 4.7+ models |
| Siri route | macOS 27: Cmd-Space, paste, Return. macOS 26: custom Siri shortcut + Type to Siri | macOS 27 is Apple silicon only. Paste into the Spotlight "Search or Ask" box is **unconfirmed**, so test it |
| Text route | Pasteboard + CGEvent Cmd-V, then restore clipboard | Fallback: AppleScript System Events keystrokes |
| Voice route | `say -a "BlackHole 2ch"` into a virtual mic | `brew install blackhole-2ch` (v0.7.1). Later: ElevenLabs Flash v2.5 |
| Data | ASL Citizen subset + our own recordings | `ASL_Citizen.zip` is 42.8 GB. Commercial-use terms unconfirmed |

Avoid: Cartesia `sonic-2` (retires Oct 20). Use `sonic-3.6` if Cartesia is wanted at all.

## 4. Environment facts to check first

Run and record: `sw_vers`, `uname -m`, `python3 --version`, `brew --version`, free disk space.

- If the Mac is Intel, it cannot run macOS 27. Use the macOS 26 Siri flow.
- Use a venv. Install: `mediapipe opencv-python pyobjc rumps anthropic torch numpy`.
- The user must grant Camera and Accessibility permissions to the terminal or Python. Tell them when a prompt is needed.
- Needs `ANTHROPIC_API_KEY` in the environment. Never write the key to a file or commit it.

## 5. Task 0 (do this before any code): GitHub repo and commit practice

Clean history and documentation matter for judges and for the user's portfolio. Set this up first.

### Setup

1. Check `gh auth status`. If not logged in, ask the user to run `gh auth login`. Do not try to work around it.
2. Create the repo before writing code: `git init`, then `gh repo create signbridge --public --source=. --remote=origin` (ask the user whether they want it public or private if unsure).
3. Add these files in the first commit:
   - `README.md`: one-paragraph pitch, the pipeline diagram from section 2, how to run, current status.
   - `.gitignore`: `venv/`, `.venv/`, `__pycache__/`, `.env`, `*.task`, `data/`, `recordings/`, `*.zip`, `*.pt` (large trained weights), `.DS_Store`. The 42.8 GB dataset, recordings, model files and API keys must never be committed.
   - `docs/HANDOFF.md`: a copy of this file.
   - `.env.example`: variable names only, no values.
4. Push to `main` and confirm it appears on GitHub.

### Commit rules

- **Do not add yourself as co-author or contributor.** No `Co-Authored-By:` line, no "Generated with Claude Code" or similar footer or tagline, in commit messages or in PR descriptions. Commits are authored by the user alone. If any tool or setting adds such a line, remove it before committing and check with `git log -1 --format=%B`.
- Use the user's existing git identity (`git config user.name` and `user.email`). Do not change it. If it is unset, ask the user.
- **Commit small and often**: after each working unit (a script that runs, a measured result, a doc update). Aim for a commit at least every 30-60 minutes of progress, and always before starting something risky.
- **Push after every commit or small batch of commits**, at minimum at the end of each task and before stopping to report. Never leave work only on the local machine.
- Message format: imperative, short subject under 72 chars, optional body that says why.
  - Good: `Add Siri route spike with 10-run pass/fail log`
  - Bad: `update`, `fix stuff`, `wip`
- Use conventional prefixes if natural: `feat:`, `fix:`, `docs:`, `test:`, `chore:`.
- Work on `main` for now (solo, short timeline). Do not force-push or rewrite pushed history.
- Never commit secrets. If one is committed by mistake, stop and tell the user so the key can be rotated. Do not just delete it in a later commit.

### Keep documentation current

- Update `README.md` status when a task finishes.
- Save each spike's measured results in `docs/results/` (e.g. `docs/results/task1-routes.md`) using the report format in Task 1, and commit it with the code that produced it.
- Add short docstrings or a header comment to each script saying what it tests and how to run it.

## 6. Task 1 (after Task 0): output-route spike, no ML

**Why first:** it can kill or confirm the pitch in a couple of hours with no data or training. If a synthesized keystroke cannot reach Siri, the headline demo changes. The recognizer is validated in Task 2.

Build `spike_routes.py`. A keypress stands in for a recognized sign.

1. **Fake-sign trigger.** Pressing `1`, `2` or `3` in the terminal selects a canned sign sequence, e.g. `["WEATHER", "TOMORROW"]`.
2. **Intent call.** Send the glosses to `claude-haiku-5-5` with a tool schema returning `{sentence, route, confidence}`. Time each call. Print median latency over 20 runs.
3. **Siri route.** Post Cmd-Space, wait about 400 ms, paste the sentence, press Return. Then run the macOS 26 variant if the Mac is on 26.
4. **Text route.** Paste a sentence into Notes, Messages and a browser text field, then restore the clipboard.
5. **Voice route.** `say -a "BlackHole 2ch" "<sentence>"`. Set BlackHole as the input in a voice-chat or dictation app and see whether it transcribes. Also try macOS Voice Control ("open Safari").

### Pass gates (targets, adjust after first measurement)

| Test | Pass |
|---|---|
| Siri route | 9 of 10 runs end with Siri answering the typed question |
| Text route | 10 of 10 in all three apps, clipboard restored |
| Voice route | Voice app transcribes 5 of 5 sentences |
| Haiku latency | Median under 1.2 s, valid tool output 95%+ |

### Report back in this format

```
macOS version / chip:
Siri route: X/10, notes:
Text route: Notes X/10, Messages X/10, Browser X/10
Voice route: X/5, Voice Control works: yes/no
Haiku 5.5 median latency / valid output rate:
Surprises:
Recommended cut or fallback:
```

### If a test fails

- Siri route fails: try AppleScript System Events keystrokes, then stop and report. Cut line is to demo text and voice routes only and say so honestly.
- Voice route fails: try a Multi-Output plus Aggregate device. Otherwise move the voice route to P1.
- Haiku too slow: shorten the prompt, enable prompt caching, template known commands.

## 7. Task 2: tracking and a 5-sign pilot (after Task 1 is reported)

1. **Tracking speed.** Holistic Landmarker on the webcam, both hands and arms in frame. Log fps and hand-detection rate. Gate: 24+ fps for 2 minutes, hands found in 95%+ of frames. Fallbacks: hands and pose separately, or the browser tasks-vision version.
2. **Recorder tool.** Save landmarks only, not video. Label by sign, session and date. Record 40+ takes per sign.
3. **Pilot.** Pick 5 clearly different signs. Train a tiny Conv1D. Hold out whole sessions, not random clips. Report accuracy per sign and the confusion pairs.

Segmentation idea to prototype: cut a clip when the hands leave and return to a rest position.

## 8. Later phases (do not start without being asked)

Scale to about 40 signs. Add the wake-sign and confirm/cancel sign. Add the misread-sign log. Then the demo script and a backup video.

**Stretch (P2):** sign-driven computer agent ("delete the latest file in my downloads folder"), signed replies via an avatar, and other items in the product doc.

Schedule gates, cut lines and the full MVP list live in the plan doc: https://claude.ai/code/artifact/3c32a202-74e4-495f-9153-9d456ded7336 and the card view https://claude.ai/artifact/XocKpP5WXdbcERhXUggHfr

## 9. Ground rules

- Test before assuming. Anything under "unconfirmed" is unknown until measured.
- Keep the project small and runnable at every step. One script per spike, committed and pushed with its results (see Task 0).
- **Never add Claude or yourself as a co-author, contributor or footer in any commit or PR.**
- Report measured numbers, not impressions.
- Safety: the app must never act on an unconfirmed destructive command. Confirm sign first.
- Do not claim the app works "for any app". Say "any voice-driven app".
- Ask the user before installing system-level software (BlackHole, drivers) or changing System Settings.
- Never put real credentials in code, logs or commits.

## 10. Unverified / known gaps

- Whether a synthesized Cmd-Space and paste reach Siri AI reliably on macOS 27.
- Whether Siri or Voice Control accept audio from a virtual mic.
- BlackHole on macOS 27 (no reports found either way).
- `rumps` and `pyobjc` on macOS 27.
- Whether Haiku 5.5 tool-call output matches Haiku 4.5's.
- Current `mediapipe` version and any Mac-specific crashes with the holistic model.
- ASL Citizen commercial-use license terms.
- Whether the 40 target command signs exist in ASL Citizen's gloss list.
