# Sign vocabulary (draft for review, 2026-10-12)

Goal: about 40 signs that cover the demo commands on all three routes (Siri, text, voice). Prefer signs
in PopSign (Kaggle asl-signs), because a PopSign-only model already scores 95% on my signing with no
takes from me. Signs PopSign lacks need 20-40 takes each, so keep that list short.

Claude turns the gloss list into a sentence ("RAIN TOMORROW" -> "Will it rain tomorrow?"), so signs are
building blocks, not fixed commands.

## Control gestures (not ASL signs; rule-based, no ML)

These gate every action, so they must be near-perfect and must never be confused with a command sign.
Detecting them from hand-landmark geometry (finger extension, palm direction, hold time) is far more
reliable than a classifier and needs no data.

| Gesture | Proposed form | Why |
|---|---|---|
| Wake | Both open palms to the camera, held apart and still for 0.7 s | Two-hand static pose; rare by accident. The handoff asks for a distinct two-part sign with a timeout |
| Confirm | Thumbs up, held still 0.5 s, other hand away | Static, unambiguous, common meaning |
| Cancel | Thumbs down, same rules | Mirror of confirm; static, so it's as reliable |

Implemented in `signbridge/gestures.py`. Checked against 6 minutes of my recorded signing (all 100 takes
and 16 sequence attempts, 9,556 frames): 1 false trigger, a HELP take read as confirm. Before the
stillness and separation rules there were 16 (THANK-YOU and HELP looked like wake, HELP like confirm:
the ASL sign HELP *is* a thumbs-up on the other palm). Gestures are only listened for in the state that
needs them (wake when idle, confirm/cancel when confirming), which limits the HELP risk further.
Not yet tested on real gestures: needs a live session.

Alternative: real ASL signs (e.g. YES / NO / FINISH). Rejected for control because YES and NO are the
most confused pair in testing (YES -> NO, NO -> HELLO), and a misread confirm runs a command.

## Command signs

Legend: P = in PopSign; R = needs my recordings (20-40 takes).

### Siri route
| Sign | Src | Example phrase -> sentence |
|---|---|---|
| RAIN | P | RAIN TOMORROW -> "Will it rain tomorrow?" |
| SUN | P | SUN TOMORROW -> "Will it be sunny tomorrow?" |
| HOT | P | HOT NOW -> "What's the temperature right now?" |
| TIME | P | TIME NOW -> "What time is it?" |
| TOMORROW | P | |
| MORNING | P | WAKE MORNING -> "Set an alarm for 7 in the morning" |
| WAKE | P | |
| CALLONPHONE | P | CALLONPHONE MOM -> "Call Mom" |
| MOM | P | |
| DAD | P | |
| RADIO | P | RADIO -> "Play music" (stand-in for MUSIC) |
| LOUD | P | LOUD -> "Turn the volume up" |
| QUIET | P | QUIET -> "Turn the volume down" |
| WEATHER | R | Optional: more natural than RAIN/SUN for "what's the weather" |

### Text route (Messages, Notes, browser)
| Sign | Src | Example |
|---|---|---|
| HELLO | P | HELLO -> "Hello!" |
| THANKYOU | P | |
| LATER | P | HOME LATER -> "I'll be home later." |
| HOME | P | |
| HUNGRY | P | HUNGRY PIZZA -> "I'm hungry, want pizza?" |
| PIZZA | P | |
| FOOD | P | |
| WHERE | P | WHERE YOU... -> "Where are you?" (uses WHERE + context) |
| WHO / WHY | P | question words |
| GO | P | GO STORE -> "I'm going to the store." |
| STORE | P | |
| FINISH | P | FINISH -> "I'm done." |
| SICK | P | SICK HOME -> "I'm sick, staying home." |

### Voice route (video call / voice chat)
| Sign | Src | Example |
|---|---|---|
| YES | P | YES -> "Yes." |
| NO | P | NO -> "No." |
| WAIT | P | WAIT -> "Hold on a second." |
| BYE | P | BYE -> "Bye, everyone!" |
| SAME | P | SAME -> "I agree." |
| LISTEN / HEAR | P | NOT HEAR -> "I can't hear you." |
| NOT | P | |
| PLEASE | P | PLEASE WAIT -> "Please wait a moment." |
| HELP | R | HELP -> "Can someone help me?" Conflicts with the confirm gesture (it is a thumbs-up on the other palm); consider dropping |

### Routing hint (optional)
Claude infers the route from content ("call Mom" -> Siri). If that misroutes in testing, add a route
prefix: point to the screen = text, touch the lips = voice. Decide after the intent tests.

## Count

About 36 PopSign signs + 1-2 recorded (HELP, maybe WEATHER) + 3 rule-based control gestures.

## Known risks

- Confusable pairs to test once trained: YES/NO, HELLO/THANKYOU (both start at the face), MOM/DAD
  (same handshape, chin vs forehead), WHO/WHY, LOUD/NOISY.
- More classes lower accuracy. Pilot accuracy was on 4 signs; measure again at 40 before the demo.
- PopSign is a children's-game vocabulary; some adult phrasing needs Claude to fill in.
