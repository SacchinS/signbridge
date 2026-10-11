# Task 2: 5-sign pilot

Signs: HELLO, YES, NO, THANK-YOU, HELP. Signer: project author (not a fluent ASL signer; learned the signs
from reference videos). Recorder: `record_signs.py`, 2 s takes, landmarks only.

## Session 1 recording quality (2026-10-10)

| Sign | Takes | Frames | Any hand | Left | Right | Both |
|---|---|---|---|---|---|---|
| HELLO | 10 | 518 | 99% | 0% | 99% | 0% |
| YES | 10 | 504 | 100% | 0% | 100% | 0% |
| NO | 10 | 517 | 100% | 0% | 100% | 0% |
| THANK-YOU | 10 | 517 | 87% | 12% | 86% | 11% |
| HELP | 10 | 520 | 87% | 43% | 83% | 38% |

- About 26 frames per second during recording (51-52 frames per 2 s take; one YES take had 36).
- One-handed signs are always tracked as the right hand.
- THANK-YOU starts at the chin; hand detection drops near the face.
- HELP rests a fist on the other palm; the hands occlude each other and both are found in only 38% of frames.
- 50 takes = 1.9 MB of landmarks.
