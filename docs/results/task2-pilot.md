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

## Pipeline sanity check (session 1 only)

`python train_pilot.py --sanity`: random 70/30 split within session 1, TinyConv (2 Conv1D layers, width 64),
32 resampled frames x 197 features (upper-body pose + both hands, shoulder-normalized, plus hand-missing flags).
15/15 correct. This only shows the pipeline works; same-session splits overstate accuracy. The real number
comes from leave-one-session-out once sessions 2-4 are recorded.

## Informal live check (`live_classify.py`, model trained on all of session 1)

New sitting, free-form: the user signed each of the 5 signs once. Guesses: THANK-YOU 98%, HELLO 86%,
NO 99%, HELP 97%, YES 96%, all reported correct by the user. Five tries with no logged ground truth, so this
is a gut check, not the pilot accuracy. The scored number still needs a held-out session
(`record_signs.py --session s2 --model models/pilot.pt`).

## Held-out session results (`python train_pilot.py`)

Session 2 was recorded live with the session-1 model (`record_signs.py --session s2 --model models/pilot.pt`),
so its score is a true live test. Leave-one-session-out over both sessions, 60 epochs, seed 0:

| Held-out session | Accuracy |
|---|---|
| s1 (trained on s2) | 100% (50/50) |
| s2 (trained on s1), live | 94% (47/50) |
| **All** | **97% (97/100)** |

| Sign | Accuracy |
|---|---|
| HELLO | 95% (19/20) |
| HELP | 100% (20/20) |
| NO | 100% (20/20) |
| THANK-YOU | 100% (20/20) |
| YES | 90% (18/20) |

Confusion pairs (true -> predicted): YES -> NO x2, HELLO -> THANK-YOU x1.

- The offline s1 -> s2 run reproduces the live result exactly (47/50, same three errors), so live and offline
  feature paths match.
- YES -> NO is the pair that matters most: they are the likely confirm/cancel signs. Before using them for
  confirm/cancel, require high confidence or pick a more distinct pair.
- HELLO -> THANK-YOU: both start near the face and move outward.

### Caveats

- One signer, who is the person the model is tested on, and not a fluent ASL signer. This says nothing
  about other signers; reported unseen-signer ceilings are 63-84%.
- **Seed variance (found 2026-10-10 during the PopSign check):** the 97% above is seed 0. The same
  evaluation with seeds 1 and 2 gives 79% and 93%, so **about 90% mean** is the honest number. With 20 takes
  per sign, training on my data alone is unstable. PopSign pretraining fixes this; see
  [popsign.md](../research/popsign.md).
- Two sessions, both on 2026-10-10. 20 takes per sign, short of the 40+ target; sessions 3-4 pending.
- Takes are pre-segmented 2 s windows. No "not a sign" class yet, so the model always picks one of the 5.
