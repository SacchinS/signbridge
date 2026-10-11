# PopSign / Kaggle asl-signs: viability check (2026-10-10)

## Two versions of the data

| | PopSign ASL v1.0 (Georgia Tech) | Kaggle "asl-signs" (Google ISLR competition, 2023) |
|---|---|---|
| Content | Video, 1944x2592 phone selfie clips | MediaPipe Holistic landmarks only (543 per frame: face 468, pose 33, hands 21+21), parquet |
| Size | 1.1 TB | about 40 GB zipped, about 57 GB extracted (community write-ups, unverified) |
| Signs | 250 | 250 (same game vocabulary) |
| Signers | 47 Deaf adults (31 train / 8 val / 8 test) | ~94,481 training sequences; participant count not verified |
| License | CC BY 4.0 (dataset page) | **Unverified**: Kaggle competition data terms, need to read the rules tab |
| Stated limits | Research use; suited to isolated sign recognition including phone apps; **not** suited to continuous recognition or sign-to-English translation | Unverified |
| Baseline | LSTM 84.2% on held-out signers | |

## Fit for SignBridge

- The Kaggle landmark version is the only one that fits: same MediaPipe Holistic landmark layout as ours,
  no video processing. Our features drop the face mesh, so face points are simply ignored.
- Disk: the full set (40-57 GB) does not fit in the 17 GB free. The Kaggle API can download single files,
  and `train.csv` maps each sequence file to its sign, so a per-sign subset is possible (rough estimate
  ~0.6 MB per sequence, so 100 sequences x 40 signs is about 2.4 GB; unverified).
- Domain gap: phone selfie camera in portrait vs. a Mac webcam in landscape, and 2023 legacy Holistic vs the
  2026 Tasks Holistic. Shoulder normalization should absorb most of the framing difference; untested.
- Vocabulary is a children's game list. Pilot coverage: HELLO, YES, NO, THANKYOU present; **HELP absent**.
  Useful command-ish signs present: please, wait, open, close, look, find, go, finish, now, later, time,
  tomorrow, yesterday, morning, night, rain, snow, sun, hot, where, who, why, callonphone, tv, radio, read,
  talk, say, listen, hear, loud, quiet, home, store, water, food, drink, sleep, wake, up, down, like.
  Absent: help, stop, weather, message/send, music, play, volume.

## Verdict

Viable as pretraining data for a subset of signs, if the Kaggle license allows our use. Not a replacement for
the user's own recordings: our 5-sign pilot already scores 97% on held-out sessions of the same signer, and
the demo signer is the user. Main benefits: (1) fewer own takes needed per sign when scaling to ~40 signs,
(2) robustness when a judge or another person signs, (3) a large pool of "other signs" to train a
"not one of my commands" class.

## Next step (cheap experiment)

Needs a Kaggle account that has accepted the asl-signs competition rules, with an API token in
`~/.kaggle/kaggle.json` (never in the repo). Then: download `train.csv`, fetch ~100 sequences each for HELLO,
YES, NO, THANKYOU, and measure (a) a model trained only on PopSign tested on our sessions, and (b) PopSign
pretraining + our session 1, tested on session 2.

Sources: https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/ ,
https://proceedings.neurips.cc/paper_files/paper/2023/hash/00dada608b8db212ea7d9d92b24c68de-Abstract.html ,
https://community.element14.com/technologies/ai-machine-learning/b/blog/posts/mediapipe_2d00_02_2d00_datasets
