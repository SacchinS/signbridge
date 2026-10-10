# Task 2: tracking speed results (in progress)

Script: `spike_tracking.py`. Mac: M1, 8 GB RAM, macOS 26.2. mediapipe 1.1.0, opencv 5.0.0, Python 3.12.

Gate: 24+ fps for 2 minutes, hands found in 95%+ of frames (both hands and arms in frame).

## Preliminary 8-10 s runs (nobody signing in frame, so hand rates are not meaningful)

| Mode | Delegate | Resolution | Capture | fps mean | inference median |
|---|---|---|---|---|---|
| holistic | CPU | 1280x720 | blocking read | 12.6 | 53.9 ms |
| holistic | CPU | 640x480 | blocking read | 17.2 | 36.8 ms |
| holistic | GPU | any | - | crash | - |
| split (hands + pose lite) | CPU | 1280x720 | blocking read | 14.1 | 55.9 ms |
| split (hands + pose lite) | CPU | 640x480 | blocking read | 14.4 | 37.6 ms |
| holistic | CPU | 640x480 | threaded | 17.9 | 48.3 ms |

- Raw camera delivers 30 fps at both resolutions when idle.
- GPU (Metal) delegate fails with the Holistic model: 59 of 182 ops unsupported, then
  `TFLGpuDelegateBindMetalBufferToTensor` RET_CHECK failure in `inference_calculator_metal.cc`.
- With a blocking `cap.read()`, any inference over the 33 ms frame interval waits for the next frame,
  roughly halving fps. `LatestFrame` (background capture thread) removes that wait.
- Later runs are unreliable: system load average was 12 on 8 cores (iCloud `fileproviderd` and `cloudd`
  using about 2 cores, plus `siriactionsd`), and the camera then stopped returning frames.
  Needs a rerun on a quieter machine.
