# 0005 - Golden data per platform; linux-x86_64 is the reference

- **Status:** accepted
- **Date:** 2026-11-02

## Context

The golden test was recorded on Linux x86_64. On a MacBook (arm64), with the
same code, weights and exactly pinned library versions, all 4 cases differed:
case 1 -3 frames, case 2 -12, case 3 -86/+3, case 4 +51 (a second track
crossed tau_base 0.65). Video decoding, YOLO and OSNet produce slightly
different floats on different CPUs and builds, and the pipeline's hard
decisions (DeepSORT gating, tau_base, the 0.9 * tau frame floor) turn small
differences into different selections (AUDIT.md F18).

`make fingerprint` locates the first difference at video decoding (the OpenCV
wheels decode the same H.264 file to different pixels on the two platforms).
After `make golden-baseline` on the Mac, `make golden` passes 4/4 there: the
extraction is faithful on both platforms; only the inputs differ.

## Decision

- Golden data are stored per platform: `tests/golden/data/<sys.platform>-<machine>/`.
- `make golden-baseline` records a platform's data by running repo 2 untouched
  on that machine; `make golden` then requires an exact match on the same
  machine type. That still proves the extraction is faithful: same inputs,
  same platform, same frames.
- linux-x86_64 is the reference platform: CI, the Docker image and the
  Hugging Face Space all run there. Frame F1, the eval gate (Day 24) and every
  number in the README are measured on it.
- `make fingerprint` compares decode / YOLO / OSNet outputs with the reference
  to show where a machine first diverges.

## Consequences

- A Mac is fine for development; numbers that go in the README, the CV or a
  decision note come from linux-x86_64 (CI or Docker). On the Mac, repo 2's
  code scores 85.34% frame F1 versus 84.70% on Linux, so the platform alone
  moves the headline number by more than half a point.
- Decoding is part of the model: if ingest ever runs on more than one platform
  (or OpenCV/FFmpeg is upgraded), the index must be rebuilt and re-evaluated.
- The sensitivity is itself a result: a +51-frame swing from float-level noise
  in case 4 shows how close its tracks sit to tau_base. Worth one line in the
  interview story and a robustness check later (e.g. scores within +-0.01 of a
  threshold).
