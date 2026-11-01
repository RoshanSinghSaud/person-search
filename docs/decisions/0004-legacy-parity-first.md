# 0004 - Reproduce first, change behind a flag

- **Status:** accepted
- **Date:** 2026-11-01

## Context

The roadmap's rule: any change that moves an evaluation number needs a decision
note. Several legacy behaviours look like bugs but shaped the published
numbers.

## Decision

The extracted code reproduces repo 2 exactly (the golden test pins it).
Behaviours kept on purpose, each with a switch or a seam for later:

| Behaviour | Where | Why it might change |
|---|---|---|
| Confirmed tracks that were *not* matched this frame are still recorded, with their Kalman-predicted box (`include_unmatched_tracks=True`) | `ingest/tracker.py` | Predicted boxes can drift onto a neighbour: a likely source of FP frames (AUDIT.md F11) |
| Frames before a track's first similarity take that first similarity (back-fill) | `core/scoring.forward_fill` | Could accept frames never compared with the query |
| `x / (norm + 1e-6)` normalisation instead of `F.normalize` | `models/preprocessing.l2_normalize` | Harmless; kept so parity is exact |
| Detection on a <= 640 px wide copy, tracking on the full frame | `ingest/detector.py` | Small far-away people are missed at 640 px |
| Long text queries lose EOT to the 8 prompt tokens (read at the last slot) | `models/irra_encoder.py` | Logged as a warning; affects descriptions over ~67 BPE tokens |

## Consequences

Turning any of these off is an experiment: run `make golden` (it will fail by
design), record the new frame-level F1 on the 4 test cases, and add a decision
note before changing the default.
