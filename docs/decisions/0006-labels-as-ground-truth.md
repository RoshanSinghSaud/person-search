# 0006 - Use the committed labels as ground truth, as they are

- **Status:** accepted
- **Date:** 2026-11-02

## Context

The 4 `evaluation.json` files reproduce the report's 89.05% F1, but they do not
match the committed output videos frame for frame, and how they were produced
is not documented (AUDIT.md F6). Re-labelling the 4 videos would take a day and
would replace the team's numbers with ours; asking for the provenance is not
going to happen.

## Decision

- The labels stay as committed (`tests/golden/data/labels/`) and are not edited.
- **Ground truth = TP + FN** of each file: the sampled frames in which the
  target appears. Every frame-level metric for a new run is computed against
  that set (`eval/frame_metrics.metrics_for_selection`).
- **TP + FP is not used** as "what the report's run selected". It only
  reproduces the report's own counts (`metrics_from_labels`, the 89.05% check).
- Regression is measured against the golden frames (decision 0002), not against
  the labels, so label noise cannot fail a build.
- The 89.05% is quoted only as the report's result, with this caveat.

## Consequences

- All F1 numbers in this repo share one ground truth and are comparable with
  each other (84.70% on linux-x86_64 for repo 2's code).
- If a label turns out to be wrong, fix it in a separate commit with a note
  listing the changed frames, and re-run every reported number. Do not fix
  labels silently.
- Day 9's text targets come from the golden track IDs, not from these labels.
