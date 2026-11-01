# 0002 - Default configuration = repo 2 as committed

- **Status:** accepted
- **Date:** 2026-11-01 (Day 2 evidence gathered before Day 1)

## Context

Four sources give four configurations (AUDIT.md section 3). One ingest pass
fixes one detection/tracking setting for every query, so the index must be
built with the setting the regression tests are pinned to.

Evidence (`tools/legacy/reproduce_repo2.py`, `tools/legacy/extract_drawn_boxes.py`):

- Repo 2's `run_reid`, untouched, with its committed values (YOLO conf 0.4,
  tau_base 0.65, delta 0.1, DeepSORT 30/3 with library-default
  max_cosine_distance 0.2) selects **exactly** the frames boxed in the
  committed `test_case_004/output_annotated.mp4`, and **every** boxed frame in
  cases 1-3, always on the same tracks.
- In cases 1-3 it also selects frames the committed videos do not box
  (81, 42, 104). Re-ingesting with the report's det 0.5 / max_cosine_distance
  0.3 and sweeping tau_base shows each committed video comes from a different
  setting: case 1 from det 0.5 / 0.3, cases 2-3 from tau_base 0.7, case 4 from
  tau_base 0.65 (AUDIT.md section 5). No single setting reproduces all four.
- Of the settings that find the target in all 4 cases, repo 2's code has the
  best frame F1 (84.70% vs 84.23% for det 0.5 / 0.3). tau_base 0.7 scores
  85.32% but misses case 4 entirely.
- Neither the re-run nor the committed videos give the report's 89.05% F1
  against the committed labels: re-run 84.70%, committed videos 83.92%. The
  labels were made on a run whose output is not in the repo (AUDIT.md F6).

## Decision

- Code defaults and `configs/report_run.yaml` = repo 2 as committed
  (det 0.4, max_cosine_distance 0.2, tau_base 0.65, delta 0.1).
- The golden test pins the re-run of that code (`tests/golden/data/*.json`),
  not the labels and not the committed videos.
- `configs/paper.yaml` keeps the values written in the paper/report for
  comparison runs only.

## Consequences

- Regression baseline for photo search: **84.70% frame F1** on the 4 cases
  (labels' TP + FN as ground truth). Quote the paper's 94.23% / the report's
  89.05% as the research results; never compare a new run against them.
- Any change to these values moves the golden frames and needs its own note.
- tau_base 0.65 was probably chosen with case 4 in view (AUDIT.md section 5,
  point 4). Treat the 4 test cases as a regression suite, not as a held-out
  test of the threshold; a fair threshold estimate needs more videos.
