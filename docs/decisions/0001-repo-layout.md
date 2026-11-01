# 0001 - Repository layout and boundaries

- **Status:** accepted
- **Date:** 2026-11-01 (before Day 1)

## Context

Both legacy repos are single 600-1200 line scripts that mix model loading,
video I/O, scoring, rendering and web glue (Django / Streamlit). The roadmap
needs the same pieces reused by a batch ingest worker, a search API, an eval
gate and ONNX/INT8 variants of the encoders.

## Decision

- `src/` layout, one installable package `person_search`, Python 3.10, `uv` for
  environments, CPU PyTorch from the PyTorch index.
- Boundaries:
  - `interfaces.py` - `Detector`, `Tracker`, `ImageEncoder`, `TextEncoder` protocols.
  - `ingest/`, `models/` - the only modules that touch YOLO, DeepSORT, torch, CLIP.
  - `core/` - pure functions (scoring, threshold, fusion). No I/O, no torch.
  - `search/` - query logic over observations; does not know where they came from
    (memory today, Qdrant on Day 10).
  - `factory.py` - the only place that maps settings to concrete classes.
  - `config.py` - every path and threshold; nothing hardcoded elsewhere.
- The legacy repos live outside this repo (`../legacy/`) and are never edited.
  `tools/legacy/` holds harnesses that import them by path.
- Weights and data are never committed (`.gitignore`, pre-commit size check).

## Consequences

- An ONNX or INT8 encoder (Days 15-16) is a new class implementing
  `ImageEncoder`; ingest, search and the gate do not change.
- Unit tests run with fakes and synthetic checkpoints, so CI needs no weights.
- Parity and golden tests need local assets and run outside CI until Day 24.
