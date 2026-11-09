# 0010 - Model registry and gate set

- **Status:** accepted
- **Date:** 2026-11-09

## Context

Day 8 asks for four model versions (osnet v0/v1, irra v0/v1) to be evaluated
on held-out data and registered with their numbers, so that a CI gate can later
block a worse model. Constraints:

- the weights are large (IRRA is 1 GB) and already have a home (team Drive,
  Hugging Face);
- the evaluation data show real students, so they must stay private;
- the gate must run on a free CPU runner, and give the same numbers on any
  machine (video decoding differs between macOS and Linux, decision 0005);
- the college text-image test split (IRRA's 65.22% Rank-1) has not arrived.

## Decision

- **`configs/models.yaml`** is the in-repo source of truth: each version's
  file, source and SHA-256, and the `production` pins. It is reviewed in PRs
  like code.
- **MLflow registry** (`make register`): one run per version with its gate
  metrics and a small `model_ref.json` (source + SHA-256), registered as
  `person-search-osnet` / `person-search-irra`, alias `production` on the pinned
  versions. The registry records *which file*, by hash; it does not copy the
  weights. The tracking server comes from `MLFLOW_TRACKING_URI`
  (DagsHub: `https://dagshub.com/<user>/person-search.mlflow` with
  `MLFLOW_TRACKING_USERNAME`/`PASSWORD`), else a local SQLite file.
- **Gate set** (`make gate-set`, private HF dataset via
  `scripts/gate_set.py upload`): the campus test split, the 4 test videos as
  cached observations plus a lossless PNG crop of every embedded detection, and
  the text test set, with a SHA-256 manifest. Re-embedding the crops one frame
  per batch reproduces the ingest's vectors exactly, so the gate needs no video,
  no YOLO and no DeepSORT, and gives the same numbers on every OS.
- **Metrics per family:**
  - OSNet: campus mAP/Rank-k (cosine, as served; torchreid's Euclidean protocol
    too, to reproduce the published numbers) and photo frame F1 on the 4 cases.
  - IRRA: text Hit@1/Hit@5/MRR on the 12 descriptions (decision 0007), and the
    image tower's campus mAP as a secondary check.
- **Until the college split arrives**, the text gate is end-to-end Hit@k on the
  12 descriptions. It is coarse (one query = 8 points), so the gate rule for text
  is "no Hit@1 lost", not a percentage threshold.
- **Campus mAP is the OSNet quality score; photo F1 is a regression check.** On
  the 4 test videos the Market-1501 model scores a *higher* photo F1 than the
  campus fine-tune (85.25% vs 84.70%; AUDIT.md F26): 4 cases and thresholds
  tuned for v1 cannot rank models. Campus mAP (29.2% vs 75.4%) can.

## Consequences

- Moving the registry to DagsHub is one environment variable and
  `make register`; nothing is re-computed (`register` reads the evaluation JSON).
- When the college split arrives, add it to the gate set and to
  `eval/gate.py` (`evaluate_text_to_image` already implements the protocol), and
  add college Rank-1 to the text gate.
- The gate set is a private artefact: never upload it to a public repo
  (`gate_set.py upload` refuses a dataset that exists and is public).
