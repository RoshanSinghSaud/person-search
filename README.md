# Person Search

Find a person in indexed video **by photo** (OSNet, from the ICCVBIC 2026 paper),
**by description** (IRRA, model 2) or **by both**. This repo turns the team's
final-year system into a production service: index once, search many.

[![CI](https://github.com/RoshanSinghSaud/person-search/actions/workflows/ci.yml/badge.svg)](https://github.com/RoshanSinghSaud/person-search/actions/workflows/ci.yml)

> **Status: Weeks 1-2 done.** Week 1 extracted both legacy repos into modules
> with parity and golden tests ([AUDIT.md](AUDIT.md)). Week 2 added evaluation
> for photo and text queries, a Qdrant index holding both vectors, indexed
> photo, text and hybrid search (exactly equal to a full scan), the first
> end-to-end text numbers and per-stage timings
> ([docs/results/week2.md](docs/results/week2.md)). The API, ONNX/INT8 and
> deployment follow the plan.

## How it works

```mermaid
flowchart LR
    V[video] --> S[12 fps sampler] --> D[YOLO11n] --> T[DeepSORT]
    T -->|every 3rd frame,<br/>crops batched per frame| O[OSNet 512-d]
    T --> I[IRRA image 512-d]
    O & I --> P[(Qdrant: one point per detection<br/>video, track, frame, bbox,<br/>vectors osnet + irra)]
    Q1[photo] --> QO[OSNet] --> PS[top-50% track score<br/>+ dynamic threshold] --> R1[frame segments]
    Q2[description] --> QI[IRRA text + prompt tokens] --> TS[top-50% track score] --> R2[ranked tracks]
    Q1 & Q2 --> HY[late fusion of normalised scores] --> R3[ranked tracks]
    P -.-> PS & TS & HY
```

Photo results are accepted automatically (the paper's rule); text and hybrid
results are ranked for a person to review. Every query first retrieves
detections from the index, then rescores whole candidate tracks; the retrieval
cut-offs are chosen so the answer always equals a full scan (decision 0008).

## Repository layout

```
src/person_search/
  config.py          typed settings (pydantic-settings): env > .env > YAML profile > defaults
  interfaces.py      Detector / Tracker / ImageEncoder / TextEncoder protocols
  factory.py         settings -> concrete components (the only place that knows them)
  pipeline.py        run(video, photo) / run_text(video, text): the legacy flow, rebuilt
  types.py errors.py timing.py logging_utils.py render.py cli.py
  ingest/            video.py (sampler) detector.py (YOLO11n) tracker.py (DeepSORT) pipeline.py
  models/            osnet.py (vendored, MIT) osnet_encoder.py irra_encoder.py checkpoint.py preprocessing.py
  core/              scoring.py (paper's scoring + threshold) fusion.py (late fusion, legacy sum)
  search/            photo.py text.py hybrid.py results.py (in memory); indexed.py (over the index)
  index/             base.py (VectorIndex) qdrant.py (local mode or server) memory.py (fake) points.py
  eval/              frame_metrics.py reid.py golden.py text_e2e.py (Hit@k) campus.py gate_set.py gate.py
  api/               Day 17 (planned contents in its __init__)
configs/             report_run.yaml (default = repo 2's final code) paper.yaml models.yaml (model versions)
scripts/             build_index eval_search gate_set eval_models profile_ingest week2_report fetch_weights
tests/unit/          unit tests, no weights needed (CI)
tests/parity/        new modules vs legacy code (make parity)
tests/golden/        pipeline.run vs repo 2 on the 4 test cases (make golden)
tools/legacy/        Day 2 harness, golden builder, drawn-box extractor (import legacy by path)
tools/analysis/      scoring sweeps, track contact sheets, label sensitivity
docs/decisions/      one note per decision (0007-0010: Week 2)
docs/results/        week2.md, generated from the run outputs
```

## Quick start

```bash
# 1. environment (Python 3.10, CPU PyTorch)
uv sync --group dev --group eval   # or: make setup
# 2. legacy repos for parity/golden tests, outside this repo (never edit them)
make legacy                    # -> ../legacy/{person-reid-clip,amrit-pbvs}
# 3. weights -> models/ (team Drive + Hugging Face), prints SHA-256s
make weights
# 4. tests
make test                      # unit, ~30 s
make parity                    # OSNet + IRRA vs legacy, ~4 min
make golden                    # 4 test cases vs repo 2, ~10-20 min on CPU
```

```bash
export PS_CONFIG_FILE=configs/report_run.yaml
person-search photo  --video ../legacy/amrit-pbvs/test_case_004/test_video.mp4 \
                     --photo ../legacy/amrit-pbvs/test_case_004/query.jpg --out outputs/tc4 --render
person-search text   --video ../legacy/amrit-pbvs/test_case_001/test_video.mp4 \
                     --query "a man in a black jacket and blue jeans"
person-search ingest --video ../legacy/amrit-pbvs/test_case_002/test_video.mp4 --out outputs/tc2.npz
person-search eval-frames --cases-dir tests/golden/data/labels      # the report's 89.05% F1
person-search config                                                # resolved settings
```

Week 2, in order (outputs go to `../legacy/runs/`; the campus split is
expected at `../custom_reid_dataset`):

```bash
make ingest-cases      # 4 videos, OSNet + IRRA vectors          (~25 min on 2 CPUs)
make index             # Qdrant local mode; a re-run adds 0 points
make clipreid-queries  # repo 1's photo embeddings, for its hybrid (comparison only)
make eval-search       # Days 11-12: indexed photo / text / hybrid on the 4 cases
make gate-set          # Day 8: campus split + cached crops + text set
make eval-models       # Day 8: osnet v0/v1, irra v0/v1 on the gate set
make register          # Day 8: MLflow registry (MLFLOW_TRACKING_URI for DagsHub)
make profile           # Day 13: time per ingest stage
uv run python scripts/week2_report.py   # -> docs/results/week2.md
```

Every setting can be overridden with `PS_<SECTION>__<FIELD>`, e.g.
`PS_SCORING__BASE_THRESHOLD=0.7`. See `.env.example`.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'person_search'` on macOS.** The
  `.pth` files in `.venv` got macOS's `hidden` flag, and Python skips hidden
  `.pth` files. Check with `ls -lO .venv/lib/python3.10/site-packages | grep pth`;
  fix with `make fix-pth` (`make setup` runs it). The flag can come back, so
  every `make` target imports from `src/` directly and is not affected; only
  `uv run person-search ...` typed by hand needs `make fix-pth` again.
- **`make golden` fails on a Mac with missing/extra frames.** Expected before a
  Mac baseline exists: the shipped golden data are for linux-x86_64. Run
  `make golden-baseline` once, then `make golden`.
- **`(base)` in the prompt.** Run `conda deactivate`; uv manages its own Python 3.10.
- **Google Drive refuses the download** in `make weights`. Download the folder
  linked in repo 2's README by hand and place the files in `models/` under the
  names in `models/README.md`.

## Results

### Parity with the legacy code (`make parity`)

| Module | Compared with | Result |
|---|---|---|
| `OSNetEncoder` (campus v1, Market v0) | repo 2 `FeatureExtractor` | min cosine 0.99999994 on 20 crops; same rankings |
| `IRRAEncoder` image | repo 1 `extract_image_embedding_for_text_query` | min cosine 0.99999988 |
| `IRRAEncoder` text (with prompt tokens) | repo 1 `extract_text_embedding` | min cosine 1.00000000 on 5 descriptions; same top-10 |

### Golden test (`make golden`)

`pipeline.run(video, photo)`, rebuilt from the new modules, selects **exactly the
same frames on the same track IDs** as repo 2's untouched `run_reid` on all 4
test cases (missing 0, extra 0), on Linux x86_64.

Golden data are per platform, because decoding and float arithmetic differ
slightly between machines and the paper's hard thresholds amplify that
(AUDIT.md F18). On a new machine type, record its baseline once:

```bash
make fingerprint        # where does this machine first differ from linux-x86_64?
make golden-baseline    # run repo 2 untouched here, save tests/golden/data/<platform>/
make golden             # pipeline.run must match it exactly
```

### Day 2 numbers (photo query, frame-level, labels' TP + FN as ground truth)

| Run | System F1 | Precision | Recall | Hit rate |
|---|---:|---:|---:|---:|
| Report (counts in the committed `evaluation.json`) | 89.05% | 93.55% | 84.96% | 4/4 |
| Committed `output_annotated.mp4` files, re-scored | 83.92% | 94.27% | 75.61% | 4/4 |
| **Repo 2 re-run = this repo's default (golden baseline)** | **84.70%** | 85.48% | 83.93% | 4/4 |
| Same, tau_base 0.7 (the report's stated value) | 85.32% | 94.86% | 77.51% | 3/4 |

- The 89.05% cannot be reproduced from what is in the repos: the labels were
  made on a run whose output is not committed, and the four committed videos
  were themselves made with at least three different settings.
- Case 4 is only found with tau_base 0.65; at 0.7 it is "no match". Ask Amrit
  how 0.65 was chosen before an interviewer does (AUDIT.md section 5).
- Quote 94.23% (paper) and 89.05% (report) as the research results. Compare
  every new run against **84.70%**.

Speed (2 vCPU container, other jobs running - indicative only):

| Measurement | Value |
|---|---|
| One text query, repo 1 untouched, case 1 (575 frames) | 408 s (16 s model loading) |
| One photo query, repo 2 untouched, cases 1-4 | 194 s / 414 s / 448 s / 121 s |
| Ingest, this repo, case 4, OSNet + IRRA, batched | 207 s = 1.7 frames/s; IRRA 47%, OSNet 23%, DeepSORT 16%, YOLO 13% |

### Week 2 (Days 8-14): evaluate, index, search

Full tables, generated from the run outputs: [docs/results/week2.md](docs/results/week2.md).

**Index once, search many (Days 10-11).** Each video is ingested once into a
Qdrant collection (one point per tracked detection, OSNet and IRRA vectors on
every 3rd frame; 16,483 points for the 4 test videos; re-ingesting adds 0).
Photo search over the index selects **exactly the golden frames** on all 4
cases (system frame F1 84.70%, unchanged):

| Case | Indexed photo query (p50) | Reprocessing the video (`pipeline.run`) |
|---|---:|---:|
| 1 | 0.26 s | 184 s |
| 2 | 0.52 s | 396 s |
| 3 | 0.37 s | 662 s |
| 4 | 0.24 s | 118 s |

(2-vCPU container, 1 PyTorch thread; the indexed time includes running OSNet on
the photo, about 0.2 s.)

**First end-to-end text numbers (Days 9, 12).** 12 witness descriptions (3 per
case), written from `query.jpg` before any text search ran:

| Query | Hit@1 | Hit@5 |
|---|---:|---:|
| text only (IRRA) | 8/12 | 11/12 |
| text, detailed descriptions only | 4/4 | 4/4 |
| text, short descriptions only | 1/4 | 3/4 |
| photo + text, late fusion (alpha 0.5) | 12/12 | 12/12 |
| photo + text, repo 1's embedding sum | 12/12 | 12/12 |
| photo only (OSNet), 4 queries | 4/4 | 4/4 |

The photo alone already finds all 4 targets, so this test set cannot show what
hybrid adds or tell late fusion from repo 1's mixed-space sum (decision 0009,
AUDIT.md F22). Indexed text and hybrid search equal a full scan for every query.

**Where ingest time goes (Day 13, case 2):** IRRA image 51%, OSNet 25%,
DeepSORT 17% (mostly its own MobileNet embedder), YOLO11n 7%, decode 0.2%.
Batching crops per frame is only 13% faster than one crop per pass on CPU.

**Model versions (Day 8)** are scored on a private gate set (campus ReID test
split + cached crops of the 4 videos + the text set) and recorded in MLflow;
see docs/results/week2.md. The campus split reproduces the published osnet v1
numbers exactly (75.4% mAP, 81.7% Rank-1). IRRA's college Rank-1 cannot be
re-measured until the college test split arrives.

Details, the config table and all findings: [AUDIT.md](AUDIT.md).

## Credits

The research system is the team's final-year project and paper:

> A. Kandel, P. Timalsina, P. Pandey, R. S. Saud, R. Shakya, "End-to-End Person
> Video Retrieval Using Track-Level Robust Scoring and Dynamic Thresholding,"
> *7th Int. Conf. on Computational Vision and Bio Inspired Computing (ICCVBIC)*,
> 2026, pp. 54-60.

- Repo 2 (YOLO + DeepSORT + OSNet pipeline, campus dataset, OSNet fine-tuning,
  evaluation data): Amrit Kandel and team.
- Repo 1 (model 2: CLIP-ReID and IRRA, prompt learning, IRRA weights on
  Hugging Face): `meteroid27` / `prasish17` and team.
- This production version: Roshan S. Saud.

Third-party code and licenses (YOLO11 AGPL-3.0, DeepSORT GPL-3.0, torchreid
MIT, CLIP MIT) are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The campus and college datasets are private and are not in this repository;
demo footage needs the participants' consent (see the roadmap's privacy notes).
