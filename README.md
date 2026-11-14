# Person Search

Find a person in indexed video **by photo** (OSNet, from the ICCVBIC 2026 paper),
**by description** (IRRA, model 2) or **by both**. This repo turns the team's
final-year system into a production service: index once, search many.

[![CI](https://github.com/RoshanSinghSaud/person-search/actions/workflows/ci.yml/badge.svg)](https://github.com/RoshanSinghSaud/person-search/actions/workflows/ci.yml)

> **Status: Weeks 1-2 done; Week 3 steps 1-3 (IRRA INT8, web API, Docker) done; step 4 (deploy on every merge to a Hugging Face Space) built and parked: since July 2026 Hugging Face runs Docker Spaces for PRO accounts only (decision 0015); step 5 (model gate in CI) done: it blocks vanilla CLIP and the Market-1501 OSNet and passes INT8 IRRA on real pull requests (#7, #9, #10).** Week 1 extracted both legacy repos into modules
> with parity and golden tests ([AUDIT.md](AUDIT.md)). Week 2 added evaluation
> for photo and text queries, a Qdrant index holding both vectors, indexed
> photo, text and hybrid search (exactly equal to a full scan), the first
> end-to-end text numbers and per-stage timings
> ([docs/results/week2.md](docs/results/week2.md)). Week 3 put IRRA's image
> tower in ONNX INT8, the search behind a FastAPI service
> ([docs/results/week3_api.md](docs/results/week3_api.md)) and the service in
> a CPU-only Docker image that downloads its weights and index at start
> ([docs/results/week3_docker.md](docs/results/week3_docker.md)). Deployment
> follows the plan.

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
  api/               app.py (FastAPI, lifespan model loading) routes.py service.py jobs.py schemas.py clips.py
configs/             report_run.yaml (default = repo 2's final code) paper.yaml models.yaml (model versions)
scripts/             build_index eval_search gate_set eval_models profile_ingest week2_report fetch_weights
tests/unit/          unit tests, no weights needed (CI)
tests/parity/        new modules vs legacy code (make parity)
tests/golden/        pipeline.run vs repo 2 on the 4 test cases (make golden)
tools/legacy/        Day 2 harness, golden builder, drawn-box extractor (import legacy by path)
tools/analysis/      scoring sweeps, track contact sheets, label sensitivity
configs/assets.yaml  where the service downloads missing weights and indexes from (SHA-256 pinned)
Dockerfile           CPU-only image: uv.lock -> venv, assets fetched at start into /data
docs/decisions/      one note per decision (0007-0010: Week 2; 0011-0014: Week 3)
docs/results/        week2.md, week3_onnx.md, week3_api.md, week3_docker.md, from the run outputs
```

## Quick start

```bash
# 1. environment (Python 3.10, CPU PyTorch)
uv sync --group dev --group eval --group serve   # or: make setup
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

### Run the API

```bash
make api-index          # copy the Week 2 index (4 test videos) to data/index
make api-videos         # keep the 4 test videos, so results come with a video clip
make serve              # http://localhost:8000/docs; models load first (~15 s)
```

```bash
# photo, description, or both (late fusion; alpha = photo weight)
curl -F photo=@query.jpg http://localhost:8000/search
curl -F description="a man in a white shirt and dark trousers" -F top_k=5 http://localhost:8000/search
curl -F photo=@query.jpg -F description="white shirt, cap" -F alpha=0.5 http://localhost:8000/search
curl -F description="..." -F video_ids=fce3e914169cac08 http://localhost:8000/search  # one video

# the video of the person, like the final-year pipeline's output: open clip_url in a browser
curl -F photo=@query.jpg -F video_ids=bed83b0fa58e9925 -F clip=true http://localhost:8000/search
curl -o track7.mp4 "http://localhost:8000/videos/bed83b0fa58e9925/tracks/7/clip?style=crop"

# index a new video: 202 + a job at once, then poll the job
curl -F video=@clip.mp4 -F camera="gate 2" http://localhost:8000/ingest
curl http://localhost:8000/jobs/<job_id>        # queued -> running (progress) -> succeeded
curl http://localhost:8000/health               # model versions, index size, jobs
curl http://localhost:8000/videos               # indexed videos and their video_id
```

| Endpoint | What it does |
|---|---|
| `POST /search` | `photo` and/or `description` (multipart). Always returns ranked `tracks` with when they are on screen (`start_s`, `end_s`) and a `clip_url`; a photo query also returns the paper's decision per video: `matched`, accepted tracks, frame segments. `clip=true` adds a ready video of the best result (`clip_url`) |
| `GET /videos/{video_id}/tracks/{track_id}/clip` | an H.264 MP4 of every frame of one track: `style=boxed` (full frame, person boxed) or `style=crop` (the person only) |
| `GET /clips/{name}` | a clip made by `clip=true` |
| `POST /ingest` | `video` (+ `camera`). **202** with a job and `Location: /jobs/{id}`; one video is ingested at a time, 429 beyond 4 pending. The original is kept for clips (`PS_API__KEEP_VIDEOS=false` to not). An already indexed video returns `skipped: true` (and is kept) |
| `GET /jobs/{id}` | `queued` / `running` (with `progress`) / `succeeded` (points, tracks, seconds) / `failed` (error) |
| `GET /health` | model versions in use (`osnet-v1`, `irra-v1`), index collection and size, job counts |

The service refuses to start if the index folder was built for other model
versions than the ones it loads (an empty folder gets a new index). Design and limits: decision 0012. `make smoke-api`
checks a running server against the Week 2 numbers.

### Run it in Docker

The image holds the code and CPU-only PyTorch; on start it downloads whatever
weights and index are missing into `/data` (`configs/assets.yaml`, SHA-256
checked), then serves on port 8000. Keep `/data` on a volume and later starts
take ~10 s.

```bash
make docker-build                    # CPU-only image for this machine (~3 min the first time)

# a) no Hugging Face needed: mount ./models and the index you already have
make api-index api-videos            # once: data/index and data/videos from Week 2
make docker-run-local                # http://localhost:8000/docs

# b) the way the server runs it: everything downloaded at start
make publish-assets REPO=<hf-user>/person-search-assets   # once, needs a write token
export PS_ASSETS__REPO=<hf-user>/person-search-assets HF_TOKEN=hf_...   # a read token is enough
make docker-run                      # first start ~1.1 GB download, then cached in the ps-data volume
```

`make publish-assets` uploads YOLO, OSNet, the INT8 IRRA tower and the Week 2
index to a private repo, after checking each file against the manifest, and
prints the commit to pin as `revision` in `configs/assets.yaml`. The IRRA
weights come from the IRRA team's public repo. On an Apple Silicon Mac the
image builds natively (arm64); `make docker-build DOCKER_PLATFORM=linux/amd64`
builds exactly the server's image (slower, emulated). Design: decision 0013.

Every setting can be overridden with `PS_<SECTION>__<FIELD>`, e.g.
`PS_SCORING__BASE_THRESHOLD=0.7`. See `.env.example`.

### Deploy to a Hugging Face Space

> **Parked.** Hugging Face now runs Docker Spaces only for PRO accounts, so the
> `deploy` job is off until the repository variable `DEPLOY_SPACE` is `true`
> (decision 0015). Everything below works once the account can host the Space.

With `DEPLOY_SPACE=true`, every push to `main` deploys itself: after the tests and the image build pass,
the CI job `deploy` uploads the build files to the Space
`<hf-user>/person-search`, waits until it runs that commit, and checks that
`/health` reports both models and the index, that a text search answers and
that an upload without the API key is refused. The API is then at
`https://<hf-user>-person-search.hf.space/docs`. One-time setup:

1. `make publish-assets REPO=<hf-user>/person-search-assets` (write token; see above).
2. In the GitHub repo, *Settings -> Secrets and variables -> Actions*, add the
   secrets `HF_TOKEN` (Hugging Face **write** token), `HF_SPACE_TOKEN` (a **read**
   token: the Space uses it to download the private assets) and `PS_API_KEY`
   (`openssl rand -hex 24`; `POST /ingest` on the Space needs it in `X-API-Key`).

The Space is created on the first deploy. To redeploy by hand: *Actions -> CI ->
Run workflow* on `main` (tick *sync secrets* after rotating a token or the key),
or `HF_TOKEN=... HF_SPACE_TOKEN=... PS_API_KEY=... make deploy-space`. The free
tier has no persistent disk: a restart downloads the weights again and forgets
videos ingested since. Design: decision 0015.

### Model gate: a worse model cannot reach production

Every pull request runs the **model gate** (`.github/workflows/model-gate.yml`,
decision 0016). If it changes a `production:` pin in `configs/models.yaml`, CI
downloads both versions' weights and the private gate set, scores both on the
same runner, and fails when the candidate breaks decision 0010's rule: campus
mAP down by more than 1 point, OSNet photo F1 down by more than 1 point, or any
text query that production ranks first lost. The verdict is posted on the pull
request as a table. Other pull requests pass it in seconds. Versions are
immutable: new weights get a new version name.

One-time setup: upload the gate set and the gate-only weights to Hugging Face
(private), and give CI a **read** token (`HF_READ_TOKEN`; the deploy's
`HF_SPACE_TOKEN` also works):

```bash
uv run hf upload <hf-user>/person-search-gate ../legacy/gate_set . --repo-type dataset --private
uv run hf upload <hf-user>/person-search-assets models/osnet_market_v0.pth weights/osnet_market_v0.pth
```

To make "fails" mean "cannot merge": *Settings -> Rules -> Rulesets*, target
`main`, require the status check `model gate`. Results: docs/results/week4_gate.md.

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

**Model versions (Day 8)**, scored on a private gate set (campus ReID test split,
cached crops of the 4 videos, the text set) and registered in MLflow
(`person-search-osnet`, `person-search-irra`, alias `production`):

| Version | Campus mAP | Photo F1, 4 videos | Text Hit@1 / Hit@5 |
|---|---:|---:|---:|
| osnet v0 (Market-1501) | 29.2% | 85.25% | - |
| **osnet v1** (campus fine-tune) | **75.4%** | **84.70%** | - |
| irra v0 (vanilla CLIP), image tower | 12.8% | - | 6/12 / 12/12 |
| **irra v1**, image tower | **78.0%** | - | **8/12 / 11/12** |

Campus mAP for OSNet uses torchreid's protocol and reproduces the published
numbers exactly; the IRRA rows use cosine. Photo F1 on 4 videos cannot rank
models (v0 scores higher; AUDIT.md F26), so the gate relies on campus mAP.
IRRA's high campus score may be train/test overlap (F27). IRRA's college Rank-1
cannot be re-measured until the college test split arrives.

Details, the config table and all findings: [AUDIT.md](AUDIT.md).

### Week 3, step 1: IRRA image tower in ONNX INT8

Full tables: [docs/results/week3_onnx.md](docs/results/week3_onnx.md); decision 0011.

| IRRA image tower | Size | ms per crop (batch 32, 2 threads) | Campus mAP | Text Hit@1 / Hit@5 |
|---|---:|---:|---:|---:|
| PyTorch FP32 (irra-v1) | 345 MB | 125 | 78.0% | 8/12 / 11/12 |
| ONNX FP32 | 345 MB | 126 | (cosine > 0.9999 vs PyTorch) | same |
| **ONNX INT8 (irra-v2)** | **89 MB** | **40** | 77.4% | 8/12 / 11/12 |

INT8 is 3.2x faster, passes the gate (mAP -0.56 points, no text hit lost) and
makes ingest 1.32x faster end to end. OSNet is now the slowest stage.

```bash
make onnx                          # export + quantize (needs models/irra_v1.pth)
make benchmark                     # speed of the three backends
make gate-compare CANDIDATE=irra-v2
export PS_IRRA__IMAGE_ONNX=models/irra_v2_image_int8.onnx PS_IRRA__VERSION=irra-v2
```

### Week 3, step 2: the web API

Full tables: [docs/results/week3_api.md](docs/results/week3_api.md); decision 0012.
With the real models over HTTP, the API gives the same answers as Week 2:

| Check (each case searched in its own video) | Result |
|---|---|
| Photo: accepted tracks equal Week 2 | 4/4 |
| Text: top track and score equal Week 2 | 12/12 |
| Photo + text (alpha 0.5): top track is a photo-accepted track | 12/12 |
| p50 latency, photo / text / both (2 vCPU) | 0.26 s / 0.27 s / 0.93 s |
| `POST /ingest` of a 10 s clip | 202 in 63 ms; searchable 59 s later, target found |

Searching all videos at once applies the paper's threshold per video, so a
photo can also "match" in other videos (AUDIT.md F29); pass `video_ids` or
`cameras` when you know where to look.

### Week 3, step 3: Docker

Full tables: [docs/results/week3_docker.md](docs/results/week3_docker.md); decision 0013.

| | |
|---|---|
| Image | 603 MB compressed, torch `2.5.0+cpu`, no CUDA; amd64 and arm64 |
| Cold start (empty `/data`, 1.1 GB to download and verify) | 37 s to `/health` |
| Warm restart (volume kept) | 9 s, no download; ingested videos kept |
| Answers inside the container | photo 4/4, text 12/12, identical to Week 2 |
| Memory | 2.3 GB idle |

### Week 4, step 5: model gate in CI

Full table: [docs/results/week4_gate.md](docs/results/week4_gate.md); decision 0016.
Real pull requests, scored in CI on GitHub's 4-vCPU runner:

| Pull request | Pins production to | Verdict | Deciding numbers | Job |
|---|---|---|---|---:|
| [#7](https://github.com/RoshanSinghSaud/person-search/pull/7) | `irra-v0` (vanilla CLIP) | **blocked** | campus mAP 78.01% -> 12.82%; 4 Hit@1 queries lost | 29.6 min |
| [#9](https://github.com/RoshanSinghSaud/person-search/pull/9) | `osnet-v0` (Market-1501) | **blocked** | campus mAP 76.41% -> 29.46% | 5.7 min |
| [#10](https://github.com/RoshanSinghSaud/person-search/pull/10) | `irra-v2` (INT8 tower) | **passed** | campus mAP -0.56 pts; no Hit@1 lost | 16.5 min |

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
The campus and college datasets are private and are not in this repository.


