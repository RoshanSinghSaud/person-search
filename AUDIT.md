# AUDIT - legacy pipelines (Day 1, plus the Day 2 reproduction)

Audited commits:

| Repo | Commit | Role |
|---|---|---|
| repo 1 `meteroid27/person-reid-clip` | `5d8768a` (2026-08-23) | base for ingest + IRRA |
| repo 2 `Amritkandel49/Person-Based-Video-Segmentation-Using-Reidentification-And-Tracking` | `5153437` (2026-05-02, single commit) | OSNet, scoring, eval data |

Confirmed by the team (2026-11-02): repo 1 holds the final IRRA code (nothing
was updated after it), and the code may be republished. Label provenance stays
unknown; decision 0006 fixes how the labels are used.

Weights audited: team Drive folder (YOLO, OSNet campus, OSNet Market-1501) and
Hugging Face `prasish17/clip-text-query` / `clip-image-query` (public, not
gated). SHA-256s are in `models/README.md` via `scripts/fetch_weights.py`.

## 1. Pipeline stages

| Stage | Repo 1 (`reid_main.py`, `clip_inference.py`) | Repo 2 (`pole/reid_main.py`) | New home |
|---|---|---|---|
| Model load | `run_reid` L437-468: HF download (silent local fallback L442-445), two `clip.load("ViT-B/16")` (L35, L38), YOLO from `models_path/` | `FeatureExtractor._load_model` L103-160 (ResNet50 fallback); YOLO path hardcoded to `/media/amrit/...` (L19) | `factory.py`, `models/*`, `ingest/detector.py` |
| Query | `_build_query_feature` L93-141 (image / text / sum) | `extractor.extract(query_img)` L383 (+ `print(query_img)` L379) | `pipeline.py`, `search/*` |
| Sampling | time-based to `OUTPUT_FPS=12`, `FRAME_SKIP=1` (L505-525) | same (L415-437) | `ingest/video.py` |
| Detection | YOLO on <=640 px copy, class 0, conf **0.30** (L22, L530-560) | same, conf **0.4** (L37, L440-470) | `ingest/detector.py` |
| Tracking | `DeepSort(max_age=90, n_init=2, embedder="mobilenet")` L463 | `DeepSort(max_age=30, n_init=3, embedder="mobilenet")` L370; `max_cosine_distance` not set -> library default **0.2** | `ingest/tracker.py` |
| Embedding | every 3rd sampled frame (L500, L584); IRRA visual for text/both, CLIP-ReID visual for image | every 3rd sampled frame (L410, L494); OSNet 256x128 | `ingest/pipeline.py`, `models/*` |
| Crops | JPEG-encoded in memory (L601) | `cv2.imwrite` for every confirmed track on every sampled frame (L505-516) | not stored; re-cropped by bbox when rendering |
| Track score | top-50% mean (`compute_track_score` L143) | same (L188) | `core/scoring.track_score` |
| Threshold | image: `max(0.20, best-0.10)`, frame floor 0.18; text: `best-0.02`, frame floor 0.138 (L664-682) | `max(0.65, best-0.10)`, frame floor `0.9*tau` (L572-576) | `core/scoring.decide` |
| Frame selection | forward-fill, dedupe by higher track (L704-749) | same (L589-631) | `core/scoring.select_frames` |
| Output | crop clip, annotated clip (H.264 attempts), selected crops | crop clip from disk crops, annotated clip | `render.py` |
| Metrics | - | `utilities/calculate_system_metrics.py`, `Finetune_OSNET/Python Codes/evaluate_*.py` (torchreid engine) | `eval/frame_metrics.py`, `eval/reid.py` |

## 2. The roadmap's findings, checked against the code

| # | Finding | Verdict | Evidence |
|---|---|---|---|
| F1 | Every search reprocesses the whole video | **Confirmed** | Both `run_reid` build the query feature before opening the video (repo 1 L471 -> L481; repo 2 L383 -> L391) and compute `sim` inside the frame loop. `run_reid_multi` (repo 1 L788) loads models once but still decodes, detects, tracks and embeds per query. |
| F2 | Hybrid mode mixes two embedding spaces | **Confirmed** | `_build_query_feature` L120 takes the image query from `image_clip_model` (CLIP-ReID), L128 the text query from IRRA, sums them at L138; gallery crops use IRRA's visual tower (L591-592). Kept as `core/fusion.legacy_embedding_sum` for the Day 12 comparison. |
| F3 | Nobody has measured speed or text search on video | **Confirmed** | Repo 2 has no timing. Repo 1 `run_reid_multi` prints a progress "proc-fps" only. Model 2's only numbers are on cropped images. First numbers are in section 4. |
| F4 | The configs disagree | **Confirmed, plus more** | See the table in section 3. The report also contradicts itself: tau_base "e.g. 0.5" (eq. 4-33) vs absolute threshold 0.7 (section 5.4.5). `reid_main_prev.py` / `reid_main_tmp.py` hold yet another config (det 0.5, threshold 0.7, 10 fps). |
| F5 | Keep the detection setting that reproduces the report's run | **No single setting does (Day 2)** | Each committed `output_annotated.mp4` is reproduced frame for frame, but by different settings: case 1 by det 0.5 / max_cosine_distance 0.3; cases 2-3 by det 0.4 / 0.2 with tau_base 0.7; case 4 by det 0.4 / 0.2 with tau_base 0.65 (at 0.7 case 4 is "no match"). Default kept = repo 2's final code (det 0.4 / 0.2 / tau 0.65): exact legacy parity, 4/4 hits, best F1 of the 4/4 configurations. Section 5 and decision 0002. |
| F6 | The repo's labels belong to the report's run, not the paper's | **Confirmed, with a caveat** | The 4 `evaluation.json` files give exactly the report's 89.05% F1 (1073/74/190; unit test `test_reproduces_report_f1`). But they were not labelled on the committed videos: every case has frames labelled TP/FP with no box drawn (25, 61, 50, 19) and boxed frames left unlabelled (3, 3, 12, 3). Use TP + FN as ground truth for any new run; do not expect TP + FP to equal a selection (decision 0006). |
| F7 | Failures are silent | **Confirmed, plus more** | Repo 2: no torchreid or any load error -> ImageNet ResNet50 (L104-106, L152-160); `extract` returns `None` on any error (L184-186). Repo 1: no matching keys -> base CLIP with a printed warning (`clip_inference.py` L242-244, L321-323); failed crops retried at 224x224 then `None` (L377-394); HF failure -> local path that may not exist (L442-445). Repo 1 `run_reid` defaults to `model_type="osnet"` but `FeatureExtractor` was deleted from that file (L416, L454): the default call raises `NameError`. |
| F8 | IRRA dominates the cost; no batching; repo 2 writes every crop | **Confirmed** | One crop per forward pass in both repos; repo 2 `cv2.imwrite` per crop (L514). Measured cost split in section 4. |
| F9 | The 4 test cases give both query types ground truth | **Confirmed** | `evaluation.json` TP + FN = frames with the target, in sampled-frame (0-based) space. Golden track IDs per test case are in `tests/golden/data/*.json`, ready for Day 9's text targets. |

## 3. Configuration table

| Setting | Repo 1 | Repo 2 (code) | Report text | Paper | **Default here** |
|---|---|---|---|---|---|
| YOLO conf | 0.30 | 0.4 | 0.5 | - | **0.4** |
| DeepSORT max_age / n_init | 90 / 2 | 30 / 3 | 30 / 3 | - | **30 / 3** |
| max_cosine_distance | 0.2 (default) | 0.2 (default) | 0.3 | 0.3 | **0.2** |
| max_iou_distance | 0.7 (default) | 0.7 (default) | 0.7 | - | **0.7** |
| Sampling / embed interval | 12 fps / 3 | 12 fps / 3 | every 3rd frame | every 3rd | **12 fps / 3** |
| tau_base | 0.20 (CLIP image) | 0.65 | 0.7 (5.4.5), 0.5 (4-33) | 0.5 | **0.65** |
| delta | 0.10 (image), 0.02 (text) | 0.1 | 0.1 | 0.1 | **0.1** |
| frame floor | 0.18 / 0.138 absolute | 0.9 * tau_eff | - | - | **0.9 * tau_eff** |

## 4. Environment and new findings

| # | Finding | Impact | Handling |
|---|---|---|---|
| F10 | The YOLO weights are **YOLO11n** (C3k2/C2PSA blocks) saved with **ultralytics 8.4.22**, but repo 2 pins `ultralytics==8.2.23`, which predates YOLO11 (8.3.0). Repo 2's requirements cannot load its own detector; both READMEs say YOLOv8. The Drive file and repo 1's file are byte-identical. | Reproducibility | `ultralytics>=8.4.22,<8.5` |
| F11 | `update_tracks` returns confirmed tracks that were **not** matched this frame (`time_since_update > 0`, up to `max_age` = 2.5 s), with Kalman-predicted boxes. Both repos record them as observations and can select them. | Likely source of FP frames when a track coasts onto a neighbour | Kept for parity; switch `tracker.include_unmatched_tracks` (decision 0004) |
| F12 | `deep-sort-realtime==1.3.2` imports `pkg_resources`, removed from setuptools >= 81. A fresh env fails at tracker construction. | Install breaks | `setuptools<81` pinned |
| F13 | Detection boxes are rescaled with numpy-scalar division, so DeepSORT receives float64 coordinates. Batched (array) division gives float32 and different sub-pixel boxes. | Golden parity | Detector keeps scalar division |
| F14 | `clip.load` on CUDA keeps fp16 weights; on CPU it casts to fp32. Repo 1's IRRA numbers therefore depend on the device. | Comparability of IRRA scores | `irra.precision` is explicit (fp32 default) |
| F15 | IRRA inserts 8 prompt tokens and drops the last 8 token slots; descriptions longer than ~67 BPE tokens lose EOT and are read at slot 76. | Long witness descriptions | Same behaviour, now logged |
| F16 | The IRRA checkpoint contains every CLIP tensor except `logit_scale`, so the base OpenAI download contributes nothing. | 335 MB download, start-up time | Model built from the checkpoint (decision 0003) |
| F18 | **The pipeline's output is platform-dependent.** The same code, weights and pinned versions select different frames on macOS arm64 than on Linux x86_64: on an M-series MacBook, case 1 lost 3 frames, case 2 lost 12, case 3 lost 86 and gained 3, case 4 gained 51 (a second track crossed tau_base). `make fingerprint` shows the drift starts at **video decoding**: the macOS arm64 OpenCV wheel decodes the H.264 test videos to different pixels than the Linux x86_64 wheel (all 4 videos, from frame 1), so YOLO boxes differ in all 4 cases. The hard thresholds (DeepSORT gating, tau_base, the 0.9 * tau frame floor) then turn small differences into different selections. Repo 2's code scores 85.34% frame F1 on the Mac (84.70% on Linux); case 4 drops from 66.1% to 53.5%, with a different best track (0.728 vs 0.683). With a Mac baseline recorded, `pipeline.run` matches repo 2 exactly on the Mac as well (4/4). | Golden data are per platform; frame F1 is only comparable on one reference platform | `tests/golden/data/<platform>/`, `make golden-baseline`, `make fingerprint`; reference = linux-x86_64 (CI, Docker image). Decision 0005 |
| F17 | Repo 2 has no license file; repo 1 is MIT; YOLO is AGPL-3.0, DeepSORT GPL-3.0. | Publishing | `THIRD_PARTY_NOTICES.md`; get the team's OK |

### First speed numbers (this audit's container: 2 vCPU, no GPU, shared with other jobs - indicative only)

Ingest of test case 4 (353 frames at 1080p, 13 tracks, 1500 detections, 500 embedded crops)
with the new pipeline, both encoders, crops batched per frame:

| Stage | Seconds | Share | Per embedded crop |
|---|---:|---:|---:|
| IRRA image (ViT-B/16, 256x128) | 98.1 | 47.3% | 196 ms |
| OSNet x1_0 | 47.2 | 22.8% | 94 ms |
| DeepSORT (incl. its MobileNet appearance embedder) | 33.1 | 16.0% | - |
| YOLO11n (640 px) | 27.9 | 13.4% | - |
| decode | 1.1 | 0.5% | - |
| **total** | **207.5** | | 1.70 sampled frames/s |

IRRA is the largest single cost (F8 confirmed) but only about 2x OSNet here,
not the 10x a FLOP count suggests: OSNet's depthwise convolutions run poorly on
this CPU. Re-measure on your own machine on Day 13 before optimising.

Repo 2 untouched, per video (same container, contended): 193.7 s, 413.9 s,
447.8 s, 121.3 s for cases 1-4. One text query with repo 1 untouched
(`tools/legacy/time_repo1_text.py`, `model_type="clip"`, output videos skipped)
on case 1: **407.5 s**, of which 15.7 s is loading two CLIP models and both
checkpoints. Repo 1's default `model_type="osnet"` path cannot run at all (F7).
All of these ran next to other jobs on 2 vCPUs; re-measure on one quiet
machine on Day 13 before quoting any of them.

## 5. Day 2 reproduction

`tools/legacy/reproduce_repo2.py` runs repo 2's `run_reid` untouched (only the
hardcoded `YOLO_PATH` / `OSNET_MODEL_PATH` are pointed at local files) and
captures the selected frames. Log check: `OSNet loaded successfully` on all 4
cases (no ResNet50 fallback).

Frame-level results, all against the labels' ground truth (TP + FN of each
`evaluation.json`, sampled-frame space):

| Case | Report (labels' own counts) | Committed `output_annotated.mp4` | Repo 2 re-run (= golden) | Re-run vs committed video |
|---|---:|---:|---:|---|
| 1 | 94.12% | 90.78% | 96.46% | all 402 boxed frames + 81 more, same 2 tracks |
| 2 | 92.48% | 84.15% | 86.77% | all 354 boxed frames + 42 more, same 2 tracks |
| 3 | 74.49% | 75.00% | 65.36% | all 195 boxed frames + 104 more (2nd track, 0.657) |
| 4 | 86.13% | 66.12% | 66.12% | **identical** (62 frames, tracks 7 + 42) |
| **System F1** | **89.05%** | **83.92%** | **84.70%** | P 85.48%, R 83.93% for the re-run |

What this means:

1. **The re-run contains every boxed frame.** With det 0.4 and
   max_cosine_distance 0.2 the re-run selects every frame the committed videos
   box, on the same track IDs; case 4 is frame-for-frame identical. The extra
   frames in cases 1-3 come from a different threshold or tracking setting
   (point 3), not from a different model.
2. **The labelled run is not in the repo.** Neither the committed videos nor a
   re-run of the committed code give 89.05% against the committed labels, and
   the labels disagree with the committed videos frame by frame (F6). The
   89.05% (and the paper's 94.23%) stay research numbers; the regression
   baseline is the re-run, pinned by the golden test.
3. **The committed videos come from at least three configurations.** Two
   ingests (repo 2's det 0.4 / max_cosine_distance 0.2, and the report's det
   0.5 / 0.3) were scored at tau_base 0.5, 0.65 and 0.7
   (`tools/analysis/sweep_scoring.py`, `tools/legacy/extract_drawn_boxes.py`):

   | Ingest | tau_base | System F1 | P | R | Hit rate | Committed videos reproduced exactly |
   |---|---:|---:|---:|---:|---:|---|
   | det 0.4, cos 0.2 (repo 2 code) | 0.5 | 80.36% | 73.30% | 88.92% | 4/4 | - |
   | det 0.4, cos 0.2 (repo 2 code) | **0.65** | **84.70%** | 85.48% | 83.93% | **4/4** | case 4 |
   | det 0.4, cos 0.2 (repo 2 code) | 0.7 | 85.32% | 94.86% | 77.51% | 3/4 | cases 2, 3 |
   | det 0.5, cos 0.3 (report text) | 0.5 | 80.03% | 85.10% | 75.53% | 4/4 | case 1 |
   | det 0.5, cos 0.3 (report text) | 0.65 | 84.23% | 95.95% | 75.06% | 4/4 | case 1 |
   | det 0.5, cos 0.3 (report text) | 0.7 | 81.62% | 98.65% | 69.60% | 3/4 | cases 1, 2 |

4. **The 4/4 hit rate depends on tau_base = 0.65.** Case 4's best track scores
   0.683 (0.681 with the report's tracking), so at the report's stated 0.7 it
   returns "no match". The pattern (cases 2-3 rendered at 0.7, case 4 at 0.65,
   committed code at 0.65) is consistent with the threshold having been
   lowered after looking at case 4. That is an inference from the artifacts,
   not something the repo states: ask Amrit before an interview does. If true,
   tau_base was tuned on the test set, and the 4 cases cannot also serve as an
   unbiased test of it.

Per-case details (selected frames, track IDs, timings, logs, crops) are written
by the harness to `../legacy/runs/repo2_default/`; golden files are in
`tests/golden/data/`.

## 6. Parity of the extracted modules

| Test | Inputs | Result |
|---|---|---|
| `tests/parity/test_osnet_parity.py` | 20 crops (4 query photos + 16 boxes from the test videos), campus and Market-1501 weights | **pass** - min cosine 0.99999994 (campus) / 1.00000000 (Market-1501); max abs diff 4.8e-7; identical rankings |
| `tests/parity/test_irra_parity.py` | same 20 crops + 5 descriptions (one long enough to hit F15) | **pass** - image min cosine 0.99999988, text 1.00000000; max abs diff 6.6e-7; text-to-image similarities within 2.1e-7; identical top-10 |
| `tests/golden/test_golden.py` | 4 test cases, `pipeline.run(video, photo)` vs repo 2 untouched, same platform (F18) | **pass on all 4** (`pytest -m golden`) - identical selected frames and track IDs; missing 0, extra 0 |

## 7. Week 2 findings (Days 8-14)

| # | Finding | Impact | Handling |
|---|---|---|---|
| F19 | **Photo search's second accepted track is the wrong person in 2 of 4 cases.** Case 3's track 162 (score 0.657, 53 frames) is a man in a green-grey jacket; case 4's track 42 is a different person. Both pass the dynamic threshold (best - 0.1). The golden frames, and therefore the 84.70% F1, include these false positives. | Precision of photo search; text targets cannot be taken from golden tracks | Targets labelled from crops (decision 0007). A later scoring change can be judged by these two cases |
| F20 | **DeepSORT switches identities in crowded scenes.** In case 3 the target is track 4 (samples 2-160), then track 52, then track 96; in case 2 track 49 is the target only for samples 482-668. | Track-level metrics; a "track" is not a person | Targets are sample ranges per track; text hits are judged on the detections behind the score (decision 0007) |
| F21 | **The campus split reproduces Amrit's published numbers exactly.** osnet v1 under torchreid's protocol (Euclidean, raw features): 75.37% mAP, 81.69/88.03/92.25/97.18% Rank-1/5/10/20 (published 75.4 / 81.7 / 88.0 / 92.3 / 97.2). With cosine on normalised features, as the service scores, 76.41% mAP / 82.39% Rank-1. | Confirms the split and the OSNet port | Gate uses cosine; Euclidean reported for comparison |
| F22 | **Repo 1's mixed-space hybrid is not measurably worse here.** CLIP-ReID and IRRA image embeddings of the same photo have cosine 0.22 (0.10 for different people): both towers start from the same CLIP weights, so a weak identity signal survives. On the 12 queries the sum and late fusion both reach Hit@1 12/12, as does photo search alone. | The Day 12 "fix" cannot be shown on this test set | Late fusion kept for correctness (decision 0009); a harder test set is future work |
| F23 | **Qdrant local mode evaluates payload filters in Python**: a video-filtered search over 16k points took ~250 ms against ~12 ms unfiltered. | Query latency | Local mode post-filters an unfiltered ranking (same result, tested) and fetches tracks by ID (decision 0008) |
| F24 | **The locked `onnxruntime==1.24.3` has no Python 3.10 wheel**, so `uv sync --group eval` failed on every machine. | Install | `onnxruntime>=1.18,<1.24` (1.23.2) |
| F25 | **Ultralytics overrides the PyTorch thread count.** Its `select_device` (called on the first prediction) runs `torch.set_num_threads(min(8, cpu_count - 1))`, so `PS_TORCH_THREADS` is ignored from the first YOLO call on: 1 thread on a 2-vCPU machine, 7 on an 8-core Mac. Repo 2 has the same behaviour. | Speed, and the last bits of every vector: OSNet vectors of the same crops differ by up to 2.6e-7 between 1 and 2 threads, enough to flip a frame at a threshold. Every Week 2 timing ran on 1 PyTorch thread | Recorded in docs/results/week2.md. The gate embeds with `PS_TORCH_THREADS` set to the ingest's count. **Fixed in Week 3:** the detector re-applies `PS_TORCH_THREADS` after each YOLO call when it is set; unset keeps the legacy behaviour (and the golden data) |
| F26 | **On the 4 test videos, photo frame F1 does not separate the models.** The Market-1501 OSNet (osnet v0) scores 85.25% system F1 against 84.70% for the campus fine-tune (v1): better on cases 2 and 3, much worse on case 4 (F1 0.41, 162 false-positive frames). The thresholds (tau_base 0.65, delta 0.1) were tuned for v1, and 4 cases are too few. Campus mAP separates them clearly (29.2% vs 75.4%). | A photo-F1-only gate would let a worse model through | The gate blocks on campus mAP; photo F1 is a regression check against production, not a model-quality score (decision 0010) |
| F27 | **IRRA's image tower scores higher on the campus ReID split than the fine-tuned OSNet**: 78.0% mAP / 84.2% Rank-1 (cosine) against 76.4% / 82.4%. IRRA was fine-tuned on the team's college text-image data, which may show the same students as the campus test split. | If there is no overlap, one model could serve photo and text queries; if there is, the number is leakage | Not acted on. Ask the IRRA teammate whether the college data and the campus ReID test IDs overlap before drawing any conclusion |
| F28 | **OSNet gets slower with more PyTorch threads on this CPU**: 104 ms per crop on 1 thread (Week 2), 147-150 ms on 2 (Week 3, same crops). Its depthwise convolutions parallelise poorly in PyTorch's CPU kernels. After IRRA moved to INT8, OSNet is the largest ingest stage (57%). | Ingest speed | Next speed-up: OSNet in ONNX Runtime, or a separate thread count per encoder (decision 0011) |

## 8. Step 2 findings (web API)

| # | Finding | Impact | Handling |
|---|---|---|---|
| F29 | **A photo searched across all videos "matches" in videos where the person may not be.** The paper's dynamic threshold is defined for one query against one video, and indexed search applies it per video (decision 0008). Over the 4 test videos, the case 1 photo also accepts track 13 of case 4's video (0.687), case 2's photo track 162 of case 3's video (0.654; the same track F19 calls the wrong person for case 3), and case 3's photo four tracks in two other videos (up to 0.726). Unchecked: lookalikes, or the same students on another camera. | Precision of unfiltered photo search, which the API allows | The API reports `matched` per video, so a client sees which video each match comes from; `video_ids` / `cameras` restrict the search. Check the crops before Day 22; a cross-video threshold is future work (decision 0012) |
