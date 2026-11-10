# 0013 - One CPU-only image; weights and index are downloaded when it starts

- **Status:** accepted
- **Date:** 2026-11-10

## Context

Step 3 of the fast plan: the same image must run on a laptop, in CI and on the
server (a Hugging Face Space, step 4). Constraints:

- the weights are 1.1 GB (IRRA alone is 1 GB) and the index 64 MB; the YOLO
  and OSNet weights and the index (vectors of real students) are private;
- the IRRA weights are already public, on the IRRA team's Hugging Face repo;
- the server has no GPU, and CUDA wheels would add ~3 GB for nothing;
- `/ingest` writes to the index, so the index on disk changes after start.

## Decision

- **Multi-stage Dockerfile, built from `uv.lock`.** The build stage runs
  `uv sync --frozen --no-dev --group serve --no-editable` (dependencies in their
  own cached layer, then the package); the runtime stage copies only the venv
  and `configs/` onto the same pinned `python:3.10-slim-bookworm` (plus `libgl1`
  and `libglib2.0-0` for the pinned `opencv-python`). torch and torchvision come
  from the PyTorch CPU index already pinned in the lockfile, so the image holds
  no CUDA or NVIDIA libraries. The lock has wheels for linux/amd64 and
  linux/arm64, so the image builds natively on an Apple Silicon Mac too.
- **No weights or data in the image.** `person-search serve` (the image's
  `CMD`) first runs `fetch-assets`: from the settings it works out which
  weights and which index collection the service will read, downloads the
  missing ones from Hugging Face Hub repos listed in `configs/assets.yaml`,
  checks size and SHA-256, and only then moves them into `/data`. A failed or
  interrupted download installs nothing.
- **Two sources.** IRRA from the team's public repo, pinned to a commit; YOLO,
  OSNet, the INT8 IRRA tower and the index from one private repo of yours
  (`PS_ASSETS__REPO`, read with `HF_TOKEN`), filled once by
  `make publish-assets`, which refuses any file whose hash differs from the
  manifest.
- **What is on disk wins.** A weight file with the manifest's size is kept
  (hashing 1 GB on every start costs seconds); an index that already holds the
  collection for the loaded versions is kept whatever its size, because ingest
  grows it. So `/data` on a volume makes restarts fast and keeps ingested videos.
- **Fail fast and say why.** Missing repo, missing token, a 404 or a hash
  mismatch stop the start with exit code 2 and one log line naming the file and
  the setting to fix, before any model loads.
- **Runs as uid 1000** (what Hugging Face Spaces expect), data in `/data` (where
  Spaces mount persistent storage), port 8000, one uvicorn worker, and a
  `HEALTHCHECK` on `/health` with a 10-minute start period for the first download.
- **CI builds the image** (linux/amd64, layer cache in GitHub Actions) and
  checks that torch is the CPU build, that every runtime import works, and
  that a start without assets fails with the clear error. A start with the real
  assets needs a token, so it waits for the deploy and eval-gate steps.

## Results (docs/results/week3_docker.md)

603 MB compressed, 2.7 GB unpacked. Cold start from an empty volume, with the
private files served by a local stand-in for the Hub: 5 files, 1.1 GB, fetched
and verified in 29 s, `/health` after 37 s. Warm restart 9 s with no download,
and a video ingested before the restart is still searchable. Inside the
container: photo 4/4 and text 12/12 identical to Week 2. 2.3 GB RAM idle.

## Consequences

- One manual step before the first deploy: `make publish-assets REPO=...`
  with a write token, then pin the printed commit in `configs/assets.yaml`.
- A new model version or a rebuilt index is a new manifest entry (new hashes),
  reviewed like code.
- If the IRRA team deletes or changes their repo, the pinned commit protects
  against a change but not against deletion: upload `irra_v1.pth` to your
  assets repo and drop the `repo:` line of that entry.
- Without a persistent `/data` (a free Space), every restart downloads 1.1 GB
  again and ingested videos are lost.
