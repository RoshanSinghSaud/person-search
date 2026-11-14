# 0015 - The service runs on a public Hugging Face Space; every push to main redeploys it

- **Status:** accepted, then parked the same day (see "Parked")
- **Date:** 2026-11-12

## Context

Step 4 of the fast plan: put the API online at a link anyone can open, and
deploy every merge to `main` without a manual step. The image (decision 0013)
already downloads its weights and index at start and runs as uid 1000 on
port 8000. Constraints:

- the free Space tier is 2 vCPU / 16 GB RAM, no GPU, no persistent disk, and it
  sleeps after 48 h without visitors;
- the YOLO and OSNet weights and the index stay in the private assets repo;
- decisions 0012 and 0014 require at least an API key before a public server
  accepts uploads, because uploaded footage is stored and served back as clips.

## Decision

- **Hugging Face builds the image from the Dockerfile.** CI uploads only the
  git-tracked files the build reads (`Dockerfile`, `.dockerignore`,
  `pyproject.toml`, `uv.lock`, `src/`, `configs/`) plus `README.md` and
  `THIRD_PARTY_NOTICES.md`, as one commit, deleting files earlier deploys left.
  The lockfile and the pinned base image make that build the same as CI's.
  The Space's settings (`sdk: docker`, `app_port: 8000`) are prepended to its
  copy of the README, so the GitHub README has no Space header.
- **Deploy is a CI job, after the tests and the image build pass**, on pushes to
  `main` and on a manual run (`workflow_dispatch`, with an option to rewrite the
  Space's secrets). It runs `scripts/deploy_space.py` (`make deploy-space` runs it
  by hand). Pushes to main no longer cancel a running workflow, so a deploy is never
  cut off halfway; deploys queue one at a time (`concurrency: deploy-space`).
- **Healthy means answering, not built.** The job waits until the Space runs the
  commit it uploaded, then checks over HTTPS that `/health` reports both model
  versions and a non-empty index, that a text search returns tracks, and that an
  upload without the key is refused (401). A failed build or start prints the
  Space's build and container logs into the CI log and fails the job.
- **Secrets stay least-privilege.** GitHub holds a write token (deploys); the
  Space gets its own read token as `HF_TOKEN` (downloads the private assets) and
  the upload key as `PS_API__API_KEY`. Settings are written only when they change
  (a secret only when missing), because each write restarts the Space.
- **Uploads need an API key; searches do not.** `PS_API__API_KEY` (>= 16
  characters) makes `POST /ingest` require the `X-API-Key` header. The check runs
  in the middleware, so a request without the key is refused before its body is
  read; the route declares the header too, so `/docs` has an Authorize button.
  Unset (local runs), uploads stay open as before.
- **The Dockerfile drops `VOLUME ["/data"]`.** `/data` is created in the image,
  owned by uid 1000; a volume declared by the image could be mounted empty and
  root-owned on a platform that honours it. `docker run -v ps-data:/data` (the
  Makefile) still mounts a named volume there, initialised from the image.
- **No footage on the public server by default.** Only the index (vectors and
  boxes) is downloaded; the 4 test videos are not, so search results there come
  without clips until the footage's owners agree to a public demo.

## Parked (2026-11-12)

The first deploy reached Hugging Face and was refused with **402 Payment Required**:
since July 2026 Static Spaces are free, but Gradio and Docker Spaces on the free
cpu-basic hardware need a PRO subscription (USD 9/month), which needs a payment
card. The pipeline is kept and switched off: the `deploy` job runs only when the
repository variable `DEPLOY_SPACE` is `true` (Settings -> Secrets and variables ->
Actions -> Variables), so `main` stays green. What did run for real: CI on the
merge, the private assets repo (`make publish-assets`, pinned in
`configs/assets.yaml`), the token check and the Space creation request.

To go live later: subscribe the Hugging Face account to PRO and set
`DEPLOY_SPACE=true`; or point the same image at another host (Azure Container Apps
on Azure for Students needs no card) and replace only the upload step.

## Results (docs/results/week4_deploy.md)

To fill after the first deploy: build time on the Hub, cold start (download +
model load) to the first healthy `/health`, warm restart, p50 search latency on
the Space's CPU against Week 3's numbers.

## Consequences

- A merge to main is live a few minutes after CI passes, or the job fails and
  says why; the previous version keeps serving only if Hugging Face keeps it up.
- No persistent storage: every restart downloads 1.1 GB (from the Hub, so fast)
  and forgets videos ingested since. Persistent storage is a paid option.
- A sleeping Space takes about a minute to wake on the first visit.
- The Space's files are public: they are the service's source, which also meets
  AGPL-3.0 section 13 for YOLO11 (THIRD_PARTY_NOTICES.md). Weights and index
  stay private.
- Next: run `make smoke-api` against the Space, rate limits per client, and the
  eval gate (Day 24) before a model change reaches this job.
