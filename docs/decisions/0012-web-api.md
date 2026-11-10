# 0012 - The web API: one process, models loaded once, ingest in a one-thread queue

- **Status:** accepted
- **Date:** 2026-11-10

## Context

Step 2 of the fast plan turns the index-once, search-many code from Week 2 into
a service other people can call: search by photo, description or both; send a
new video and get an answer at once while it is indexed in the background; and
report which model versions are running. Constraints:

- the search code must not change: `IndexedSearch` is proven equal to a full
  scan (decision 0008), and the gate numbers were measured through it;
- ingest is slow (about 1 sampled frame per second per CPU) and stateful: one
  DeepSORT tracker, and Qdrant local mode, which allows one writer per folder;
- the first deployment is one small CPU machine (a Hugging Face Space, step 4);
- CI has no weights.

## Decision

- **FastAPI, one uvicorn worker.** `create_app()` loads YOLO, DeepSORT, OSNet
  and IRRA once, in the lifespan, and opens (or creates) the Qdrant folder for
  the loaded versions. Handlers are plain `def`, so model calls run in the thread
  pool and never block the event loop.
- **Endpoints.** `POST /search` (multipart: `photo` and/or `description`,
  `top_k`, `alpha`, `video_ids`, `cameras`), `POST /ingest` (multipart `video`,
  `camera`) -> **202** with a job and a `Location: /jobs/{id}` header,
  `GET /jobs/{id}`, `GET /health` (model versions, index size, job counts),
  `GET /videos`. The response always has a ranked `tracks` list; photo queries
  also return the paper's per-video decision (`matched`, accepted tracks, frame
  segments).
- **Ingest runs in a one-thread queue inside the process**
  (`ThreadPoolExecutor(max_workers=1)`), not FastAPI's `BackgroundTasks`, which
  would run every upload at once in the request pool, and not Redis + RQ, which
  adds a broker and a second process for no gain on one machine. Beyond 4
  pending jobs `/ingest` answers 429. Jobs live in memory; a restart forgets
  them but not what they wrote.
- **Idempotent uploads.** A video is identified by its content hash (the
  index's `video_id`). Uploading an indexed video returns a finished job with
  `skipped: true`; uploading a queued one returns the queued job.
- **One lock around the index.** Searches and the ingest's final write take it;
  decoding and embedding do not. Qdrant local mode is not thread-safe and
  `QdrantIndex` keeps a track map that `upsert` changes.
- **The service refuses to start on an index built for other models.** The
  collection name carries the encoder versions. If the index folder holds
  collections but none for the loaded versions (e.g. `PS_IRRA__VERSION=irra-v2`
  on the Week 2 index), startup fails with "rebuild the index" instead of
  quietly serving an empty second collection; an empty folder gets a new index.
- **Limits.** Photos 10 MB, videos 500 MB (`PS_API__MAX_*`), checked from
  `Content-Length` before the body is read and again on the bytes received.
- **Contract tests in CI** (`tests/unit/test_api.py`): the real app, routes,
  queue, `SearchService`, `IndexedSearch` and both index backends, with fake
  encoders/detector/tracker. They pin the OpenAPI paths and form fields, the
  response fields, every error code, and that the API returns exactly what
  `IndexedSearch` returns.
- `serve` dependencies trimmed to FastAPI, uvicorn and python-multipart (Redis,
  RQ and the Prometheus client were placeholders from the 30-day plan).

## Results (docs/results/week3_api.md)

With the real models on the Week 2 index, over HTTP: photo search accepts the
same tracks as Week 2 on 4/4 cases, text search returns the same top track and
score for 12/12 descriptions, hybrid's top track is a photo-accepted track for
12/12. p50 latency: photo 0.26 s, text 0.27 s, hybrid 0.93 s. `/ingest`
answered in 63 ms; a 10 s clip was searchable 59 s later.

## Consequences

- One machine, one worker. Scaling out needs a Qdrant server and a real queue
  (Redis + RQ or similar); `VectorIndex` and `SearchService` already allow both.
- Ingest competes with searches for the CPU while it runs (a text query took
  0.35 s during an ingest on 2 vCPU).
- Searching all videos at once applies the paper's threshold to each video, so
  a photo can "match" a lookalike in a video where the person never appears
  (AUDIT.md F29). Clients should pass `video_ids` or `cameras` when they know
  where to look.
- Not yet: authentication, rate limits per client, `/metrics`, persistent job
  history. The public demo (Day 22) needs at least an API key before it accepts
  uploads.
