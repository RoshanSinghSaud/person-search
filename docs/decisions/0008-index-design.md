# 0008 - Index design: one point per observation, exact two-stage search

- **Status:** accepted
- **Date:** 2026-11-09

## Context

Both legacy repos reprocess the whole video for every query (AUDIT.md F1). Week
2 indexes each video once and answers queries from the index. Requirements:

- photo search over the index must select **exactly** the frames
  `pipeline.run` selects (the golden frames), or the paper's numbers no longer
  apply;
- re-ingesting a video must not duplicate it;
- an index must never be searched with encoders it was not built with;
- it must run inside one container with no extra service (the fast plan).

## Decision

**Points.** One Qdrant point per observation (tracked person per sampled
frame), payload `video_id, video, camera, track_id, frame_index, sample_index,
t_seconds, bbox, embedded, seq`. Points on embedding frames carry two named
vectors, `osnet` and `irra`; the others carry none. The vector-less points are
needed because the paper's frame selection forward-fills scores over every
frame of an accepted track.

**IDs.** `uuid5(namespace, "video_hash:frame:track")`, with the video hash the
first 16 hex characters of the file's SHA-256. Re-ingesting writes the same IDs,
so it overwrites instead of adding (test: 16,483 points before and after).

**Distance.** DOT on L2-normalised vectors. COSINE would re-normalise on insert
and shift scores by ~1e-7; DOT stores the vectors unchanged, so the rerank
stage reproduces the in-memory scores bit for bit.

**Versions.** The collection name carries the encoder versions
(`detections__irra-v1__osnet-v1`). Opening an index for other versions fails
with a clear error; a model change means a new collection.

**Two-stage search** (`search/indexed.py`): retrieve detections by one vector,
then fetch every point of the candidate tracks and run the unchanged scoring.
Each cut-off comes with a bound that proves the answer equals a full scan,
and the cut-off grows until the bound holds:

| Query | Retrieval | Why nothing is missed |
|---|---|---|
| photo | every detection with OSNet score >= tau_base (minus 1e-4) | an accepted track's top-50% mean is >= tau_base, so it has a detection above it |
| text | top N by IRRA | a track with no detection in the top N scores below the N-th hit; stop when the k-th candidate track scores above that |
| hybrid | top N by each vector; min/max of each modality from the top hit of q and of -q | same argument with the fused score of the two N-th hits |

**Local mode.** Qdrant runs in-process with its data in a folder. It evaluates
payload filters in Python point by point, so `QdrantIndex` keeps a
`(video, track) -> point IDs` map in memory and fetches tracks by ID. Local mode
locks its folder, so this process is the only writer and the map cannot go
stale.

## Consequences

- Indexed photo search equals the golden frames on all 4 cases, and indexed
  text and hybrid search equal the full scan on all 12 queries (Day 11-12 runs).
- The paper's dynamic threshold is defined for one query against one video, so
  a photo query over several videos applies it per video.
- Z-score fusion needs every score in scope, so the indexed hybrid search
  supports min-max only (z-score would be a scan).
- Local mode is fine for a demo-sized index: with 16k points, retrieval takes
  ~30 ms and fetching candidate tracks 50-230 ms (about 50 us per point), so an
  indexed photo query is 0.25-0.5 s including OSNet on the photo, against 2-11
  minutes of reprocessing (docs/results/week2.md). Fetch is the next cost to cut.
- A Qdrant server with payload indexes is the step up: the same class takes a
  `QdrantClient(url=...)`; post-filtering and the track map are then off,
  because filters run server-side and other writers may exist.
