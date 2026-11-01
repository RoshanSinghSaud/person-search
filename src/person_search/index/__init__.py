"""Vector index (Day 10).

Planned contents:

* ``base.py``   - ``VectorIndex`` protocol: ``upsert(points)``, ``search(vector_name,
  query, limit, filters)``, ``fetch_track(video, track_id)``, ``count()``.
* ``qdrant.py`` - Qdrant implementation; one point per sampled detection with two
  named vectors (``osnet``, ``irra``) and payload (video, camera, track, frame,
  sample, bbox); point IDs from ``uuid5(video_hash, frame, track)`` so
  re-ingesting a video adds 0 points.
* ``memory.py`` - in-memory fake for tests.

``person_search.ingest.IngestResult.observations`` already carries everything a
point needs.
"""
