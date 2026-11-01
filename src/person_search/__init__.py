"""Photo + text person search over indexed video.

Package layout (see README.md for the full map):

* ``person_search.ingest``  - video sampling, YOLO11n detection, DeepSORT tracking,
  and the ingest pipeline that turns a video into per-detection embeddings.
* ``person_search.models``  - OSNet (photo) and IRRA (text) encoders, strict
  checkpoint loading.
* ``person_search.core``    - pure logic: similarity, the paper's track scoring and
  dynamic threshold, late fusion.
* ``person_search.search``  - query-time search over ingested observations.
* ``person_search.eval``    - frame-level P/R/F1, ReID mAP/CMC, golden helpers.
* ``person_search.index``   - vector index interface (Qdrant lands on Day 10).
* ``person_search.api``     - FastAPI service (Day 17+).
"""

__version__ = "0.1.0"
