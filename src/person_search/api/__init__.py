"""HTTP service (Days 17-18).

Planned contents: ``app.py`` (FastAPI with lifespan model loading via
``person_search.factory.build_components``), ``routes/search.py``
(``POST /search``: photo, description or both), ``routes/ingest.py``
(``POST /ingest`` -> 202 + job id, ``GET /jobs/{id}``), ``worker.py`` (RQ worker
calling ``IngestPipeline``), ``/health`` and ``/metrics``.
"""
