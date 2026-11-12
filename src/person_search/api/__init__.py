"""HTTP service (Step 2 of the fast plan; decision 0012).

* :mod:`~person_search.api.app`     - ``create_app()``: lifespan model loading, error mapping
* :mod:`~person_search.api.routes`  - ``POST /search``, ``POST /ingest``, ``GET /jobs/{id}``,
  ``GET /health``, ``GET /videos``
* :mod:`~person_search.api.service` - ``SearchService``: encoders + index + ``IndexedSearch``
* :mod:`~person_search.api.jobs`    - one-thread ingest queue, jobs kept in memory
* :mod:`~person_search.api.schemas` - response bodies (the contract the tests pin)

Imports stay light here: ``create_app`` pulls in FastAPI, and the real models
load only when the app starts.
"""
