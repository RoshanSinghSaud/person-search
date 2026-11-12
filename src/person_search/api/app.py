"""FastAPI application factory.

Run it with::

    uvicorn person_search.api.app:create_app --factory --port 8000     # or: make serve

Models load once, in the lifespan, before the first request is accepted. One
uvicorn worker only: Qdrant local mode locks its folder to one process, and the
models would be loaded once per worker.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from person_search import __version__
from person_search.api.jobs import JobRunner, QueueFullError
from person_search.api.routes import MB, router
from person_search.api.service import SearchService, build_service
from person_search.config import Settings, load_settings
from person_search.errors import QueryError, VideoReadError
from person_search.logging_utils import configure_logging

log = logging.getLogger(__name__)

DESCRIPTION = """\
Find a person in indexed CCTV video **by photo** (OSNet, ICCVBIC 2026 paper),
**by description** (IRRA) or **by both**. Videos are indexed once with
`POST /ingest`; every search then reads the index instead of reprocessing video.
"""


def create_app(service: SearchService | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Without ``service`` the lifespan loads the real models from
    ``settings`` (default: ``load_settings()``); tests pass a service built from fakes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not logging.getLogger().handlers:
            configure_logging("INFO")
        svc = service if service is not None else build_service(settings or load_settings())
        runner = JobRunner(svc.settings.api.max_pending_jobs)
        app.state.service, app.state.jobs = svc, runner
        try:
            yield
        finally:
            runner.shutdown(wait=False)
            svc.close()

    app = FastAPI(
        title="Person Search",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.middleware("http")
    async def limit_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Refuse an oversized upload from its Content-Length, before reading the body."""
        api = request.app.state.service.settings.api
        limits = {"/ingest": api.max_video_mb, "/search": api.max_photo_mb}
        limit_mb = limits.get(request.url.path)
        length = request.headers.get("content-length", "")
        if limit_mb is not None and length.isdigit() and int(length) > (limit_mb + 1) * MB:
            return JSONResponse({"detail": f"upload larger than {limit_mb:g} MB"}, 413)
        return await call_next(request)

    @app.exception_handler(QueryError)
    @app.exception_handler(VideoReadError)
    async def bad_input(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, 400)

    @app.exception_handler(QueueFullError)
    async def queue_full(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            {"detail": f"{exc}; try again later"}, 429, headers={"Retry-After": "60"}
        )

    return app
