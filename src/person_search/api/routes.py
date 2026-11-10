"""HTTP endpoints. Handlers are plain ``def``: FastAPI runs them in its thread pool,
so a slow model call never blocks the event loop."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import RedirectResponse

from person_search.api.jobs import Job, JobRunner, QueueFullError
from person_search.api.schemas import (
    ErrorOut,
    HealthResponse,
    JobOut,
    SearchResponse,
    VideoOut,
)
from person_search.api.service import SearchService
from person_search.index.base import SearchFilter
from person_search.index.points import video_fingerprint

log = logging.getLogger(__name__)

MB = 1 << 20
MAX_TOP_K = 50
MAX_DESCRIPTION = 500
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

router = APIRouter()


def get_service(request: Request) -> SearchService:
    service: SearchService = request.app.state.service
    return service


def get_jobs(request: Request) -> JobRunner:
    jobs: JobRunner = request.app.state.jobs
    return jobs


ServiceDep = Annotated[SearchService, Depends(get_service)]
JobsDep = Annotated[JobRunner, Depends(get_jobs)]


def _error(*codes: int) -> dict[int | str, dict[str, object]]:
    return {c: {"model": ErrorOut} for c in codes}


def _split(values: list[str] | None) -> tuple[str, ...] | None:
    """Repeated form fields and/or comma-separated values -> tuple (None when empty)."""
    out = tuple(v.strip() for raw in values or [] for v in raw.split(",") if v.strip())
    return out or None


def _job_out(job: Job) -> JobOut:
    return JobOut(
        job_id=job.id,
        status=job.status,
        video_id=job.video_id,
        video=job.video,
        camera=job.camera,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        result=job.result,
        error=job.error,
    )


# ---------------------------------------------------------------------- meta
@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(service: ServiceDep, jobs: JobsDep) -> HealthResponse:
    """Liveness and readiness: the model versions in use and what the index holds."""
    return service.health(jobs.counts())


@router.get("/videos", response_model=list[VideoOut], tags=["meta"])
def videos(service: ServiceDep) -> list[VideoOut]:
    """Indexed videos; their ``video_id`` values can restrict a search."""
    return service.videos()


# -------------------------------------------------------------------- search
def _read_photo(photo: UploadFile, max_bytes: int) -> np.ndarray | None:
    data = photo.file.read(max_bytes + 1)
    if not data:
        return None  # an empty file field counts as "no photo"
    if len(data) > max_bytes:
        raise HTTPException(413, f"photo is larger than {max_bytes // MB} MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "photo is not a readable image (JPEG, PNG, BMP or WebP)")
    return image


@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["search"],
    responses=_error(400, 413, 422),
)
def search(
    service: ServiceDep,
    photo: Annotated[
        UploadFile | None,
        File(description="photo of the person (OSNet; the paper's threshold decides matches)"),
    ] = None,
    description: Annotated[
        str | None,
        Form(max_length=MAX_DESCRIPTION, description="what the person looks like (IRRA)"),
    ] = None,
    top_k: Annotated[int, Form(ge=1, le=MAX_TOP_K, description="tracks to return")] = 5,
    alpha: Annotated[
        float | None,
        Form(ge=0.0, le=1.0, description="photo + description only: weight of the photo score"),
    ] = None,
    video_ids: Annotated[
        list[str] | None,
        Form(description="search only these videos (repeat the field or comma-separate)"),
    ] = None,
    cameras: Annotated[list[str] | None, Form(description="search only these cameras")] = None,
) -> SearchResponse:
    """Find a person by **photo**, **description** or **both**.

    * photo only: the paper's rule decides which tracks match and returns their frames;
    * description only: tracks ranked by IRRA score, for a person to review;
    * both: late fusion of the two normalised scores (``alpha`` = photo weight).
    """
    image = None
    if photo is not None:
        image = _read_photo(photo, int(service.settings.api.max_photo_mb * MB))
    text = (description or "").strip() or None
    if image is None and text is None:
        raise HTTPException(422, "give a photo, a description or both")
    flt = None
    vids, cams = _split(video_ids), _split(cameras)
    if vids or cams:
        flt = SearchFilter(video_ids=vids, cameras=cams)
    result = service.search(photo=image, description=text, top_k=top_k, alpha=alpha, flt=flt)
    log.info(
        "search mode=%s top=%s total_ms=%s",
        result.mode,
        result.tracks[0].score if result.tracks else None,
        result.timing_ms.get("total"),
    )
    return result


# -------------------------------------------------------------------- ingest
def _save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as f:
        while chunk := upload.file.read(MB):
            size += len(chunk)
            if size > max_bytes:
                break
            f.write(chunk)
    if size > max_bytes:
        dest.unlink(missing_ok=True)
        raise HTTPException(413, f"video is larger than {max_bytes // MB} MB")
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "video file is empty")


def _check_video(path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    try:
        ok = cap.isOpened() and cap.read()[0]
    finally:
        cap.release()
    if not ok:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "video cannot be decoded")


@router.post(
    "/ingest",
    status_code=202,
    response_model=JobOut,
    tags=["ingest"],
    responses=_error(400, 413, 415, 422, 429),
)
def ingest(
    response: Response,
    service: ServiceDep,
    jobs: JobsDep,
    video: Annotated[UploadFile, File(description="video to index (mp4, avi, mov, mkv, webm)")],
    camera: Annotated[
        str,
        Form(
            min_length=1,
            max_length=64,
            pattern=r"^[\w .:-]+$",
            description="camera label, stored with every detection and usable as a filter",
        ),
    ] = "unknown",
) -> JobOut:
    """Queue a video for indexing and return a job at once (**202**).

    Poll ``GET /jobs/{job_id}`` until ``status`` is ``succeeded`` or ``failed``.
    Uploading a video that is already indexed, or already queued, does no new work.
    """
    name = Path(video.filename or "upload").name
    suffix = Path(name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(415, f"unsupported video type {suffix or '(none)'}: {VIDEO_SUFFIXES}")
    api = service.settings.api
    path = api.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    _save_upload(video, path, int(api.max_video_mb * MB))
    _check_video(path)
    video_id = video_fingerprint(path)

    pending = jobs.pending_for(video_id)
    if pending is not None:
        path.unlink(missing_ok=True)
        job = pending
    elif service.has_video(video_id):
        path.unlink(missing_ok=True)
        job = jobs.finished(
            service.indexed_summary(video_id), video_id=video_id, video=name, camera=camera
        )
    else:

        def work(progress: Callable[[float], None]) -> dict[str, Any]:
            try:
                return service.ingest(
                    path, video_id=video_id, name=name, camera=camera, progress=progress
                )
            finally:
                path.unlink(missing_ok=True)

        try:
            job = jobs.submit(work, video_id=video_id, video=name, camera=camera)
        except QueueFullError:
            path.unlink(missing_ok=True)
            raise
    response.headers["Location"] = f"/jobs/{job.id}"
    return _job_out(job)


@router.get("/jobs/{job_id}", response_model=JobOut, tags=["ingest"], responses=_error(404))
def get_job(job_id: str, jobs: JobsDep) -> JobOut:
    """Status of an ingest job (jobs are kept in memory; a restart forgets them)."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return _job_out(job)
