"""HTTP endpoints. Handlers are plain ``def``: FastAPI runs them in its thread pool,
so a slow model call never blocks the event loop."""

from __future__ import annotations

import logging
import secrets
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    Security,
    UploadFile,
)
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import SecretStr

from person_search.api.clips import Style
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
API_KEY_HEADER = "X-API-Key"
API_KEY_REFUSED = f"uploads need a valid {API_KEY_HEADER} header"

router = APIRouter()


def get_service(request: Request) -> SearchService:
    service: SearchService = request.app.state.service
    return service


def get_jobs(request: Request) -> JobRunner:
    jobs: JobRunner = request.app.state.jobs
    return jobs


ServiceDep = Annotated[SearchService, Depends(get_service)]
JobsDep = Annotated[JobRunner, Depends(get_jobs)]

_api_key_header = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="Needed by POST /ingest when the server sets PS_API__API_KEY (the public demo).",
)


def api_key_ok(expected: SecretStr | None, given: str | None) -> bool:
    """No key configured (local runs): open. Otherwise a constant-time comparison."""
    if expected is None:
        return True
    return given is not None and secrets.compare_digest(
        given.encode(), expected.get_secret_value().encode()
    )


def require_api_key(
    service: ServiceDep, key: Annotated[str | None, Security(_api_key_header)]
) -> None:
    """Uploads only. The ``limit_body`` middleware already refused a wrong key before the
    body was read; this dependency documents the header in /docs (the Authorize button)."""
    if not api_key_ok(service.settings.api.api_key, key):
        raise HTTPException(401, API_KEY_REFUSED)


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
    request: Request,
    service: ServiceDep,
    photo: Annotated[
        UploadFile | None,
        File(description="photo of the person (OSNet; the paper's threshold decides matches)"),
    ] = None,
    # The `examples` are what the /docs page pre-fills: without them it fills
    # "string" into every text and list field, which silently changes the query.
    description: Annotated[
        str | None,
        Form(
            max_length=MAX_DESCRIPTION,
            description="what the person looks like (IRRA); leave empty for a photo-only search",
            examples=[""],
        ),
    ] = None,
    top_k: Annotated[
        int, Form(ge=1, le=MAX_TOP_K, description="tracks to return", examples=[5])
    ] = 5,
    alpha: Annotated[
        float | None,
        Form(
            ge=0.0,
            le=1.0,
            description="photo + description only: weight of the photo score (default 0.5)",
            examples=[0.5],
        ),
    ] = None,
    video_ids: Annotated[
        list[str] | None,
        Form(
            description="search only these videos, by video_id from GET /videos "
            "(repeat the field or comma-separate); empty = all",
            examples=[[]],
        ),
    ] = None,
    cameras: Annotated[
        list[str] | None,
        Form(description="search only these camera labels; empty = all", examples=[[]]),
    ] = None,
    clip: Annotated[
        bool,
        Form(
            description="also cut the best result into a video and return its clip_url "
            "(photo: the paper's selected frames, like the final-year pipeline's output)",
            examples=[False],
        ),
    ] = False,
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
        errors = service.filter_errors(vids, cams)
        if errors:
            if "string" in (vids or ()) + (cams or ()):
                errors.append("the /docs page pre-fills list fields with 'string': clear them")
            raise HTTPException(422, "; ".join(errors))
        flt = SearchFilter(video_ids=vids, cameras=cams)
    result = service.search(
        photo=image, description=text, top_k=top_k, alpha=alpha, flt=flt, clip=clip
    )
    _absolute_urls(request, result)
    log.info(
        "search mode=%s top=%s total_ms=%s",
        result.mode,
        result.tracks[0].score if result.tracks else None,
        result.timing_ms.get("total"),
    )
    return result


def _absolute_urls(request: Request, result: SearchResponse) -> None:
    """Clip URLs that can be pasted straight into a browser."""
    base = str(request.base_url).rstrip("/")

    def fix(url: str | None) -> str | None:
        return f"{base}{url}" if url and url.startswith("/") else url

    result.clip_url = fix(result.clip_url)
    for v in result.videos:
        v.clip_url = fix(v.clip_url)
    for t in result.tracks:
        t.clip_url = fix(t.clip_url)


# --------------------------------------------------------------------- clips
_MP4: dict[int | str, dict[str, Any]] = {
    200: {"content": {"video/mp4": {}}, "description": "an H.264 MP4 (plays in a browser)"}
}


@router.get(
    "/videos/{video_id}/tracks/{track_id}/clip",
    response_class=FileResponse,
    tags=["clips"],
    responses={**_MP4, **_error(404, 422)},
)
def track_clip(
    video_id: str, track_id: str, service: ServiceDep, style: Style = "boxed"
) -> FileResponse:
    """Every frame of one track as a video: `boxed` (full frame, person boxed) or
    `crop` (the person only, like the final-year pipeline's output.mp4)."""
    path = service.track_clip(video_id, track_id, style)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"track_{track_id}_{style}.mp4",
        content_disposition_type="inline",
    )


@router.get(
    "/clips/{name}",
    response_class=FileResponse,
    tags=["clips"],
    responses={**_MP4, **_error(404)},
)
def get_clip(name: str, service: ServiceDep) -> FileResponse:
    """A clip made by a search with `clip=true` (its `clip_url`)."""
    return FileResponse(
        service.clip_file(name), media_type="video/mp4", content_disposition_type="inline"
    )


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
    responses=_error(400, 401, 413, 415, 422, 429),
    dependencies=[Depends(require_api_key)],
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
    """Queue a video for indexing and return a job at once (**202**). When the server has
    an API key (the public demo), send it in the ``X-API-Key`` header (**401** otherwise).

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
        service.store_video(path, video_id)  # indexed already: keep the original for clips
        job = jobs.finished(
            service.indexed_summary(video_id), video_id=video_id, video=name, camera=camera
        )
    else:

        def work(progress: Callable[[float], None]) -> dict[str, Any]:
            try:
                result = service.ingest(
                    path, video_id=video_id, name=name, camera=camera, progress=progress
                )
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            result["video_stored"] = service.store_video(path, video_id)
            return result

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
