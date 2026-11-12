"""Response bodies. These are the API contract; tests/unit/test_api.py pins them."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from person_search.api.jobs import JobStatus


class IndexInfo(BaseModel):
    collection: str | None = Field(description="Qdrant collection; carries the model versions")
    points: int = Field(description="tracked detections stored")
    videos: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = Field(description="person-search package version")
    models: dict[str, str] = Field(
        description="model version per vector, e.g. {'osnet': 'osnet-v1', 'irra': 'irra-v1'}"
    )
    index: IndexInfo
    jobs: dict[str, int] = Field(description="ingest jobs by status")


class VideoOut(BaseModel):
    video_id: str = Field(description="content hash of the video file (16 hex chars)")
    video: str | None = Field(description="file name given at ingest")
    camera: str | None
    points: int
    stored: bool = Field(
        False, description="the original video is on the server, so result clips can be cut"
    )


class TrackOut(BaseModel):
    video_id: str
    video: str | None
    camera: str | None
    track_id: str = Field(description="tracker ID, unique within its video only")
    score: float = Field(description="mean of the track's top 50% detection scores")
    n_frames: int = Field(description="frames of the track that were scored")
    accepted: bool | None = Field(
        None, description="photo queries only: passed the paper's dynamic threshold"
    )
    start_s: float | None = Field(None, description="first time the track is on screen (s)")
    end_s: float | None = Field(None, description="last time the track is on screen (s)")
    start_frame: int | None = None
    end_frame: int | None = None
    clip_url: str | None = Field(
        None,
        description="video of every frame of this track, person boxed; add ?style=crop for "
        "the crops (needs the original video on the server)",
    )


class SegmentOut(BaseModel):
    """Consecutive sampled frames where an accepted track was selected (inclusive)."""

    start_frame: int = Field(description="1-based raw frame index")
    end_frame: int
    start_sample: int = Field(description="0-based index in the 12 fps sampled stream")
    end_sample: int
    n_samples: int
    start_s: float | None = None
    end_s: float | None = None


class PhotoVideoResult(BaseModel):
    """The paper's decision for one video: accept tracks, select frames."""

    video_id: str
    video: str | None
    camera: str | None
    matched: bool
    best_score: float | None
    effective_threshold: float
    accepted_tracks: list[str]
    selected_frames: int
    segments: list[SegmentOut]
    clip_url: str | None = Field(
        None, description="with clip=true: the selected frames as a video, like the paper's output"
    )


class SearchResponse(BaseModel):
    mode: Literal["photo", "text", "hybrid"]
    model_versions: dict[str, str]
    tracks: list[TrackOut] = Field(description="best tracks first, across all searched videos")
    matched: bool | None = Field(
        None, description="photo only: some video has a track above the threshold"
    )
    videos: list[PhotoVideoResult] = Field(
        default_factory=list, description="photo only: per-video decision and frame segments"
    )
    alpha: float | None = Field(None, description="hybrid only: weight of the photo score")
    clip_url: str | None = Field(
        None, description="with clip=true: a video of the best result (open it in a browser)"
    )
    notes: list[str] = Field(default_factory=list)
    timing_ms: dict[str, float]


class JobOut(BaseModel):
    job_id: str
    status: JobStatus
    video_id: str
    video: str
    camera: str
    progress: float = Field(description="0..1, sampled frames processed")
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: dict[str, Any] | None = None
    error: str | None = None


class ErrorOut(BaseModel):
    detail: str
