"""Background ingest jobs: a one-thread FIFO queue inside the API process.

Why not FastAPI's ``BackgroundTasks``: it runs each task in the request thread
pool with no limit on how many run at once. Ingest must run one video at a
time: the DeepSORT tracker is a single stateful object, and Qdrant local mode
allows one writer. A ``ThreadPoolExecutor(max_workers=1)`` gives exactly that,
plus a queue length the API can refuse beyond (429). Redis + RQ would add a
second process and a broker for no gain until the service runs on more than
one machine (decision 0012).

Jobs live in memory: a restart forgets them (the index keeps what they wrote).
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


PENDING = (JobStatus.QUEUED, JobStatus.RUNNING)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    id: str
    video_id: str
    video: str
    camera: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class QueueFullError(Exception):
    """Too many ingest jobs are already queued or running."""


# The work a job runs: gets a progress callback (0..1), returns the job result.
JobFn = Callable[[Callable[[float], None]], dict[str, Any]]


class JobRunner:
    def __init__(self, max_pending: int, keep_finished: int = 200) -> None:
        self.max_pending = max_pending
        self.keep_finished = keep_finished
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

    # ------------------------------------------------------------------ reads
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return replace(job) if job else None

    def pending_for(self, video_id: str) -> Job | None:
        """A queued or running job for this video, if any (re-uploads reuse it)."""
        with self._lock:
            for job in self._jobs.values():
                if job.video_id == video_id and job.status in PENDING:
                    return replace(job)
        return None

    def n_pending(self) -> int:
        with self._lock:
            return sum(j.status in PENDING for j in self._jobs.values())

    def counts(self) -> dict[str, int]:
        with self._lock:
            out = {s.value: 0 for s in JobStatus}
            for job in self._jobs.values():
                out[job.status.value] += 1
            return out

    # ----------------------------------------------------------------- writes
    def submit(self, fn: JobFn, *, video_id: str, video: str, camera: str) -> Job:
        job = Job(uuid.uuid4().hex, video_id, video, camera)
        with self._lock:
            pending = sum(j.status in PENDING for j in self._jobs.values())
            if pending >= self.max_pending:
                raise QueueFullError(f"{pending} ingest jobs already pending")
            self._jobs[job.id] = job
            self._prune()
        self._executor.submit(self._run, job.id, fn)
        return replace(job)

    def finished(self, result: dict[str, Any], *, video_id: str, video: str, camera: str) -> Job:
        """Record a job that needed no work (the video is already indexed)."""
        now = _now()
        job = Job(
            uuid.uuid4().hex,
            video_id,
            video,
            camera,
            status=JobStatus.SUCCEEDED,
            progress=1.0,
            created_at=now,
            started_at=now,
            finished_at=now,
            result=result,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        return replace(job)

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    # --------------------------------------------------------------- internals
    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for k, v in changes.items():
                setattr(job, k, v)

    def _run(self, job_id: str, fn: JobFn) -> None:
        self._update(job_id, status=JobStatus.RUNNING, started_at=_now())

        def progress(fraction: float) -> None:
            self._update(job_id, progress=round(min(max(fraction, 0.0), 1.0), 4))

        try:
            result = fn(progress)
        except Exception as exc:  # the job records it; the worker thread lives on
            log.exception("ingest job %s failed", job_id)
            self._update(
                job_id,
                status=JobStatus.FAILED,
                finished_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        self._update(
            job_id, status=JobStatus.SUCCEEDED, finished_at=_now(), progress=1.0, result=result
        )

    def _prune(self) -> None:
        """Forget the oldest finished jobs beyond ``keep_finished`` (caller holds the lock)."""
        done = [j for j in self._jobs.values() if j.status not in PENDING]
        for job in done[: max(0, len(done) - self.keep_finished)]:
            del self._jobs[job.id]
