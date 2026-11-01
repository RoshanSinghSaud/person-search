"""Ingest: video -> per-detection observations with named embeddings.

Extracted from the frame loop of the legacy ``run_reid`` (both repos), with the
query-first design removed: the loop no longer needs a query, it stores vectors.
That is what turns "reprocess the video for every query" into "index once,
search many" (AUDIT.md F1).

Kept exactly (golden-test parity):

* 12 fps time-based sampling, 1-based raw frame indices,
* detection on a <=640 px wide copy, tracking on the full frame,
* only confirmed tracks; crops clamped to the frame, dropped if < 10 px,
* embeddings on every 3rd sampled frame (``(sample_index + 1) % 3 == 0``),
* every observation recorded, embedded or not (frame selection forward-fills).

Changed:

* all crops of a frame go through each encoder as one batch (AUDIT.md F8),
* crops are not written to disk or JPEG-encoded; rendering re-crops by bbox,
* several encoders can run in one pass (OSNet + IRRA -> one point, two vectors).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from person_search.config import SamplingSettings
from person_search.ingest.video import iter_sampled_frames, probe
from person_search.interfaces import Detector, ImageEncoder, Tracker
from person_search.timing import StageTimer
from person_search.types import BBox, Observation, VideoInfo

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    video: VideoInfo
    observations: list[Observation]
    n_samples: int
    encoder_versions: dict[str, str]
    timings: dict[str, dict[str, float]] = field(default_factory=dict)
    wall_seconds: float = 0.0

    def by_track(self) -> dict[str, list[Observation]]:
        """Observations grouped by track, tracks in order of first appearance."""
        out: dict[str, list[Observation]] = {}
        for obs in self.observations:
            out.setdefault(obs.track_id, []).append(obs)
        return out

    @property
    def track_ids(self) -> list[str]:
        return list(self.by_track())

    @property
    def fps_processed(self) -> float:
        return self.n_samples / self.wall_seconds if self.wall_seconds else 0.0

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Write to a compressed ``.npz`` (arrays + JSON metadata, no pickle)."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        obs = self.observations
        arrays: dict[str, np.ndarray] = {
            "track_id": np.array([o.track_id for o in obs], dtype=str),
            "frame_index": np.array([o.frame_index for o in obs], dtype=np.int64),
            "sample_index": np.array([o.sample_index for o in obs], dtype=np.int64),
            "bbox": np.array([tuple(o.bbox) for o in obs], dtype=np.int64).reshape(-1, 4),
        }
        for name in self.encoder_versions:
            dims = {len(o.embeddings[name]) for o in obs if name in o.embeddings}
            dim = dims.pop() if dims else 0
            mat = np.full((len(obs), dim), np.nan, dtype=np.float32)
            for i, o in enumerate(obs):
                if name in o.embeddings:
                    mat[i] = o.embeddings[name]
            arrays[f"emb__{name}"] = mat
        meta = {
            "video": asdict(self.video),
            "n_samples": self.n_samples,
            "encoder_versions": self.encoder_versions,
            "timings": self.timings,
            "wall_seconds": self.wall_seconds,
        }
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> IngestResult:
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            emb_names = [k.removeprefix("emb__") for k in data.files if k.startswith("emb__")]
            embs = {n: data[f"emb__{n}"] for n in emb_names}
            observations = []
            for i, (tid, fi, si, box) in enumerate(
                zip(
                    data["track_id"],
                    data["frame_index"],
                    data["sample_index"],
                    data["bbox"],
                    strict=True,
                )
            ):
                vectors = {n: m[i] for n, m in embs.items() if m.shape[1] and not np.isnan(m[i, 0])}
                observations.append(
                    Observation(str(tid), int(fi), int(si), BBox(*map(int, box)), vectors)
                )
        return cls(
            video=VideoInfo(**meta["video"]),
            observations=observations,
            n_samples=meta["n_samples"],
            encoder_versions=meta["encoder_versions"],
            timings=meta["timings"],
            wall_seconds=meta["wall_seconds"],
        )


class IngestPipeline:
    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        encoders: Mapping[str, ImageEncoder],
        *,
        sampling: SamplingSettings | None = None,
        min_box_size: int = 10,
        progress_every_s: float = 10.0,
    ) -> None:
        if not encoders:
            raise ValueError("IngestPipeline needs at least one image encoder")
        self.detector = detector
        self.tracker = tracker
        self.encoders = dict(encoders)
        self.sampling = sampling or SamplingSettings()
        self.min_box_size = min_box_size
        self.progress_every_s = progress_every_s

    def is_embedding_frame(self, sample_index: int) -> bool:
        return (sample_index + 1) % self.sampling.embed_every == 0

    def run(
        self,
        video_path: str | Path,
        *,
        max_samples: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> IngestResult:
        info = probe(video_path, self.sampling.fallback_fps)
        self.tracker.reset()
        timer = StageTimer()
        observations: list[Observation] = []
        n_samples = 0
        t_start = last_report = time.perf_counter()

        frames = iter_sampled_frames(
            video_path, self.sampling.target_fps, self.sampling.fallback_fps, max_samples
        )
        while True:
            with timer.stage("decode"):
                sample = next(frames, None)
            if sample is None:
                break
            n_samples += 1
            frame = sample.image
            height, width = frame.shape[:2]

            with timer.stage("detect"):
                detections = self.detector.detect(frame)
            with timer.stage("track"):
                tracks = self.tracker.update(detections, frame)

            frame_obs: list[Observation] = []
            crops: list[np.ndarray] = []
            for track in tracks:
                bbox = BBox(*(int(v) for v in track.ltrb)).clip(width, height)
                if bbox.width < self.min_box_size or bbox.height < self.min_box_size:
                    continue
                frame_obs.append(
                    Observation(track.track_id, sample.frame_index, sample.sample_index, bbox)
                )
                crops.append(bbox.crop(frame))

            if crops and self.is_embedding_frame(sample.sample_index):
                for name, encoder in self.encoders.items():
                    with timer.stage(f"embed:{name}"):
                        vectors = encoder.encode_images(crops)
                    for obs, vec in zip(frame_obs, vectors, strict=True):
                        obs.embeddings[name] = vec
            observations.extend(frame_obs)

            now = time.perf_counter()
            if on_progress is not None:
                on_progress(n_samples, info.frame_count)
            if now - last_report >= self.progress_every_s:
                log.info(
                    "ingest %s: %d samples, %.1f samples/s, %d observations",
                    Path(video_path).name,
                    n_samples,
                    n_samples / (now - t_start),
                    len(observations),
                )
                last_report = now

        wall = time.perf_counter() - t_start
        result = IngestResult(
            video=info,
            observations=observations,
            n_samples=n_samples,
            encoder_versions={n: e.version for n, e in self.encoders.items()},
            timings=timer.as_dict(),
            wall_seconds=wall,
        )
        log.info(
            "ingested %s: %d samples, %d tracks, %d observations in %.1fs (%.2f samples/s)",
            Path(video_path).name,
            n_samples,
            len(result.track_ids),
            len(observations),
            wall,
            result.fps_processed,
        )
        return result
