"""Cached crops of the 4 test cases: what the quality gate embeds instead of videos.

The gate compares candidate *models*, so it skips decoding, detection and
tracking: those stages are frozen in a cached case. A cached case holds

* every observation of the default ingest (track, frame, sample, bbox), so the
  paper's frame selection (which forward-fills over un-embedded frames) runs
  exactly as in the pipeline;
* a PNG crop for every embedded observation (lossless, so re-embedding gives
  the ingest's vectors bit for bit);
* the video's metadata and the ingest's encoder versions.

Crops are re-embedded one sampled frame per batch, as the ingest does, which
keeps the results identical to ``pipeline.run``. Because the crops were decoded
once on linux-x86_64, the gate gives the same numbers on any machine (decoding
is where macOS and Linux diverge, AUDIT.md F18).

File format: one ``.npz`` per case, arrays only (no pickle); crops are PNG bytes
concatenated in ``crop_blob`` with ``crop_offsets``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from person_search.ingest.pipeline import IngestResult
from person_search.ingest.video import iter_sampled_frames
from person_search.interfaces import ImageEncoder
from person_search.types import BBox, Observation, VideoInfo


@dataclass
class CachedCase:
    case: str
    video: VideoInfo
    observations: list[Observation]
    embed_rows: np.ndarray  # indices into observations that have a crop
    crops: list[np.ndarray]  # BGR, aligned with embed_rows
    meta: dict[str, Any]

    def embed(self, encoders: Mapping[str, ImageEncoder]) -> list[Observation]:
        """Fresh observations carrying the given encoders' vectors."""
        obs = [
            Observation(o.track_id, o.frame_index, o.sample_index, o.bbox)
            for o in self.observations
        ]
        groups: dict[int, list[int]] = {}
        for k, row in enumerate(self.embed_rows.tolist()):
            groups.setdefault(obs[row].sample_index, []).append(k)
        for ks in groups.values():  # one sampled frame per batch, like the ingest
            crops = [self.crops[k] for k in ks]
            for name, enc in encoders.items():
                vecs = enc.encode_images(crops)
                for k, v in zip(ks, vecs, strict=True):
                    obs[int(self.embed_rows[k])].embeddings[name] = v
        return obs


def export_case(
    case: str,
    video_path: str | Path,
    ingest: IngestResult,
    out_path: str | Path,
    *,
    target_fps: float = 12.0,
    fallback_fps: float = 30.0,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    obs = ingest.observations
    rows = [i for i, o in enumerate(obs) if o.has_embeddings]
    wanted: dict[int, list[int]] = {}
    for i in rows:
        wanted.setdefault(obs[i].sample_index, []).append(i)
    png: dict[int, bytes] = {}
    for sample in iter_sampled_frames(video_path, target_fps, fallback_fps):
        for i in wanted.get(sample.sample_index, []):
            ok, buf = cv2.imencode(
                ".png", obs[i].bbox.crop(sample.image), [cv2.IMWRITE_PNG_COMPRESSION, 6]
            )
            if not ok:
                raise RuntimeError(f"PNG encoding failed for observation {i}")
            png[i] = buf.tobytes()
    missing = [i for i in rows if i not in png]
    if missing:
        raise RuntimeError(f"{len(missing)} embedded observations have no frame in {video_path}")
    blobs = [png[i] for i in rows]
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    meta = {
        "case": case,
        "video": asdict(ingest.video),
        "encoder_versions": ingest.encoder_versions,
        **(extra_meta or {}),
    }
    out = Path(out_path).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        meta=np.array(json.dumps(meta)),
        track_id=np.array([o.track_id for o in obs], dtype=str),
        frame_index=np.array([o.frame_index for o in obs], dtype=np.int64),
        sample_index=np.array([o.sample_index for o in obs], dtype=np.int64),
        bbox=np.array([tuple(o.bbox) for o in obs], dtype=np.int64).reshape(-1, 4),
        embed_rows=np.array(rows, dtype=np.int64),
        crop_offsets=offsets,
        crop_blob=np.frombuffer(b"".join(blobs), dtype=np.uint8),
    )
    return out


def load_case(path: str | Path) -> CachedCase:
    with np.load(Path(path), allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        obs = [
            Observation(str(t), int(f), int(s), BBox(*map(int, b)))
            for t, f, s, b in zip(
                d["track_id"], d["frame_index"], d["sample_index"], d["bbox"], strict=True
            )
        ]
        offsets, blob = d["crop_offsets"], d["crop_blob"]
        crops = [
            cv2.imdecode(blob[offsets[k] : offsets[k + 1]], cv2.IMREAD_COLOR)
            for k in range(len(offsets) - 1)
        ]
        rows = d["embed_rows"].copy()
    return CachedCase(meta["case"], VideoInfo(**meta["video"]), obs, rows, crops, meta)
