from __future__ import annotations

import cv2
import numpy as np
import pytest

from person_search.errors import VideoReadError
from person_search.ingest.video import iter_sampled_frames, probe, sample_schedule


def legacy_sampler(n_frames: int, fps: float, target_fps: float) -> list[int]:
    """Verbatim arithmetic of the legacy run_reid loop."""
    kept, next_t = [], 0.0
    for frame_idx in range(1, n_frames + 1):
        if frame_idx / fps < next_t:
            continue
        next_t += 1.0 / target_fps
        kept.append(frame_idx)
    return kept


@pytest.mark.parametrize(("fps", "n"), [(12.0, 575), (30.0, 300), (25.0, 250), (29.97, 600)])
def test_schedule_matches_legacy(fps, n):
    assert sample_schedule(n, fps, 12.0) == legacy_sampler(n, fps, 12.0)


def test_12fps_source_keeps_every_frame():
    assert sample_schedule(575, 12.0, 12.0) == list(range(1, 576))


def test_30fps_source_keeps_about_12_per_second():
    kept = sample_schedule(300, 30.0, 12.0)
    assert abs(len(kept) - 120) <= 1


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 48))
    assert writer.isOpened()
    for i in range(48):
        frame = np.full((48, 64, 3), i * 5, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_iter_sampled_frames_indices(tiny_video):
    info = probe(tiny_video)
    assert (info.width, info.height) == (64, 48)
    frames = list(iter_sampled_frames(tiny_video, target_fps=12.0))
    expected = sample_schedule(info.frame_count, info.fps, 12.0)
    assert [f.frame_index for f in frames] == expected
    assert [f.sample_index for f in frames] == list(range(len(expected)))
    assert frames[0].image.shape == (48, 64, 3)


def test_max_samples(tiny_video):
    assert len(list(iter_sampled_frames(tiny_video, 12.0, max_samples=5))) == 5


def test_missing_video_raises(tmp_path):
    with pytest.raises(VideoReadError):
        list(iter_sampled_frames(tmp_path / "nope.mp4", 12.0))
