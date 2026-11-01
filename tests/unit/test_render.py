from __future__ import annotations

import cv2
import numpy as np

from person_search.render import write_annotated_video, write_crop_clip
from person_search.search.results import FrameHit
from person_search.types import BBox


def _video(path, n=12, size=(96, 64)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), 10 * i, dtype=np.uint8))
    w.release()
    return path


def _count(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def test_annotated_video_has_every_sample_and_green_boxes(tmp_path):
    video = _video(tmp_path / "in.avi")
    hits = [FrameHit(f, f - 1, "7", BBox(10, 20, 40, 60)) for f in (3, 4, 5)]
    out = tmp_path / "annotated.mp4"
    assert write_annotated_video(video, hits, out) == 12
    assert _count(out) == 12
    cap = cv2.VideoCapture(str(out))
    frames = [cap.read()[1] for _ in range(12)]
    cap.release()
    green = [
        int(((f[:, :, 1] > 180) & (f[:, :, 0] < 100) & (f[:, :, 2] < 100)).sum()) for f in frames
    ]
    assert all(g > 0 for g in green[2:5])
    assert green[0] == 0 and green[-1] == 0


def test_crop_clip_has_one_frame_per_hit(tmp_path):
    video = _video(tmp_path / "in.avi")
    hits = [FrameHit(f, f - 1, "7", BBox(10, 20, 41, 61)) for f in (2, 3, 9)]
    out = tmp_path / "crops.mp4"
    assert write_crop_clip(video, hits, out) == 3
    assert _count(out) == 3
