"""The detector undoes Ultralytics' thread-count reset when a count is configured (AUDIT F25)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from person_search.ingest.detector import YoloDetector


class _ResettingModel:
    """Stands in for ultralytics.YOLO: resets threads like select_device does."""

    def __call__(self, *args, **kwargs):
        torch.set_num_threads(1)
        return []


def _detector(torch_threads):
    det = object.__new__(YoloDetector)  # skip loading real weights
    det.model = _ResettingModel()
    det.confidence, det.classes, det.resize_width, det.device = 0.4, [0], 640, "cpu"
    det.torch_threads = torch_threads
    return det


@pytest.fixture
def restore_threads():
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


def test_configured_thread_count_survives_a_prediction(restore_threads):
    _detector(3).detect(np.zeros((48, 64, 3), np.uint8))
    assert torch.get_num_threads() == 3


def test_default_keeps_ultralytics_behaviour(restore_threads):
    torch.set_num_threads(3)
    _detector(None).detect(np.zeros((48, 64, 3), np.uint8))
    assert torch.get_num_threads() == 1  # legacy behaviour, as in the golden runs
