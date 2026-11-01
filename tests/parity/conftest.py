"""Fixtures for parity tests: legacy modules and a fixed set of person crops."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

from tests.conftest import CASES, CASES_DIR, require

N_CROPS = 20


def load_legacy_module(path: Path, name: str) -> ModuleType:
    """Import a legacy file by path without touching sys.path or the file."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def person_crops() -> list[np.ndarray]:
    """20 deterministic crops: the 4 query photos plus 16 person-shaped boxes
    from frames of the test videos (sizes and aspect ratios vary on purpose)."""
    require(*(CASES_DIR / c / "test_video.mp4" for c in CASES))
    rng = np.random.default_rng(1234)
    crops = [cv2.imread(str(CASES_DIR / c / "query.jpg")) for c in CASES]
    per_video = (N_CROPS - len(crops)) // len(CASES)
    for case in CASES:
        cap = cv2.VideoCapture(str(CASES_DIR / case / "test_video.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(rng.integers(0, n)))
            ok, frame = cap.read()
            assert ok
            h, w = frame.shape[:2]
            bh = int(rng.integers(80, 500))
            bw = max(12, int(bh * rng.uniform(0.3, 0.6)))
            x, y = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
            crops.append(frame[y : y + bh, x : x + bw].copy())
        cap.release()
    assert len(crops) == N_CROPS and all(c is not None for c in crops)
    return crops


DESCRIPTIONS = [
    "A young man wearing a black jacket, blue jeans and white sneakers, carrying a backpack.",
    "A woman in a red kurta with a white shawl, walking slowly.",
    "Person with short hair, grey hoodie and dark trousers.",
    "A student in a white shirt and black pants holding a phone in his right hand.",
    # long enough that the 8 prompt tokens push EOT to the clamp (legacy edge case)
    "A tall man with short black hair wearing a light blue denim jacket over a plain white "
    "t-shirt, dark grey cargo trousers with many pockets, brown leather boots, a black "
    "backpack on both shoulders, a silver wrist watch, a black cap, round glasses, a grey "
    "scarf, and holding a red umbrella in his left hand while talking on a phone near the gate.",
]


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)
