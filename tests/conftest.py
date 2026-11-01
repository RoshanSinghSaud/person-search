"""Shared fixtures and path resolution for optional assets.

Unit tests need nothing but the package. Parity and golden tests need the
legacy checkouts and weights; they look for them via environment variables and
skip (with the reason) when absent:

* ``PS_LEGACY_ROOT``  - folder holding ``person-reid-clip/`` and ``amrit-pbvs/``
  (default: ``../legacy`` next to this repo)
* ``PS_MODELS_DIR``   - folder with the weights (default: ``models/``)
* ``PS_CASES_DIR``    - folder with ``test_case_00X/`` (default: ``<legacy>/amrit-pbvs``)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default)).resolve()


LEGACY_ROOT = _path("PS_LEGACY_ROOT", REPO_ROOT.parent / "legacy")
MODELS_DIR = _path("PS_MODELS_DIR", REPO_ROOT / "models")
CASES_DIR = _path("PS_CASES_DIR", LEGACY_ROOT / "amrit-pbvs")
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "data"

WEIGHTS = {
    "yolo": MODELS_DIR / "yolo11n_person.pt",
    "osnet": MODELS_DIR / "osnet_campus_v1.pth.tar",
    "osnet_market": MODELS_DIR / "osnet_market_v0.pth",
    "irra": MODELS_DIR / "irra_v1.pth",
}
CASES = ("test_case_001", "test_case_002", "test_case_003", "test_case_004")


def require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"missing: {', '.join(missing)}")


@pytest.fixture(scope="session")
def legacy_root() -> Path:
    require(LEGACY_ROOT / "amrit-pbvs", LEGACY_ROOT / "person-reid-clip")
    return LEGACY_ROOT


@pytest.fixture(scope="session")
def cases_dir() -> Path:
    require(*(CASES_DIR / c / "test_video.mp4" for c in CASES))
    return CASES_DIR
