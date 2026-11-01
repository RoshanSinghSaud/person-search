"""Golden test: ``pipeline.run(video, photo)`` selects exactly the frames that
repo 2's untouched ``run_reid`` selected on each test case.

Run with ``make golden`` (about 2-4 minutes per case on a laptop CPU).
"""

from __future__ import annotations

import pytest

from person_search.config import load_settings
from person_search.eval.golden import GoldenCase, compare
from person_search.factory import build_components
from person_search.pipeline import run
from person_search.types import OSNET
from tests.conftest import CASES, GOLDEN_DIR, REPO_ROOT, WEIGHTS, require

pytestmark = [pytest.mark.golden, pytest.mark.slow]


@pytest.fixture(scope="module")
def components():
    require(WEIGHTS["yolo"], WEIGHTS["osnet"])
    settings = load_settings(
        REPO_ROOT / "configs" / "report_run.yaml",
        detector={"weights": WEIGHTS["yolo"]},
        osnet={"weights": WEIGHTS["osnet"]},
    )
    return build_components(settings, encoders=[OSNET])


@pytest.mark.parametrize("case", CASES)
def test_selected_frames_match_legacy(case, cases_dir, components):
    golden_file = GOLDEN_DIR / f"{case}.json"
    require(golden_file)
    golden = GoldenCase.load(golden_file)
    out = run(
        cases_dir / case / "test_video.mp4", cases_dir / case / "query.jpg", components=components
    )
    diff = compare(golden, out.result.selected_samples)
    assert diff.ok, f"{case}: {diff}"
    # The same tracks carry the selection (track IDs are deterministic on CPU).
    got = {str(h.sample_index): h.track_id for h in out.result.hits}
    assert {k: got[k] for k in golden.tracks} == golden.tracks
