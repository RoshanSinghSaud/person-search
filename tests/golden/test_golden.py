"""Golden test: ``pipeline.run(video, photo)`` selects exactly the frames that
repo 2's untouched ``run_reid`` selects on each test case, on the same platform.

Golden data live in ``tests/golden/data/<platform>/`` (e.g. ``linux-x86_64``,
``darwin-arm64``). Record them once per machine type with
``make golden-baseline``, then run ``make golden``.
"""

from __future__ import annotations

import pytest

from person_search.config import load_settings
from person_search.eval.golden import GoldenCase, compare
from person_search.factory import build_components
from person_search.pipeline import run
from person_search.types import OSNET
from tests.conftest import CASES, GOLDEN_DIR, PLATFORM, REPO_ROOT, WEIGHTS, require

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
    if not golden_file.exists():
        pytest.skip(f"no golden data for {PLATFORM}: run `make golden-baseline` on this machine")
    golden = GoldenCase.load(golden_file)
    out = run(
        cases_dir / case / "test_video.mp4", cases_dir / case / "query.jpg", components=components
    )
    diff = compare(golden, out.result.selected_samples)
    assert diff.ok, f"{case}: {diff}"
    # The same tracks carry the selection (track IDs are deterministic on CPU).
    got = {str(h.sample_index): h.track_id for h in out.result.hits}
    assert {k: got[k] for k in golden.tracks} == golden.tracks
