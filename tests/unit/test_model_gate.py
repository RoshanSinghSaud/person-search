"""Day 24: the model gate's plan, weight sources and verdict (no network, no models)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from scripts.model_gate import (
    GateError,
    Source,
    expected_hashes,
    make_plan,
    needed_files,
    source_for,
    verdict_report,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
MANIFEST = yaml.safe_load((ROOT / "configs" / "assets.yaml").read_text())


def pinned(**production: str) -> dict:
    head = copy.deepcopy(MODELS)
    head["production"].update(production)
    return head


# ---------------------------------------------------------------------- plan
def test_no_pin_change_means_nothing_to_score():
    assert make_plan(MODELS, copy.deepcopy(MODELS))["pairs"] == []


@pytest.mark.parametrize(
    ("change", "pair"),
    [
        ({"irra": "irra-v0"}, ("irra", "irra-v1", "irra-v0")),
        ({"irra": "irra-v2"}, ("irra", "irra-v1", "irra-v2")),
        ({"osnet": "osnet-v0"}, ("osnet", "osnet-v1", "osnet-v0")),
    ],
)
def test_a_changed_pin_is_scored_against_production(change, pair):
    plan = make_plan(MODELS, pinned(**change))
    assert [(p["family"], p["baseline"], p["candidate"]) for p in plan["pairs"]] == [pair]
    assert set(plan["baseline_models"]) == {pair[1]}
    assert set(plan["candidate_models"]) == {pair[2]}


def test_both_families_at_once():
    plan = make_plan(MODELS, pinned(irra="irra-v0", osnet="osnet-v0"))
    assert [p["family"] for p in plan["pairs"]] == ["irra", "osnet"]


def test_versions_are_immutable():
    head = copy.deepcopy(MODELS)
    head["models"]["irra-v1"]["sha256"] = "0" * 64  # new weights under an old name
    with pytest.raises(GateError, match=r"immutable.*irra-v1"):
        make_plan(MODELS, head)


@pytest.mark.parametrize(
    ("production", "message"),
    [
        ({"irra": "irra-v9"}, "not listed"),
        ({"irra": "osnet-v0"}, "is a osnet model"),
    ],
)
def test_bad_pins_are_refused(production, message):
    with pytest.raises(GateError, match=message):
        make_plan(MODELS, pinned(**production))


def test_version_names_cannot_inject_shell():
    head = pinned(irra="x; curl evil")
    head["models"]["x; curl evil"] = {**MODELS["models"]["irra-v0"]}
    with pytest.raises(GateError, match="version names"):
        make_plan(MODELS, head)


# --------------------------------------------------------------------- fetch
def test_every_version_has_a_source_and_a_hash():
    plan = make_plan(MODELS, pinned(irra="irra-v2"))
    plan["baseline_models"] |= {v: MODELS["models"][v] for v in MODELS["models"]}
    hashes = expected_hashes(MODELS["models"], MANIFEST)
    for name in needed_files(plan):
        assert len(hashes[name]) == 64, name
        source_for(name, MANIFEST, MANIFEST["repo"])  # raises when there is none


def test_sources():
    repo, pin = MANIFEST["repo"], MANIFEST["revision"]
    # the service's own files: from the manifest, at its pinned commit
    assert source_for("osnet_campus_v1.pth.tar", MANIFEST, repo) == Source(
        "hf", repo, "weights/osnet_campus_v1.pth.tar", pin
    )
    assert source_for("irra_v1.pth", MANIFEST, repo).where == "prasish17/clip-text-query"
    # vanilla CLIP from OpenAI; the Market-1501 OSNet from the assets repo's latest commit
    assert source_for("clip_vit_b16_v0.pt", MANIFEST, repo).kind == "url"
    assert source_for("osnet_market_v0.pth", MANIFEST, repo) == Source(
        "hf", repo, "weights/osnet_market_v0.pth", "main"
    )
    with pytest.raises(GateError, match="no source"):
        source_for("osnet_market_v0.pth", MANIFEST, None)


def test_irra_v2_needs_v1_text_tower_and_its_own_onnx():
    plan = make_plan(MODELS, pinned(irra="irra-v2"))
    assert needed_files(plan) == ["irra_v1.pth", "irra_v2_image_int8.onnx"]
    hashes = expected_hashes(MODELS["models"], MANIFEST)
    assert hashes["irra_v2_image_int8.onnx"] == MODELS["models"]["irra-v2"]["sha256"]
    assert hashes["irra_v1.pth"] == MODELS["models"]["irra-v1"]["sha256"]


# ------------------------------------------------------------------- verdict
def irra(hits: list[str], map_: float) -> dict:
    queries = ["q1", "q2", "q3", "q4"]
    return {
        "family": "irra",
        "metrics": {
            "campus_mAP_cosine": map_,
            "campus_rank1_cosine": map_ + 0.05,
            "text_hit1_count": len(hits),
            "text_hit5_count": 4,
            "text_mrr": 0.5,
            "text_n": 4,
        },
        "details": {"text": [{"query_id": q, "hit@1": q in hits} for q in queries]},
        "seconds": 10.0,
    }


def results(**versions: dict) -> dict:
    return {"versions": versions, "gate_manifest_built": "2026-11-09 05:59 UTC"}


PAIR = {"pairs": [{"family": "irra", "baseline": "irra-v1", "candidate": "irra-v9"}]}


def test_a_worse_candidate_is_blocked_and_the_report_says_why():
    blocked, report = verdict_report(
        PAIR,
        results(**{"irra-v1": irra(["q1", "q2"], 0.78)}),
        results(**{"irra-v9": irra(["q1", "q3"], 0.12)}),
    )
    assert blocked
    assert "## Model gate: ❌ blocked" in report
    assert "| text Hit@1 | 2/4 | 2/4 | +0 | ❌ no query lost |" in report
    assert (
        "| image tower campus mAP | 78.00% | 12.00% | -66.00 pts | ❌ may drop at most 1 pt |"
        in report
    )
    assert "text Hit@1 lost on q2" in report  # same count, but a production hit was lost


def test_an_equal_or_better_candidate_passes():
    blocked, report = verdict_report(
        PAIR,
        results(**{"irra-v1": irra(["q1", "q2"], 0.78)}),
        results(**{"irra-v9": irra(["q1", "q2", "q3"], 0.775)}),
    )
    assert not blocked
    assert "## Model gate: ✅ passed" in report
    assert (
        "| image tower campus mAP | 78.00% | 77.50% | -0.50 pts | ✅ may drop at most 1 pt |"
        in report
    )
