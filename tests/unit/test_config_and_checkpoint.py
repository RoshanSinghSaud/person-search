from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from person_search.config import load_settings
from person_search.errors import CheckpointMismatchError, ModelLoadError
from person_search.models.checkpoint import drop_prefixes, load_strict, read_state_dict

PROFILES = Path(__file__).resolve().parents[2] / "configs"


def test_defaults_are_repo2_values(monkeypatch):
    monkeypatch.delenv("PS_CONFIG_FILE", raising=False)
    s = load_settings()
    assert s.detector.confidence == 0.4
    assert s.scoring.base_threshold == 0.65 and s.scoring.relative_margin == 0.1
    assert s.tracker.max_age == 30 and s.tracker.n_init == 3
    assert s.tracker.max_cosine_distance == 0.2
    assert s.sampling.target_fps == 12 and s.sampling.embed_every == 3


def test_env_overrides_nested(monkeypatch):
    monkeypatch.setenv("PS_SCORING__BASE_THRESHOLD", "0.7")
    monkeypatch.setenv("PS_TRACKER__MAX_COSINE_DISTANCE", "0.3")
    s = load_settings()
    assert s.scoring.base_threshold == 0.7
    assert s.tracker.max_cosine_distance == 0.3


def test_yaml_profile_then_env_then_kwargs(tmp_path, monkeypatch):
    profile = tmp_path / "p.yaml"
    profile.write_text("scoring:\n  base_threshold: 0.5\ndetector:\n  confidence: 0.5\n")
    s = load_settings(profile)
    assert s.scoring.base_threshold == 0.5 and s.detector.confidence == 0.5
    monkeypatch.setenv("PS_DETECTOR__CONFIDENCE", "0.3")
    assert load_settings(profile).detector.confidence == 0.3
    assert load_settings(profile, device="cpu").scoring.base_threshold == 0.5


@pytest.mark.parametrize("name", ["report_run.yaml", "paper.yaml"])
def test_shipped_profiles_load(name):
    load_settings(PROFILES / name)


def test_missing_profile_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.yaml")


def test_invalid_values_rejected():
    with pytest.raises(ValidationError):
        load_settings(device="tpu")
    with pytest.raises(ValidationError):
        load_settings(irra={"precision": "fp16"})  # fp16 needs a GPU


def test_read_state_dict_unwraps_and_strips_module(tmp_path):
    p = tmp_path / "c.pth"
    torch.save({"state_dict": {"module.a": torch.ones(1), "module.b": torch.zeros(2)}}, p)
    assert set(read_state_dict(p)) == {"a", "b"}


def test_read_state_dict_errors(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        read_state_dict(tmp_path / "x.pth")
    p = tmp_path / "empty.pth"
    torch.save({"epoch": 3}, p)
    with pytest.raises(ModelLoadError, match="no tensors"):
        read_state_dict(p)


def test_drop_prefixes():
    s = {"classifier.weight": 1, "conv1.weight": 2}
    assert drop_prefixes(s, ["classifier."]) == {"conv1.weight": 2}


def test_load_strict_reports_every_problem():
    model = torch.nn.Linear(2, 3)
    with pytest.raises(CheckpointMismatchError) as err:
        load_strict(
            model,
            {"weight": torch.zeros(3, 4), "extra": torch.zeros(1)},
            model_name="lin",
            source="mem",
        )
    e = err.value
    assert e.missing == ["bias"] and e.unexpected == ["extra"] and len(e.shape_mismatch) == 1


def test_load_strict_allow_missing():
    model = torch.nn.Linear(2, 3)
    report = load_strict(
        model, {"weight": torch.ones(3, 2)}, model_name="lin", source="mem", allow_missing=["bias"]
    )
    assert report.allowed_missing == ("bias",)
    assert torch.equal(model.weight, torch.ones(3, 2))
