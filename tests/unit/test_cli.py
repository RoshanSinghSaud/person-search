from __future__ import annotations

from pathlib import Path

import yaml

from person_search.cli import main

LABELS = Path(__file__).resolve().parents[1] / "golden" / "data" / "labels"


def test_eval_frames_prints_report_numbers(capsys):
    assert main(["eval-frames", "--cases-dir", str(LABELS)]) == 0
    out = capsys.readouterr().out
    assert "SYSTEM" in out and "89.05%" in out


def test_eval_frames_without_labels_fails(tmp_path):
    assert main(["eval-frames", "--cases-dir", str(tmp_path)]) == 2


def test_config_prints_resolved_yaml(capsys, monkeypatch):
    monkeypatch.setenv("PS_SCORING__BASE_THRESHOLD", "0.61")
    assert main(["config"]) == 0
    cfg = yaml.safe_load(capsys.readouterr().out)
    assert cfg["scoring"]["base_threshold"] == 0.61
    assert cfg["tracker"]["n_init"] == 3
