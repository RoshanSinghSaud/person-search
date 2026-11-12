"""Step 4: what the deploy script sends to the Hugging Face Space (no network)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from scripts.deploy_space import (
    APP_PORT,
    BUILD_FILES,
    SetupError,
    build_paths,
    check_health,
    secrets_to_write,
    space_readme,
    variables_to_write,
)

ROOT = Path(__file__).resolve().parents[2]
TRACKED = [
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "src/person_search/__init__.py",
    "configs/assets.yaml",
    "tests/unit/test_api.py",
    "docs/decisions/0015-hugging-face-space.md",
    "AUDIT.md",
    ".github/workflows/ci.yml",
]


def test_only_the_build_files_go_to_the_space():
    assert build_paths(TRACKED) == sorted(TRACKED[:8])
    with pytest.raises(SetupError, match=r"uv\.lock"):
        build_paths([p for p in TRACKED if p != "uv.lock"])


def test_this_repo_tracks_every_build_file_and_the_dockerfile_reads_nothing_else():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    build_paths(tracked)  # raises if one is missing
    allowed = [
        line[1:]
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.startswith("!")
    ]
    assert set(allowed) <= set(BUILD_FILES)  # the build context is a subset of the upload


def test_space_settings_match_the_image():
    readme = space_readme("https://github.com/someone/person-search", "0123456789" * 4)
    assert readme.startswith("---\n")
    meta = yaml.safe_load(readme.split("---\n")[1])
    assert meta["sdk"] == "docker"
    assert meta["app_port"] == APP_PORT
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert f"EXPOSE {APP_PORT}" in dockerfile and f"PORT={APP_PORT}" in dockerfile
    assert len(meta["short_description"]) <= 60  # the Hub's limit
    assert "commit `0123456`" in readme


def test_health_check_needs_both_models_and_an_index():
    healthy = {
        "status": "ok",
        "models": {"osnet": "osnet-v1", "irra": "irra-v1"},
        "index": {"points": 16483, "videos": 4},
    }
    assert check_health(healthy) == []
    assert check_health({**healthy, "models": {"osnet": "osnet-v1"}}) == ["no irra model version"]
    assert check_health({**healthy, "index": {"points": 0}}) == ["empty index"]
    assert check_health({}) == [
        "status None",
        "no osnet model version",
        "no irra model version",
        "empty index",
    ]


def test_settings_are_written_only_when_needed():
    # every write restarts the Space
    wanted = {"HF_TOKEN": "t", "PS_API__API_KEY": "k"}
    assert secrets_to_write(wanted, {"HF_TOKEN"}, sync=False) == {"PS_API__API_KEY": "k"}
    assert secrets_to_write(wanted, set(wanted), sync=False) == {}
    assert secrets_to_write(wanted, set(wanted), sync=True) == wanted
    assert variables_to_write({"PS_ASSETS__REPO": "u/a"}, {"PS_ASSETS__REPO": "u/a"}) == {}
    assert variables_to_write({"PS_ASSETS__REPO": "u/b"}, {"PS_ASSETS__REPO": "u/a"}) == {
        "PS_ASSETS__REPO": "u/b"
    }
