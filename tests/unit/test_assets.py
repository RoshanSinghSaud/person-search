"""Asset fetching at start (decision 0013), with a fake downloader instead of the Hub."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from person_search.assets import (
    STAGING,
    AssetError,
    Source,
    ensure_assets,
    fetch_assets,
    load_manifest,
    plan_assets,
)
from person_search.cli import main
from person_search.config import AssetsSettings, load_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX = "detections__irra-test__osnet-test"
BLOBS = {
    "yolo.pt": b"yolo weights",
    "osnet.pth": b"osnet weights!",
    "irra.pth": b"irra weights, from the IRRA team's repo",
    "meta.json": b'{"collections": {}}',
    "storage.sqlite": b"sqlite bytes " * 100,
}


def entry(name: str, **extra) -> dict:
    data = BLOBS[name]
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), **extra}


def manifest(repo: str | None = "me/assets") -> dict:
    return {
        "repo": repo,
        "revision": "abc123",
        "weights": {
            "yolo.pt": entry("yolo.pt"),
            "osnet.pth": entry("osnet.pth"),
            "irra.pth": entry(
                "irra.pth", repo="team/irra", filename="best.pth", revision="deadbeef"
            ),
        },
        "indexes": {
            INDEX: {
                "collection/" + INDEX + "/storage.sqlite": entry("storage.sqlite"),
                "meta.json": entry("meta.json"),
            }
        },
    }


@pytest.fixture
def settings(tmp_path):
    models = tmp_path / "models"
    return load_settings(
        detector={"weights": models / "yolo.pt"},
        osnet={"weights": models / "osnet.pth", "version": "osnet-test"},
        irra={"weights": models / "irra.pth", "version": "irra-test"},
        api={"index_path": tmp_path / "index"},
    )


class FakeHub:
    """Serves BLOBS by base name; ``corrupt`` names come back altered."""

    def __init__(self, corrupt: tuple[str, ...] = ()) -> None:
        self.calls: list[Source] = []
        self.corrupt = corrupt

    def __call__(self, source: Source, out_dir: Path) -> Path:
        self.calls.append(source)
        name = Path(source.filename).name
        name = "irra.pth" if name == "best.pth" else name
        path = out_dir / source.filename  # like hf_hub_download(local_dir=...)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = BLOBS[name]
        path.write_bytes(data[:-1] + b"X" if name in self.corrupt else data)
        return path


def test_real_manifest_is_valid_and_agrees_with_the_model_registry():
    assets = load_manifest(REPO_ROOT / "configs" / "assets.yaml")
    registry = yaml.safe_load((REPO_ROOT / "configs" / "models.yaml").read_text())["models"]
    by_file = {m["image_onnx"] if "image_onnx" in m else m["file"]: m for m in registry.values()}
    for name, e in assets["weights"].items():
        if name in by_file:
            assert e["sha256"] == by_file[name]["sha256"], name
    assert {"osnet_campus_v1.pth.tar", "irra_v1.pth", "yolo11n_person.pt"} <= set(assets["weights"])
    assert "detections__irra-v1__osnet-v1" in assets["indexes"]
    # the default settings' files and collection are all covered
    default = load_settings()
    assert {p.name for p in [default.detector.weights, default.osnet.weights]} <= set(
        assets["weights"]
    )


def test_fresh_start_downloads_everything_from_the_right_repos(settings, tmp_path):
    plan = plan_assets(settings, manifest())
    assert not plan.problems and not plan.present
    sources = {a.dest.name: a.source for a in plan.fetch}
    assert sources["yolo.pt"] == Source("me/assets", "weights/yolo.pt", "abc123")
    assert sources["irra.pth"] == Source("team/irra", "best.pth", "deadbeef")
    assert sources["meta.json"] == Source("me/assets", f"indexes/{INDEX}/meta.json", "abc123")
    # the collection folder is moved into place last
    assert plan.fetch[-1].dest.name == "storage.sqlite"

    hub = FakeHub()
    fetch_assets(plan, hub)
    assert (tmp_path / "models" / "irra.pth").read_bytes() == BLOBS["irra.pth"]
    assert (tmp_path / "index" / "collection" / INDEX / "storage.sqlite").is_file()
    assert (tmp_path / "index" / "meta.json").read_bytes() == BLOBS["meta.json"]
    assert not (tmp_path / "models" / STAGING).exists()
    assert not (tmp_path / "index" / STAGING).exists()

    again = plan_assets(settings, manifest())
    assert again.fetch == [] and len(again.present) == 4


def test_a_bad_download_installs_nothing(settings, tmp_path):
    plan = plan_assets(settings, manifest())
    with pytest.raises(AssetError, match="sha256"):
        fetch_assets(plan, FakeHub(corrupt=("storage.sqlite",)))
    # nothing moved, not even the files that downloaded fine; no staging left
    assert not (tmp_path / "models").exists() or not any(
        p.is_file() for p in (tmp_path / "models").rglob("*")
    )
    assert not (tmp_path / "index" / "meta.json").exists()


def test_present_files_are_kept_and_a_grown_index_is_never_replaced(settings, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    for name in ("yolo.pt", "osnet.pth", "irra.pth"):
        (models / name).write_bytes(BLOBS[name])
    grown = tmp_path / "index" / "collection" / INDEX
    grown.mkdir(parents=True)
    (grown / "storage.sqlite").write_bytes(b"more videos than the published index" * 999)
    plan = plan_assets(settings, manifest(repo=None))  # no repo needed: nothing to fetch
    assert plan.fetch == [] and not plan.problems
    hub = FakeHub()
    ensure_assets_with(settings, hub)
    assert hub.calls == []


def ensure_assets_with(settings, hub):
    path = settings.api.index_path.parent / "assets.yaml"
    path.write_text(yaml.safe_dump(manifest(repo=None)))
    return ensure_assets(settings.model_copy(update={"assets": AssetsSettings(manifest=path)}), hub)


def test_wrong_size_weight_is_downloaded_again(settings, tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "yolo.pt").write_bytes(b"truncated")
    plan = plan_assets(settings, manifest())
    assert any("wrong size" in n for n in plan.notes)
    assert "yolo.pt" in {a.dest.name for a in plan.fetch}


def test_missing_repo_or_unknown_file_is_a_clear_error(settings, tmp_path):
    plan = plan_assets(settings, manifest(repo=None))
    assert any("PS_ASSETS__REPO" in p for p in plan.problems)
    # the IRRA file has its own public repo, so it is still planned
    assert [a.dest.name for a in plan.fetch] == ["irra.pth"]
    with pytest.raises(AssetError, match="PS_ASSETS__REPO"):
        fetch_assets(plan, FakeHub())

    m = manifest()
    del m["weights"]["osnet.pth"]
    assert any("not in the manifest" in p for p in plan_assets(settings, m).problems)


def test_repo_and_revision_settings_override_the_manifest(settings):
    s = settings.model_copy(update={"assets": AssetsSettings(repo="other/repo", revision="f00d")})
    sources = {a.dest.name: a.source for a in plan_assets(s, manifest()).fetch}
    assert sources["osnet.pth"] == Source("other/repo", "weights/osnet.pth", "f00d")
    assert sources["irra.pth"].repo == "team/irra"  # an entry's own repo wins


def test_index_folder_for_other_models_is_refused(settings, tmp_path):
    (tmp_path / "index" / "collection" / "detections__irra-v9__osnet-v9").mkdir(parents=True)
    plan = plan_assets(settings, manifest())
    assert any("empty PS_API__INDEX_PATH" in p for p in plan.problems)


def test_unpublished_index_starts_empty(settings):
    m = manifest()
    m["indexes"] = {}
    plan = plan_assets(settings, m)
    assert any("starts with an empty one" in n for n in plan.notes)
    assert all(a.dest.suffix != ".sqlite" for a in plan.fetch)


def test_cli_dry_run_prints_the_plan(tmp_path, monkeypatch, capsys):
    path = tmp_path / "assets.yaml"
    path.write_text(yaml.safe_dump(manifest()))
    monkeypatch.setenv("PS_ASSETS__MANIFEST", str(path))
    monkeypatch.setenv("PS_DETECTOR__WEIGHTS", str(tmp_path / "m" / "yolo.pt"))
    monkeypatch.setenv("PS_OSNET__WEIGHTS", str(tmp_path / "m" / "osnet.pth"))
    monkeypatch.setenv("PS_IRRA__WEIGHTS", str(tmp_path / "m" / "irra.pth"))
    monkeypatch.setenv("PS_API__INDEX_PATH", str(tmp_path / "idx"))
    assert main(["fetch-assets", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["fetch"]) == 3 and out["problems"] == []  # versions differ: no index entry
    assert any("empty one" in n for n in out["notes"])


def test_fetch_and_serve_exit_cleanly_when_assets_cannot_be_fetched(tmp_path, monkeypatch, capsys):
    path = tmp_path / "assets.yaml"
    path.write_text(yaml.safe_dump(manifest(repo=None)))
    monkeypatch.setenv("PS_ASSETS__MANIFEST", str(path))
    monkeypatch.setenv("PS_DETECTOR__WEIGHTS", str(tmp_path / "m" / "yolo.pt"))
    monkeypatch.setenv("PS_API__INDEX_PATH", str(tmp_path / "idx"))
    assert main(["fetch-assets"]) == 2
    assert main(["serve", "--port", "0"]) == 2  # never reaches uvicorn
    assert "PS_ASSETS__REPO" in capsys.readouterr().err  # a log line, not a traceback


def test_publish_collects_only_files_that_match_the_manifest(tmp_path):
    from scripts.publish_assets import collect

    models, index = tmp_path / "models", tmp_path / "index"
    (index / "collection" / INDEX).mkdir(parents=True)
    models.mkdir()
    for name in ("yolo.pt", "osnet.pth"):
        (models / name).write_bytes(BLOBS[name])
    (index / "meta.json").write_bytes(BLOBS["meta.json"])
    (index / "collection" / INDEX / "storage.sqlite").write_bytes(BLOBS["storage.sqlite"])
    m = manifest()
    m["weights"]["yolo11n_person.pt"] = m["weights"].pop("yolo.pt")  # a required file name
    (models / "yolo11n_person.pt").write_bytes(BLOBS["yolo.pt"])
    uploads, problems = collect(m, models, index)
    assert problems == []
    assert sorted(p for p, _ in uploads) == [
        f"indexes/{INDEX}/collection/{INDEX}/storage.sqlite",
        f"indexes/{INDEX}/meta.json",
        "weights/osnet.pth",
        "weights/yolo11n_person.pt",
    ]  # irra.pth has its own public repo: not uploaded
    (models / "osnet.pth").write_bytes(b"tampered")
    _, problems = collect(m, models, index)
    assert any("differs" in p for p in problems)


def test_store_videos_keeps_originals_under_their_video_id(tmp_path, monkeypatch, capsys):
    from person_search.index.points import video_fingerprint

    src = tmp_path / "case" / "test_video.MP4"
    src.parent.mkdir()
    src.write_bytes(b"not really a video, but hashed the same way")
    monkeypatch.setenv("PS_API__VIDEO_DIR", str(tmp_path / "videos"))
    assert main(["store-videos", str(src)]) == 0
    stored = tmp_path / "videos" / f"{video_fingerprint(src)}.mp4"
    assert stored.read_bytes() == src.read_bytes()
    assert main(["store-videos", str(src)]) == 0  # again: nothing to do
