"""Day 8: build the gate set folder, and upload it to a private Hugging Face dataset.

The gate set is everything the quality gate needs to score a model without
videos or detection:

    campus/query/, campus/bounding_box_test/   campus ReID test split (30 IDs, never trained on)
    cases/<case>.npz                           cached crops + observations (eval/gate_set.py)
    cases/<case>/evaluation.json, query.jpg    frame labels and the query photo
    text/text_queries.yaml, text_targets.yaml  the text test set (Day 9)
    manifest.json                              SHA-256 and size of every file
    README.md                                  dataset card

It stays private: the campus images and the test videos' crops show real
students. Only metric numbers are published.

Usage::

    python scripts/gate_set.py build --ingest-dir ../legacy/runs/ingest_v1 \
        --cases-dir ../legacy/amrit-pbvs --campus ../custom_reid_dataset --out ../legacy/gate_set
    HF_TOKEN=... python scripts/gate_set.py upload --dir ../legacy/gate_set \
        --repo <user>/person-search-gate
    python scripts/gate_set.py download --repo <user>/person-search-gate --dir ../legacy/gate_set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from person_search.eval.gate_set import export_case
from person_search.index.points import video_fingerprint
from person_search.ingest.pipeline import IngestResult

CASES = ("test_case_001", "test_case_002", "test_case_003", "test_case_004")
DATA = Path(__file__).resolve().parents[1] / "tests" / "golden" / "data"

CARD = """---
license: other
pretty_name: person-search gate set
---
# person-search gate set (private)

Evaluation data for the person-search quality gate. **Private**: the images show
real students on a college campus. Publish metric numbers only.

| Part | Contents | Used for |
|---|---|---|
| `campus/` | query (284) and gallery (1191) crops of the 30 campus IDs held out \
of OSNet fine-tuning, Market-1501 layout | OSNet campus mAP / Rank-k |
| `cases/*.npz` | the 4 test videos' tracked detections and a lossless crop of \
every embedded one (default ingest, linux-x86_64) | photo frame F1 and text Hit@k |
| `cases/*/` | `evaluation.json` frame labels, `query.jpg` | ground truth, photo queries |
| `text/` | 12 witness descriptions and the target track segments | text Hit@k |

Missing: the college text-image test split (IRRA's published 65.22% Rank-1).
Until it arrives, the text gate is end-to-end Hit@k on the 12 descriptions.

Built by `scripts/gate_set.py build` on {built}. `manifest.json` lists every file's SHA-256.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(args: argparse.Namespace) -> int:
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    for split in ("query", "bounding_box_test"):
        src = args.campus / split
        if not src.is_dir():
            raise SystemExit(f"missing campus split folder {src}")
        shutil.copytree(src, out / "campus" / split, ignore=shutil.ignore_patterns(".*"))
    for case in args.cases:
        video = args.cases_dir / case / "test_video.mp4"
        ingest = IngestResult.load(args.ingest_dir / f"{case}.npz")
        export_case(
            case,
            video,
            ingest,
            out / "cases" / f"{case}.npz",
            extra_meta={
                "video_id": video_fingerprint(video),
                "source": "default ingest (det 0.4, max_cosine_distance 0.2), linux-x86_64",
            },
        )
        (out / "cases" / case).mkdir(parents=True, exist_ok=True)
        shutil.copy(
            DATA / "labels" / case / "evaluation.json", out / "cases" / case / "evaluation.json"
        )
        shutil.copy(args.cases_dir / case / "query.jpg", out / "cases" / case / "query.jpg")
        print("cached", case)
    (out / "text").mkdir(parents=True)
    for name in ("text_queries.yaml", "text_targets.yaml"):
        shutil.copy(DATA / name, out / "text" / name)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (out / "README.md").write_text(CARD.replace("{built}", built))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "built": built,
        "n_files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "files": {
            str(p.relative_to(out)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(files)} files, {manifest['bytes'] / 1e6:.1f} MB -> {out}")
    return 0


def upload(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN (a write token) first")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(args.repo, repo_type="dataset")
    if not info.private:
        raise SystemExit(f"{args.repo} exists and is PUBLIC; refusing to upload the gate set there")
    api.upload_folder(
        repo_id=args.repo, repo_type="dataset", folder_path=str(args.dir), commit_message="gate set"
    )
    print(f"uploaded {args.dir} -> https://huggingface.co/datasets/{args.repo} (private)")
    return 0


def download(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        args.repo, repo_type="dataset", local_dir=str(args.dir), token=os.environ.get("HF_TOKEN")
    )
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    bad = [f for f, m in manifest["files"].items() if sha256(Path(path) / f) != m["sha256"]]
    if bad:
        raise SystemExit(f"checksum mismatch: {bad[:5]}")
    print(f"gate set ready in {path} ({manifest['n_files']} files verified)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ingest-dir", type=Path, required=True)
    b.add_argument("--cases-dir", type=Path, required=True)
    b.add_argument("--campus", type=Path, required=True, help="custom_reid_dataset folder")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--cases", nargs="+", default=list(CASES))
    b.set_defaults(func=build)
    u = sub.add_parser("upload")
    u.add_argument("--dir", type=Path, required=True)
    u.add_argument("--repo", required=True, help="<user>/<dataset name>")
    u.set_defaults(func=upload)
    d = sub.add_parser("download")
    d.add_argument("--repo", required=True)
    d.add_argument("--dir", type=Path, required=True)
    d.set_defaults(func=download)
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
