"""Step 3: upload the service's private assets to a Hugging Face repo, once.

The container downloads its weights and index at start (configs/assets.yaml,
decision 0013). The IRRA weights come from the IRRA team's public release; this
script uploads everything that comes from *your* assets repo: the YOLO and
OSNet weights, the INT8 IRRA image tower if present, and the Week 2 index.
The repo is created private. Every file is checked against the manifest's size
and SHA-256 before anything is uploaded, so the container will accept them.

Needs a Hugging Face token with write access (``hf auth login``, or HF_TOKEN)::

    python scripts/publish_assets.py --repo <hf-user>/person-search-assets \
        --models-dir models --index ../legacy/runs/week2/index        # or: make publish-assets

It prints the commit to pin as ``revision`` in configs/assets.yaml (or set
PS_ASSETS__REVISION). ``--dry-run`` only checks the files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from person_search.assets import load_manifest, sha256_of

REQUIRED_WEIGHTS = ("yolo11n_person.pt", "osnet_campus_v1.pth.tar")


def collect(manifest: dict, models_dir: Path, index_dir: Path) -> tuple[list, list[str]]:
    """``(path_in_repo, local_path)`` pairs to upload, and the problems found."""
    uploads: list[tuple[str, Path]] = []
    problems: list[str] = []

    def check(local: Path, entry: dict, required: bool) -> bool:
        if not local.is_file():
            if required:
                problems.append(f"missing: {local}")
            return False
        if local.stat().st_size != entry["size"] or sha256_of(local) != entry["sha256"]:
            problems.append(f"{local} differs from configs/assets.yaml (size or sha256)")
            return False
        return True

    for name, entry in manifest["weights"].items():
        if entry.get("repo"):  # has its own public source
            continue
        local = models_dir / name
        if check(local, entry, required=name in REQUIRED_WEIGHTS):
            uploads.append((f"weights/{name}", local))
        elif name not in REQUIRED_WEIGHTS and not local.is_file():
            print(f"skipped (not in {models_dir}): {name}")
    for collection, files in manifest["indexes"].items():
        if not (index_dir / "collection" / collection).is_dir():
            print(f"skipped (not in {index_dir}): index {collection}")
            continue
        for rel, entry in files.items():
            if check(index_dir / rel, entry, required=True):
                uploads.append((f"indexes/{collection}/{rel}", index_dir / rel))
    return uploads, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repo", required=True, help="<hf-user>/person-search-assets")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--index", type=Path, default=Path("../legacy/runs/week2/index"))
    ap.add_argument("--manifest", type=Path, default=Path("configs/assets.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    uploads, problems = collect(load_manifest(args.manifest), args.models_dir, args.index)
    for p in problems:
        print("ERROR:", p, file=sys.stderr)
    if problems:
        return 1
    total = sum(local.stat().st_size for _, local in uploads)
    for path_in_repo, local in uploads:
        print(f"  {local} -> {path_in_repo}")
    print(f"{len(uploads)} files, {total / 1e6:.1f} MB, all match configs/assets.yaml")
    if args.dry_run:
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True)
    commit = api.create_commit(
        args.repo,
        operations=[CommitOperationAdd(p, str(local)) for p, local in uploads],
        commit_message="person-search assets (configs/assets.yaml)",
    )
    print(f"\nUploaded to https://huggingface.co/{args.repo} (private)")
    print(f"Pin it: in configs/assets.yaml set  repo: {args.repo}  and  revision: {commit.oid}")
    print("(or set PS_ASSETS__REPO and PS_ASSETS__REVISION where the container runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
