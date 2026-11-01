"""Download the model weights into ``models/`` under their registry names.

Sources:
* team Google Drive folder (repo 2 README): YOLO11n, OSNet campus, OSNet Market-1501
* Hugging Face ``prasish17/clip-text-query``: IRRA (``best_model_text.pth``)

Prints the SHA-256 of every file; record those when registering the models
(Day 8). Re-running skips files that are already present.

Usage::

    python scripts/fetch_weights.py --out models
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1VQd-cBfqZlj2fPI_jKezDSEI3FgAWSlQ"
DRIVE_FILES = {
    "yolo_finetuned_best.pt": "yolo11n_person.pt",
    "osnet_finetuned_model.pth.tar-60": "osnet_campus_v1.pth.tar",
    "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth": (  # noqa: E501
        "osnet_market_v0.pth"
    ),
}
HF_FILES = {("prasish17/clip-text-query", "best_model_text.pth"): "irra_v1.pth"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_drive(out: Path) -> None:
    targets = [out / name for name in DRIVE_FILES.values()]
    if all(t.exists() for t in targets):
        return
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required: uv sync --group dev")
    with tempfile.TemporaryDirectory() as tmp:
        gdown.download_folder(DRIVE_FOLDER, output=tmp, quiet=False)
        for src_name, dst_name in DRIVE_FILES.items():
            src = Path(tmp) / src_name
            if not src.exists():
                sys.exit(f"Drive folder no longer contains {src_name}")
            shutil.move(src, out / dst_name)


def fetch_hf(out: Path) -> None:
    from huggingface_hub import hf_hub_download

    for (repo_id, filename), dst_name in HF_FILES.items():
        dst = out / dst_name
        if dst.exists():
            continue
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copyfile(path, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--skip-drive", action="store_true")
    ap.add_argument("--skip-hf", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_drive:
        fetch_drive(args.out)
    if not args.skip_hf:
        fetch_hf(args.out)
    for name in [*DRIVE_FILES.values(), *HF_FILES.values()]:
        p = args.out / name
        status = f"{sha256(p)}  {p.stat().st_size / 1e6:8.1f} MB" if p.exists() else "MISSING"
        print(f"{name:<28} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
