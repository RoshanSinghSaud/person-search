"""Turn a Day 2 harness run (reproduce_repo2.py) into golden files.

Golden data are per platform (AUDIT.md F18); ``make golden-baseline`` runs
the harness and this script for the current machine.

Usage::

    python tools/legacy/make_golden.py --run ../legacy/runs/repo2_darwin-arm64 \
        --out tests/golden/data/darwin-arm64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from person_search.eval.golden import GoldenCase


def track_of(raw_frame: int, bbox: list[int] | None, tracks: dict[str, list[dict]]) -> str | None:
    """Recover the track ID of a selected crop from the harness's track index."""
    candidates = [
        tid
        for tid, rows in tracks.items()
        for r in rows
        if r["frame"] == raw_frame and (bbox is None or r["bbox"] == bbox)
    ]
    return candidates[0] if len(candidates) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    for case_dir in sorted(args.run.glob("test_case_*")):
        if not (case_dir / "result.json").exists():
            print(f"{case_dir.name}: no result.json yet, skipped")
            continue
        result = json.loads((case_dir / "result.json").read_text())
        tracks = json.loads((case_dir / "tracks.json").read_text())
        mapping = {}
        for sample, info in result["selected_detail"].items():
            tid = track_of(info["raw_frame"], info["bbox"], tracks)
            if tid is not None:
                mapping[sample] = tid
        golden = GoldenCase(
            case=case_dir.name,
            selected_samples=result["selected_frames"],
            source="repo 2 pole/reid_main.py run_reid, untouched (tools/legacy/reproduce_repo2.py)",
            # weight paths are machine-specific: keep file names only
            config={
                k: Path(v).name if k in ("yolo", "osnet") else v
                for k, v in result["config"].items()
            }
            | {"best_track": result["best_track"]},
            tracks=mapping,
        )
        golden.save(args.out / f"{case_dir.name}.json")
        n_tracks = len(set(mapping.values()))
        print(f"{case_dir.name}: {len(golden.selected_samples)} frames, {n_tracks} tracks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
