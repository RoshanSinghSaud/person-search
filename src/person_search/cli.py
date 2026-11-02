"""Command-line interface: ``person-search <command>``.

Commands
--------
config        print the resolved settings
ingest        ingest one video and save its observations (.npz)
photo         photo query on one video (reprocesses the video, the legacy path)
text          text query on one video, ranked tracks
eval-frames   frame-level P/R/F1 from the test cases' evaluation.json files
golden        run the photo pipeline on the test cases and compare with golden data
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path

from person_search.config import load_settings
from person_search.logging_utils import configure_logging

log = logging.getLogger("person_search.cli")


def _cmd_config(args: argparse.Namespace) -> int:
    print(load_settings(args.config).to_yaml())
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from person_search.factory import build_components

    settings = load_settings(args.config)
    components = build_components(settings, encoders=args.encoders)
    result = components.ingest_pipeline().run(args.video)
    path = result.save(args.out)
    print(
        json.dumps(
            {
                "saved": str(path),
                "samples": result.n_samples,
                "tracks": len(result.track_ids),
                "observations": len(result.observations),
                "wall_seconds": round(result.wall_seconds, 2),
                "samples_per_s": round(result.fps_processed, 2),
                "timings": result.timings,
            },
            indent=2,
        )
    )
    return 0


def _cmd_photo(args: argparse.Namespace) -> int:
    from person_search.pipeline import run
    from person_search.render import write_annotated_video, write_crop_clip

    settings = load_settings(args.config)
    out = run(args.video, args.photo, settings)
    res = out.result
    best = res.decision.best
    payload = {
        "video": str(args.video),
        "matched": res.matched,
        "best_track": best.track_id if best else None,
        "best_score": best.score if best else None,
        "effective_threshold": res.decision.effective_threshold,
        "accepted_tracks": list(res.decision.accepted),
        "selected_samples": res.selected_samples,
        "segments": [
            {"start_frame": s.start_frame, "end_frame": s.end_frame, "n_samples": s.n_samples}
            for s in res.segments()
        ],
        "timings": out.ingest.timings,
        "wall_seconds": round(out.ingest.wall_seconds, 2),
    }
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(payload, indent=2))
        if args.render and res.hits:
            write_annotated_video(
                args.video, res.hits, out_dir / "output_annotated.mp4", settings.sampling.target_fps
            )
            write_crop_clip(
                args.video, res.hits, out_dir / "output.mp4", settings.sampling.target_fps
            )
    print(json.dumps({k: v for k, v in payload.items() if k != "selected_samples"}, indent=2))
    return 0 if res.matched else 3


def _cmd_text(args: argparse.Namespace) -> int:
    from person_search.pipeline import run_text

    out = run_text(args.video, args.query, args.top_k, load_settings(args.config))
    print(
        json.dumps(
            {
                "video": str(args.video),
                "query": args.query,
                "ranked_tracks": [
                    {"track_id": t.track_id, "score": round(t.score, 4), "n_frames": t.n_frames}
                    for t in out.ranked
                ],
                "wall_seconds": round(out.ingest.wall_seconds, 2),
            },
            indent=2,
        )
    )
    return 0


def _cmd_eval_frames(args: argparse.Namespace) -> int:
    from person_search.eval.frame_metrics import format_table, load_cases, metrics_from_labels

    cases = load_cases(args.cases_dir)
    if not cases:
        log.error("no */evaluation.json under %s", args.cases_dir)
        return 2
    print(format_table({name: metrics_from_labels(lbl) for name, lbl in cases.items()}))
    return 0


def _cmd_golden(args: argparse.Namespace) -> int:
    from person_search.eval.golden import GoldenCase, compare
    from person_search.factory import build_components
    from person_search.pipeline import run
    from person_search.types import OSNET

    settings = load_settings(args.config)
    components = build_components(settings, encoders=[OSNET])
    failures = 0
    for golden_path in sorted(Path(args.golden_dir).glob("test_case_*.json")):
        golden = GoldenCase.load(golden_path)
        case_dir = Path(args.cases_dir) / golden.case
        out = run(case_dir / "test_video.mp4", case_dir / "query.jpg", settings, components)
        diff = compare(golden, out.result.selected_samples)
        failures += not diff.ok
        status = "PASS" if diff.ok else "FAIL"
        print(f"{golden.case}: {status} ({diff}) [{out.ingest.wall_seconds:.0f}s]")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="person-search",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=None, help="YAML profile (default: $PS_CONFIG_FILE)"
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--log-json", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="print resolved settings").set_defaults(func=_cmd_config)

    si = sub.add_parser("ingest", help="ingest one video and save observations")
    si.add_argument("--video", type=Path, required=True)
    si.add_argument("--out", type=Path, required=True, help="output .npz path")
    si.add_argument("--encoders", nargs="+", default=["osnet", "irra"], choices=["osnet", "irra"])
    si.set_defaults(func=_cmd_ingest)

    sp = sub.add_parser("photo", help="photo query on one video")
    sp.add_argument("--video", type=Path, required=True)
    sp.add_argument("--photo", type=Path, required=True)
    sp.add_argument("--out", type=Path, default=None)
    sp.add_argument(
        "--render", action="store_true", help="write output_annotated.mp4 and output.mp4"
    )
    sp.set_defaults(func=_cmd_photo)

    st = sub.add_parser("text", help="text query on one video")
    st.add_argument("--video", type=Path, required=True)
    st.add_argument("--query", required=True)
    st.add_argument("--top-k", type=int, default=5)
    st.set_defaults(func=_cmd_text)

    se = sub.add_parser("eval-frames", help="frame-level metrics from evaluation.json files")
    se.add_argument("--cases-dir", type=Path, required=True)
    se.set_defaults(func=_cmd_eval_frames)

    sg = sub.add_parser("golden", help="compare pipeline output with golden frames")
    sg.add_argument("--cases-dir", type=Path, required=True)
    sg.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("tests/golden/data") / f"{sys.platform}-{platform.machine().lower()}",
        help="golden data for this platform (default: tests/golden/data/<platform>)",
    )
    sg.set_defaults(func=_cmd_golden)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_json)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
