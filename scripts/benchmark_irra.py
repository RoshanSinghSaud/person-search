"""Week 3 step 1: IRRA image tower speed - PyTorch FP32 vs ONNX FP32 vs ONNX INT8.

Times one forward pass of each backend on real person crops (the cached crops
of test case 2 from the gate set) at batch sizes 1 and 32: warm-up first, then
repeated runs, reporting p50/p95 per call and milliseconds per crop. Every
backend uses the same number of CPU threads (``--threads``).

Accuracy is not measured here: the gate does that (scripts/eval_models.py,
version irra-v2).

Usage::

    python scripts/benchmark_irra.py --gate-dir ../legacy/gate_set --threads 2 \
        --out ../legacy/runs/week3/benchmark_irra.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from person_search.eval.gate_set import load_case
from person_search.models.irra_encoder import IRRAEncoder
from person_search.models.onnx_image import OnnxImageSession

REPO = Path(__file__).resolve().parents[1]


def timed(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1000)
    return ms


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gate-dir", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=REPO / "models")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--repeats", type=int, nargs=2, default=[50, 10], help="for batch 1, batch 32")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    crops = load_case(args.gate_dir / "cases" / "test_case_002.npz").crops[:32]
    m = args.models_dir
    torch_enc = IRRAEncoder(m / "irra_v1.pth", batch_size=32)
    backends = {
        "PyTorch FP32": torch_enc,
        "ONNX FP32": OnnxImageSession(
            m / "irra_v1_image_fp32.onnx", threads=args.threads, batch_size=32
        ),
        "ONNX INT8": OnnxImageSession(
            m / "irra_v2_image_int8.onnx", threads=args.threads, batch_size=32
        ),
    }
    visual_params = sum(p.numel() for p in torch_enc.model.visual.parameters())
    sizes_mb = {
        "PyTorch FP32": round(visual_params * 4 / 1e6, 1),
        "ONNX FP32": round((m / "irra_v1_image_fp32.onnx").stat().st_size / 1e6, 1),
        "ONNX INT8": round((m / "irra_v2_image_int8.onnx").stat().st_size / 1e6, 1),
    }
    results: dict = {
        "threads": args.threads,
        "torch_threads_seen": torch.get_num_threads(),
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "backends": {},
    }
    for name, enc in backends.items():
        row = {"size_mb": sizes_mb[name]}
        for bs, reps in zip((1, 32), args.repeats, strict=True):
            batch = crops[:bs]
            ms = timed(lambda enc=enc, batch=batch: enc.encode_images(batch), 3, reps)
            row[f"batch{bs}"] = {
                "p50_ms": round(float(np.percentile(ms, 50)), 1),
                "p95_ms": round(float(np.percentile(ms, 95)), 1),
                "ms_per_crop": round(float(np.percentile(ms, 50)) / bs, 1),
            }
        results["backends"][name] = row
        print(name, row)
    base = results["backends"]["PyTorch FP32"]
    for row in results["backends"].values():
        row["speedup_batch32"] = round(base["batch32"]["p50_ms"] / row["batch32"]["p50_ms"], 2)
        row["speedup_batch1"] = round(base["batch1"]["p50_ms"] / row["batch1"]["p50_ms"], 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
