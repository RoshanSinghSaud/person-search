"""Week 3 step 1: export IRRA's image tower to ONNX and quantize it to INT8.

Writes, next to the PyTorch checkpoint:

* ``irra_v1_image_fp32.onnx`` - the image tower as is (ONNX opset 17, dynamic
  batch); must match PyTorch to cosine > 0.9999 (``make parity``);
* ``irra_v2_image_int8.onnx`` - the same graph with INT8 MatMul/Gemm weights
  (ONNX Runtime dynamic quantization, per channel). This is the irra-v2
  candidate in ``configs/models.yaml``; it ships only if the gate passes.

Prints the size and SHA-256 of each file (record them in configs/models.yaml).

Usage::

    python scripts/export_onnx.py --weights models/irra_v1.pth --out-dir models
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from person_search.models.irra_encoder import IRRAEncoder
from person_search.models.onnx_image import OPSET, export_image_tower, quantize_int8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--weights", type=Path, default=Path("models/irra_v1.pth"))
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    enc = IRRAEncoder(args.weights, version="irra-v1")
    meta = {
        "source_checkpoint": args.weights.name,
        "source_sha256": sha256(args.weights),
        "image_size": f"{enc.image_size[0]}x{enc.image_size[1]}",
        "opset": str(OPSET),
    }
    fp32 = export_image_tower(
        enc.model.visual, enc.image_size, args.out_dir / "irra_v1_image_fp32.onnx", metadata=meta
    )
    int8 = quantize_int8(fp32, args.out_dir / "irra_v2_image_int8.onnx")
    out = {
        p.name: {"mb": round(p.stat().st_size / 1e6, 1), "sha256": sha256(p)} for p in (fp32, int8)
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
