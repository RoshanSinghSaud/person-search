# 0011 - IRRA's image tower runs as ONNX INT8 (irra-v2), text stays in PyTorch

- **Status:** accepted (candidate passes the gate, also in CI: decision 0016; promotion waits for an index built with irra-v2)
- **Date:** 2026-11-09

## Context

IRRA's image encoder was half of all ingest time in Week 2 (51% on test case
2), and cProfile put most of it in the ViT's linear layers. Text queries are one
short forward pass each, so they are not worth optimising.

## Decision

- **Only the image tower moves.** `scripts/export_onnx.py` exports CLIP's
  `VisionTransformer` (IRRA's image tower) to ONNX, opset 17, dynamic batch
  axis. The text tower, with its learned prompt tokens, stays in PyTorch.
- **Dynamic INT8 quantization** (ONNX Runtime `quantize_dynamic`): the weights
  of every MatMul/Gemm become INT8, per output channel; activations are
  quantized on the fly; the patch convolution, layer norms and softmax stay
  FP32. Chosen before looking at any result, because it needs no calibration
  images (so the gate set stays a clean test set) and the attention/MLP matmuls
  are where the time goes. `reduce_range` is off: the CPUs we run on have VNNI.
- **Same preprocessing and normalisation as the PyTorch encoder**, so the ONNX
  backend is a drop-in: `PS_IRRA__IMAGE_ONNX=models/irra_v2_image_int8.onnx`.
- The INT8 tower is model version **irra-v2** in `configs/models.yaml`, scored
  by the same gate as every version.

## Results (docs/results/week3_onnx.md; 2 vCPU, 2 threads)

- ONNX FP32 matches PyTorch (cosine > 0.9999, identical top-10) but is not
  faster (126 vs 125 ms per crop).
- **ONNX INT8: 3.2x faster** (40 ms per crop at batch 32; 45 ms at batch 1),
  **3.9x smaller** (89 MB vs 345 MB).
- **Gate: passed.** Image tower campus mAP 77.45% vs 78.01% (-0.56 points, limit
  1); text Hit@1 8/12 and Hit@5 11/12 unchanged, same top track for all 12
  descriptions; MRR 0.785 vs 0.771.
- Ingest of 240 frames of case 2: 201 s vs 266 s (**1.32x faster end to end**).

## Consequences

- Promoting irra-v2 needs a new index: the vectors change, and the collection
  name carries the version (`detections__irra-v2__osnet-v1`). Step 5 does the
  promotion through the gate PR and rebuilds the index.
- INT8 speed depends on the CPU. Without VNNI (older Xeons, some cloud CPUs)
  the gain is smaller; re-run `make benchmark` on the deployment CPU.
- OSNet is now the slowest stage, and it gets slower with 2 threads (AUDIT.md
  F28). Exporting it to ONNX, cut from the fast plan, is now the next speed-up.
