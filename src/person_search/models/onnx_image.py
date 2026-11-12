"""IRRA's image tower in ONNX Runtime (FP32 export or INT8), Week 3 step 1.

Ingest spends about half its time in IRRA's image encoder (docs/results/week2.md,
Day 13), almost all of it in the ViT's linear layers. This module:

* :func:`export_image_tower` - exports CLIP's ``VisionTransformer`` (IRRA's
  image tower) to ONNX with a dynamic batch axis; the input is the
  preprocessed pixel tensor, the output the 512-d embedding before
  normalisation (normalising stays in NumPy, as in the PyTorch encoder);
* :func:`quantize_int8` - ONNX Runtime *dynamic* quantization: the weights of
  every MatMul/Gemm become INT8 (per output channel); activations are quantized
  on the fly; everything else (patch convolution, layer norms, softmax) stays
  FP32. No calibration data is needed, so the gate set stays untouched;
* :class:`OnnxImageSession` - runs either file with exactly the preprocessing
  of :class:`~person_search.models.irra_encoder.IRRAEncoder`.

Only the image tower moves to ONNX. Text queries are one short forward pass
each and stay in PyTorch, with the learned prompt tokens.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from person_search.errors import ModelLoadError
from person_search.models.preprocessing import (
    CLIP_MEAN,
    CLIP_STD,
    batched,
    build_transform,
    to_batch,
)

log = logging.getLogger(__name__)

INPUT_NAME = "pixels"
OUTPUT_NAME = "embedding"
OPSET = 17


def export_image_tower(
    visual: nn.Module,
    image_size: tuple[int, int],
    path: str | Path,
    *,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Export a CLIP ``VisionTransformer`` to ONNX (FP32, dynamic batch)."""
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    visual = visual.float().eval()
    dummy = torch.zeros(2, 3, *image_size)
    with torch.inference_mode():
        torch.onnx.export(
            visual,
            (dummy,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    model = onnx.load(str(path))
    for k, v in (metadata or {}).items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(model, str(path))
    onnx.checker.check_model(str(path))
    return path


def quantize_int8(fp32_path: str | Path, int8_path: str | Path) -> Path:
    """Dynamic INT8 quantization of the MatMul/Gemm weights (per channel)."""
    import tempfile

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    int8_path = Path(int8_path)
    with tempfile.TemporaryDirectory() as tmp:
        prepped = Path(tmp) / "prepped.onnx"
        # shape inference + graph clean-up, as ONNX Runtime recommends before quantizing
        quant_pre_process(str(fp32_path), str(prepped), skip_symbolic_shape=True)
        quantize_dynamic(
            str(prepped),
            str(int8_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
    return int8_path


class OnnxImageSession:
    """Embeds BGR crops with an exported image tower; output L2-normalised."""

    def __init__(
        self,
        path: str | Path,
        *,
        threads: int | None = None,
        batch_size: int = 16,
    ) -> None:
        import onnxruntime as ort

        self.path = Path(path)
        if not self.path.is_file():
            raise ModelLoadError(f"ONNX model not found: {self.path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = threads or 0  # 0 = ONNX Runtime default (all cores)
        try:
            self.session = ort.InferenceSession(
                str(self.path), opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise ModelLoadError(f"cannot load ONNX model {self.path}: {exc}") from exc
        inp = self.session.get_inputs()[0]
        if inp.name != INPUT_NAME or len(inp.shape) != 4:
            raise ModelLoadError(f"{self.path}: unexpected input {inp.name} {inp.shape}")
        h, w = inp.shape[2], inp.shape[3]
        if not (isinstance(h, int) and isinstance(w, int)):
            raise ModelLoadError(f"{self.path}: image size must be fixed, got {inp.shape}")
        self.image_size = (h, w)
        self.dim = int(self.session.get_outputs()[0].shape[1])
        self.batch_size = batch_size
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.transform = build_transform(self.image_size, CLIP_MEAN, CLIP_STD)
        log.info("ONNX image tower %s: input %sx%s, dim %d", self.path.name, h, w, self.dim)

    def encode_images(self, images: Sequence[np.ndarray], *, normalize: bool = True) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for chunk in batched(images, self.batch_size):
            x = to_batch(chunk, self.transform).numpy()
            (feats,) = self.session.run([OUTPUT_NAME], {INPUT_NAME: x})
            out.append(feats.astype(np.float32, copy=False))
        feats = np.concatenate(out)
        if normalize:  # x / (||x|| + 1e-6), as the PyTorch encoder does
            feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)
        return feats.astype(np.float32, copy=False)
