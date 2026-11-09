# models/

Weights live here locally and are git-ignored. `make weights` fills this folder.
Which versions exist, their SHA-256 and their gate scores are recorded in
`configs/models.yaml` and, after `make register`, in the MLflow registry
(`person-search-osnet`, `person-search-irra`; alias `production`).

| File | Version | Source | Notes |
|---|---|---|---|
| `yolo11n_person.pt` | - | team Drive (`yolo_finetuned_best.pt`), same bytes as repo 1 | YOLO11n, saved with ultralytics 8.4.22. AGPL-3.0 |
| `osnet_campus_v1.pth.tar` | osnet-v1 | team Drive (`osnet_finetuned_model.pth.tar-60`) | 70 campus IDs, epoch 60. Production (photo) |
| `osnet_market_v0.pth` | osnet-v0 | team Drive (`osnet_x1_0_market_256x128_...pth`) | Market-1501 only; the gate must block it |
| `irra_v1.pth` | irra-v1 | Hugging Face `prasish17/clip-text-query` (`best_model_text.pth`) | ViT-B/16, 256x128, 8 prompt tokens, FP32. Production (text) |
| `clip_vit_b16_v0.pt` | irra-v0 | OpenAI CLIP ViT-B/16 | vanilla CLIP, 224x224, no prompt tokens; the gate must block it |
| `irra_v1_image_fp32.onnx` | - | `scripts/export_onnx.py` from `irra_v1.pth` | IRRA image tower in ONNX FP32 (parity check against PyTorch) |
| `irra_v2_image_int8.onnx` | irra-v2 (image) | `scripts/export_onnx.py` (dynamic INT8) | candidate: 3.2x faster, passes the gate (decision 0011); text stays `irra_v1.pth` |
| `clipreid_image_legacy.pth` | - | Hugging Face `prasish17/clip-image-query` (`best_model_image.pth`) | repo 1's CLIP-ReID image model; only for the Day 12 comparison with repo 1's hybrid |

`scripts/fetch_weights.py` prints the SHA-256 of each file; they must match
`configs/models.yaml`.
