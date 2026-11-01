# models/

Weights live here locally and are git-ignored. `make weights` fills this folder;
from Day 8 the MLflow registry on DagsHub is the source of truth.

| File | Registry name | Source | Notes |
|---|---|---|---|
| `yolo11n_person.pt` | - | team Drive (`yolo_finetuned_best.pt`), same bytes as repo 1 | YOLO11n, saved with ultralytics 8.4.22. AGPL-3.0 |
| `osnet_campus_v1.pth.tar` | osnet v1 | team Drive (`osnet_finetuned_model.pth.tar-60`) | 70 campus IDs, epoch 60, Rank-1 81.7% |
| `osnet_market_v0.pth` | osnet v0 | team Drive (`osnet_x1_0_market_256x128_...pth`) | Market-1501 only; the gate must block it |
| `irra_v1.pth` | irra v1 | Hugging Face `prasish17/clip-text-query` (`best_model_text.pth`) | ViT-B/16, 256x128, 8 prompt tokens, FP32 |

SHA-256 of each file is printed by `scripts/fetch_weights.py`; record them in
the registry when you register the models.
