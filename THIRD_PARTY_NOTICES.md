# Third-party notices

## Vendored code

### torchreid - `src/person_search/models/osnet.py`

From https://github.com/KaiyangZhou/deep-person-reid (`torchreid/models/osnet.py`,
identical to `torchreid==0.2.5` on PyPI), kept verbatim.

```
MIT License

Copyright (c) 2018 Kaiyang Zhou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Zhou, Yang, Cavallaro, Xiang. *Omni-Scale Feature Learning for Person Re-Identification.* ICCV 2019.

## Ported logic (rewritten, not copied)

| From | What | License |
|---|---|---|
| repo 1 `meteroid27/person-reid-clip` (`clip_inference.py`, `reid_main.py`) | IRRA loading, prompt-token text path, ingest loop | MIT |
| repo 2 `Amritkandel49/Person-Based-Video-Segmentation-...` (`pole/reid_main.py`, `utilities/`) | OSNet extractor, track scoring, dynamic threshold, annotated video, frame metrics | no license file; the team approved republishing this code (2026-11-02) |
| torchreid `metrics/rank.py` | Market-1501 mAP/CMC protocol (reimplemented; a unit test compares against it) | MIT |

## Runtime dependencies with copyleft licenses

| Package | License | Effect |
|---|---|---|
| `ultralytics` (YOLO11) | AGPL-3.0 | Serving it over a network obliges you to offer the source of the service. Fine for a public portfolio repo; matters for any closed deployment. |
| `deep-sort-realtime` | GPL-3.0 | Distributing the combined program requires GPL-compatible terms. |
| `imageio-ffmpeg` (API result clips) | BSD-2-Clause; bundles an FFmpeg binary built with libx264 (GPL-2.0+) | Called as a separate program to encode H.264 clips; the Docker image ships that binary, so its license applies to the image. |
| OpenAI CLIP | MIT | - |
| IRRA (method) | - | Jiang & Ye, *Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval*, CVPR 2023. |

Because of the two copyleft dependencies, choose this repository's own license
(AGPL-3.0 is the simplest consistent choice) together with the team before
making it public.
