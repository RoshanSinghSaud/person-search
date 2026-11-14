# Week 4, step 5: model gate in CI

Workflow `.github/workflows/model-gate.yml` (decision 0016) on GitHub's hosted
runner (`ubuntu-latest`, 4 vCPU, `PS_TORCH_THREADS=4`). Both versions are scored
in the same job, every crop re-embedded, on the private gate set (decision 0010).
Numbers copied from the verdicts the workflow posted on each pull request.

| Pull request | Pins production to | Verdict | Deciding numbers | Scoring: production / candidate | Whole job |
|---|---|---|---|---:|---:|
| [#7](https://github.com/RoshanSinghSaud/person-search/pull/7) | `irra-v0` (vanilla CLIP) | **blocked** | image tower campus mAP 78.01% -> 12.82% (-65.18 pts); text Hit@1 8/12 -> 6/12, 4 of production's Hit@1 queries lost | 613 s / 1,031 s | 29.6 min |
| [#9](https://github.com/RoshanSinghSaud/person-search/pull/9) | `osnet-v0` (Market-1501) | **blocked** | campus mAP 76.41% -> 29.46% (-46.95 pts) | 111 s / 116 s | 5.7 min |
| [#10](https://github.com/RoshanSinghSaud/person-search/pull/10) | `irra-v2` (INT8 image tower) | **passed** | image tower campus mAP 78.01% -> 77.45% (-0.56 pts); text Hit@1 8/12, no query lost; MRR 0.771 -> 0.785 | 577 s / 284 s | 16.5 min |
| [#6](https://github.com/RoshanSinghSaud/person-search/pull/6), [#8](https://github.com/RoshanSinghSaud/person-search/pull/8) | no change | passed | nothing to score | | 0.2 min |

- **Same numbers as Weeks 2-3** (docs/results/week2.md, week3_onnx.md) to two
  decimals, measured there on another machine: the gate set reproduces across
  machines, as decision 0010 requires.
- **Photo F1 alone would have let the Market-1501 OSNet through** (84.70% ->
  85.25%, AUDIT.md F26); campus mAP is what blocks it.
- **INT8 on the CI runner:** scoring the gate set took 284 s with the INT8 image
  tower against 577 s in FP32 (2.0x, campus split + 4 videos + 12 queries).
- **#10 passed and was not merged.** Passing the gate does not promote a model:
  the service still runs `irra-v1`, and `irra-v2` needs an index built with it
  first (decision 0011). Merging the pin alone would make the next gate compare
  candidates against a model the service does not run.
- #7's first run also failed the unit tests: the gate's own tests read the live
  production pins. Fixed in #8 (the tests fix their own pins).
