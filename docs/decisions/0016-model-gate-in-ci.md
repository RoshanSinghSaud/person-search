# 0016 - A pull request that changes a production model must pass the gate in CI

- **Status:** accepted
- **Date:** 2026-11-12

## Context

Step 5 of the fast plan (Day 24). Decision 0010 defined the gate set (private,
on the Hub) and the rule; `eval_models.py compare` applied it by hand. Nothing
stopped a pull request from pinning a worse model in `configs/models.yaml`.
Constraints:

- the gate set and two of the weights are private; GitHub's free runners have
  4 vCPUs and no GPU, and re-embedding every crop with the FP32 IRRA tower takes
  minutes, not seconds;
- most pull requests do not touch a model, and a required check that only runs
  on some of them leaves the others waiting for it forever;
- the weights come from four places (the assets repo, the IRRA team's repo,
  OpenAI's CDN, the team Drive), and the Drive is not reachable from CI.

## Decision

- **`.github/workflows/model-gate.yml` runs on every pull request.** Its first
  step (`scripts/model_gate.py plan`, seconds, no ML dependencies) compares the
  PR's `configs/models.yaml` with the base branch's. No `production:` pin changed:
  the check passes at once, so it can be required on `main`. A pin changed: one
  pair per family, production -> candidate.
- **Versions are immutable.** Changing an existing version's file, ONNX file or
  hash in place fails the check: new weights are a new version. Version names are
  validated, so a PR cannot inject shell through them.
- **Both sides are scored in the same job**, on the same runner and thread count,
  with every crop re-embedded (`eval_models.py evaluate`, unchanged), so the
  comparison never mixes machines. Production is scored from the base branch's
  file, the candidate from the PR's.
- **Weights are fetched by name and checked by SHA-256**: from `configs/assets.yaml`
  (the service's files, pinned commit), else OpenAI's URL (vanilla CLIP), else
  `<assets repo>/weights/<file>` (gate-only files such as the Market-1501 OSNet,
  latest commit; the hash makes the commit irrelevant). The gate set is downloaded
  and verified against its manifest. One read-only Hugging Face token
  (`HF_READ_TOKEN`, or the deploy's `HF_SPACE_TOKEN`) reads both private repos.
- **The rule is decision 0010's, unchanged** (`compare_versions`): campus mAP may
  drop at most 1 point; OSNet photo frame F1 at most 1 point; IRRA may not lose
  any query production ranks first. The verdict is a table in the job summary and
  a PR comment (edited in place on re-runs); the scores are kept as an artifact.

## Results (docs/results/week4_gate.md)

Checked before the first CI run, on the scores of Weeks 2-3 (same code):

| Candidate | vs | Verdict | Why |
|---|---|---|---|
| irra-v0 (vanilla CLIP) | irra-v1 | blocked | campus mAP 78.01% -> 12.82%; Hit@1 lost on 4 queries (8/12 -> 6/12) |
| osnet-v0 (Market-1501) | osnet-v1 | blocked | campus mAP 76.41% -> 29.46% (photo F1 alone would have let it through: 84.70% -> 85.25%) |
| irra-v2 (INT8 tower) | irra-v1 | passed | campus mAP -0.56 pts; Hit@1 8/12, no query lost |

In CI, on real pull requests (#7, #9, #10), the verdicts and numbers were the same;
jobs took 5.7-29.6 minutes (docs/results/week4_gate.md).

## Consequences

- A model change costs one CI run (5.7 min for OSNet, 16.5-29.6 min for IRRA);
  other pull requests pay about 10 seconds.
- "Blocks" is literal only when the check is required: *Settings -> Rules ->
  Rulesets -> main -> Require status checks: `model gate`*.
- Passing the gate does not promote a model by itself: `irra-v2` passes, but the
  service also needs an index built with it (decision 0011), so its pull request
  is closed, not merged, until that index exists.
- A change to model *code* under an unchanged version name is not gated (both
  sides would run the PR's code). The parity and golden tests cover that path.
- When the college text-image split arrives, it joins the gate set and the IRRA
  rule (decision 0010); the workflow does not change.
