# 0003 - Fail loudly on model loading

- **Status:** accepted
- **Date:** 2026-11-01

## Context

AUDIT.md F7: repo 2 replaces a failed OSNet load with an ImageNet ResNet50;
repo 1 runs base CLIP when no IRRA keys match, retries failed crops at 224x224,
and returns `None` from extractors. In every case the system keeps answering,
with wrong results.

## Decision

- `models/checkpoint.load_strict` requires the checkpoint's keys and shapes to
  match the architecture exactly. Allowed exceptions are explicit: OSNet's
  training-only `classifier.*`, IRRA's training heads (`cross_attn`, `mlm_head`,
  `*_bn`, `*_classifier`), and CLIP's unused `logit_scale`.
- Anything else raises `CheckpointMismatchError` listing missing, unexpected and
  mis-shaped keys. Missing files raise `ModelLoadError`; YOLO raises before
  Ultralytics can try to download a file of the same name.
- The IRRA model is built from the checkpoint's own shapes. The checkpoint holds
  every CLIP tensor except `logit_scale`, so the 335 MB OpenAI download is not
  needed and cannot silently leak base weights into the model.
- Encoders raise on bad input instead of returning `None`.

## Consequences

- A wrong file in the registry stops the service at start-up (and `/health`
  on Day 17), not after a day of bad search results.
- Loading vanilla CLIP as "irra v0" works through the same path (TorchScript
  archives are read too) because it is a complete CLIP state dict.
