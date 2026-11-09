# 0009 - Text returns ranked tracks; hybrid uses late fusion

- **Status:** accepted
- **Date:** 2026-11-09

## Context

Photo queries keep the paper's rule: a dynamic threshold accepts tracks and the
system returns frames. Text and hybrid queries need their own design:

- IRRA's text-to-image similarities are low and compressed (top tracks score
  0.15-0.27; the gap between the target and the next person is often 0.01).
  An absolute threshold tuned on 12 queries would be tuned on the test set.
- Repo 1 built hybrid queries by adding a CLIP-ReID image embedding to an IRRA
  text embedding and comparing the sum with IRRA image vectors (AUDIT.md F2):
  two different embedding spaces.

## Decision

- **Text queries return the top 5 tracks**, ranked by the paper's top-50% track
  score, with no threshold. A person reviews them. (`search/text.py`,
  `IndexedSearch.text`.)
- **Hybrid queries use late fusion**: score every detection in each model's own
  space (OSNet for the photo, IRRA for the text), min-max normalise each set
  over the searched scope, mix as `alpha * photo + (1 - alpha) * text`, then
  rank tracks by top-50% score. Default `alpha = 0.5`, chosen before any
  result and not tuned on the 12 queries.
- Repo 1's embedding sum stays in `core/fusion.legacy_embedding_sum` only to be
  measured against.

## Results (12 descriptions, 4 videos; docs/results/week2.md)

- Text alone: Hit@1 8/12, Hit@5 11/12. Detailed descriptions 4/4 at Hit@1,
  short ones 1/4, so the more a witness says, the better it works.
- Hybrid, late fusion: 12/12 at alpha 0.5 and 0.7; 10/12 at 0.3.
- Repo 1's embedding sum: also 12/12. **On this test set the fix cannot be
  told apart from the bug**, because the photo alone already finds all 4
  targets (OSNet track ranking 4/4). The mixed-space sum still works here
  because CLIP-ReID and IRRA start from the same CLIP weights: their image
  embeddings of the same photo agree weakly (cosine 0.22, against 0.10 for
  different people), so the sum carries a weak identity signal plus the text.

## Consequences

- Late fusion is kept because it is correct by construction (each score is
  measured in the space it was trained for) and lets alpha mean something, not
  because it won on this data. Do not claim it raised Hit@1.
- To show what hybrid adds, the test set needs queries where the photo alone
  fails (blurred, tiny or occluded photos, or a different camera) and a gallery
  larger than one video. That is future work, and its queries must again be
  written before results are seen.
- A text threshold (for "no match" answers) needs calibration data the project
  does not have yet; until then text never answers "no match".
