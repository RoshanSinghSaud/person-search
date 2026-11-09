# 0007 - Text test set: witness descriptions first, targets labelled by eye

- **Status:** accepted
- **Date:** 2026-11-09
- **Supersedes:** the last bullet of 0006 ("text targets come from the golden track IDs")

## Context

Nobody had measured text search on video (AUDIT.md F3). Model 2's numbers are
Rank-1 on cropped images. To measure it end to end we need text queries and,
for each, the tracks that count as a correct answer. Two traps:

1. **Leakage.** A description written after seeing which tracks the text
   search ranks first will drift towards whatever the model already finds.
2. **Wrong targets.** The golden track IDs are what *photo* search accepted,
   and photo search is sometimes wrong: in case 3 its second accepted track
   (162, score 0.657) is a different man; in case 4 its second track (42) is a
   different person. The frame labels cannot pick the target either: they mark
   frames, not boxes, and the scenes are crowded.

## Decision

- **Queries** (`tests/golden/data/text_queries.yaml`): 3 descriptions per case
  (short, medium, detailed), written from `query.jpg` only, committed on their
  own before any text search ran (`git log -- tests/golden/data/text_queries.yaml`).
  They are never edited after results are seen; a new description gets a new ID
  and says so in the commit.
- **Targets** (`tests/golden/data/text_targets.yaml`): for each case, the tracks
  whose box is on the target, as **sample ranges** per track, read off crop
  strips of every track (`tools/analysis/track_contact_sheet.py`) and committed
  before any text search ran, in a commit of their own. Ranges are needed because
  DeepSORT switches identities: in case 3 the target is track 4 for samples
  2-160, then track 52, then track 96. The ranges agree with the TP + FN frames
  of each `evaluation.json`.
- **When a ranked track counts as the target:** when more than half of the
  detections that produced its score (its top 50%) show the target. That is what
  a reviewer sees first when they open the track. A stricter "more than half of
  all its detections" is reported alongside (`Hit@k (pure)`).
- Metrics: Hit@1, Hit@5, MRR and frame recall of the top track
  (`eval/text_e2e.py`).

## Consequences

- 12 queries is a small test set: one query is 8 points of Hit@1. Report counts
  (7/12) next to percentages, and do not tune anything (alpha, thresholds) on it.
- Case 1's query crop is 29x77 px. The descriptions say "white top, cap"; the
  person actually wears a red-and-white colour-block jacket and a white face
  mask. That is a realistic witness error and stays in the set.
- The descriptions and target labels were written by the assistant, not by an
  independent person. Spot-check the contact sheets in
  `../legacy/runs/text_targets/` and add your own descriptions as new IDs.
