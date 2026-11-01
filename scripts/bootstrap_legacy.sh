#!/usr/bin/env bash
# Clone both legacy repos into a folder OUTSIDE this repo (default ../legacy).
# They are reference material: never edit them. Pinned to the commits audited
# in AUDIT.md so parity/golden results stay reproducible.
set -euo pipefail

DEST="${1:-../legacy}"
mkdir -p "$DEST"

clone() {  # url dir commit
  if [[ -d "$DEST/$2/.git" ]]; then
    echo "exists: $DEST/$2"
  else
    git clone --filter=blob:none "$1" "$DEST/$2"
  fi
  git -C "$DEST/$2" checkout --quiet "$3"
  echo "$2 @ $(git -C "$DEST/$2" rev-parse --short HEAD)"
}

clone https://github.com/meteroid27/person-reid-clip.git person-reid-clip \
  5d8768aa2af6875cebde75f042a5dd6714d81006
clone https://github.com/Amritkandel49/Person-Based-Video-Segmentation-Using-Reidentification-And-Tracking.git \
  amrit-pbvs 51534374acb74984e358a41d1389674e97f0e0a2

# repo 2 expects its weights under models_path/ (only needed to run it untouched)
mkdir -p "$DEST/amrit-pbvs/models_path"
echo "done. Weights: make weights; Day 2 reproduction: make reproduce"
