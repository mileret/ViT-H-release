#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${VITH_DATASET_ROOT:?Set VITH_DATASET_ROOT to the extracted dataset directory}"
DEST_ROOT="${1:-/tmp/$USER/vith-training-set}"

for directory in amass bedlam bedlam2 h36m bones_seed/paired metadata; do
  [[ -d "$SOURCE_ROOT/$directory" ]] || {
    echo "Missing dataset directory: $SOURCE_ROOT/$directory" >&2
    exit 2
  }
done

mkdir -p "$DEST_ROOT"
rsync -a --whole-file --prune-empty-dirs \
  --exclude='/bones_seed/soma/***' \
  --exclude='/bones_seed/soma_filtered/***' \
  --exclude='/bones_seed/robot/***' \
  --exclude='/bones_seed/robot_filtered/***' \
  --include='*/' \
  --include='*.pth' \
  --include='*.pt' \
  --include='*.pkl' \
  --include='*.json' \
  --include='*.npz' \
  --exclude='*' \
  "$SOURCE_ROOT/" "$DEST_ROOT/"

echo "Staged data at: $DEST_ROOT"
echo "export VITH_DATASET_ROOT=$DEST_ROOT"
