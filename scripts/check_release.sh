#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
bad=0

if grep -RIniE --exclude-dir=.git --exclude='check_release.sh' --exclude='*.ckpt' 'g[v]hmr' .; then
  echo "Legacy project label detected." >&2
  bad=1
fi

if grep -RInP --exclude-dir=.git --exclude='*.ckpt' '\p{Han}' .; then
  echo "Chinese text detected." >&2
  bad=1
fi

if grep -RIniE --exclude-dir=.git --exclude='check_release.sh' --exclude='*.ckpt'   'a[8]00|cam[_]static|cam[s]tatic' .; then
  echo "Stale hardware or unused-condition label detected." >&2
  bad=1
fi

if grep -RInE --exclude-dir=.git --exclude='check_release.sh' --exclude='*.ckpt'   'wandb_v1_|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY' .; then
  echo "Potential secret detected." >&2
  bad=1
fi

if grep -RInE --exclude-dir=.git --exclude='check_release.sh' --exclude='*.ckpt'   '(/ceph/home/|/mnt/ddn/|/home/[A-Za-z0-9._-]+/)' .; then
  echo "Machine- or user-specific absolute path detected." >&2
  bad=1
fi

local_checkpoint="$(find checkpoints -type f -name '*.ckpt' -print -quit 2>/dev/null || true)"
if [[ -n "$local_checkpoint" ]] && ! git check-ignore -q "$local_checkpoint"; then
  echo "Local checkpoint is not ignored by Git." >&2
  bad=1
fi

large_files="$(find . -path './.git' -prune -o -path './checkpoints' -prune -o -path './inputs/checkpoints' -prune -o -type f -size +95M -print)"
if [[ -n "$large_files" ]]; then
  echo "Unexpected files larger than 95 MiB:" >&2
  echo "$large_files" >&2
  bad=1
fi

symlinks="$(find . -path './.git' -prune -o -type l -print)"
if [[ -n "$symlinks" ]]; then
  echo "Symlinks are not allowed in the public release:" >&2
  echo "$symlinks" >&2
  bad=1
fi

leftovers="$(find . -path './.git' -prune -o \(   -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.swp' -o -name '*.bak*' -o -name '*.orig' -o -name '*.rej' -o -name '*before*' \)   -o -type d -name __pycache__ \) -print)"
if [[ -n "$leftovers" ]]; then
  echo "Cache or backup files detected:" >&2
  echo "$leftovers" >&2
  bad=1
fi

bash -n scripts/*.sh slurm/*.sbatch
"$PYTHON_BIN" -m compileall -q hmr4d motion_lib tools canonicalize_motion.py roma.py
find . -type d -name __pycache__ -prune -exec rm -r -- {} +
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

if [[ "$bad" -ne 0 ]]; then
  exit 1
fi
echo "Release checks passed."
