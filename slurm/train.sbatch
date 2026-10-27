#!/usr/bin/env bash
#SBATCH --job-name=vith-vit-h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=480G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Add site-specific partition, account, or QoS directives above when needed.
# Export dataset roots before sbatch, or source a private environment file here.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export NUM_DEVICES="${NUM_DEVICES:-8}"
MASTER_PORT="$("$PYTHON_BIN" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
export MASTER_PORT

exec bash scripts/train.sh
