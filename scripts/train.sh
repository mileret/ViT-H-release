#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE="${WANDB_MODE:-offline}"
NUM_DEVICES="${NUM_DEVICES:-8}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-ddp_find_unused_parameters_true}"

export WANDB_MODE
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

required_vars=(VITH_DATASET_ROOT)
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 2
  fi
done
logger_args=("logger.offline=true")
if [[ "$WANDB_MODE" == "online" ]]; then
  [[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required for online logging" >&2; exit 2; }
  [[ -n "${WANDB_ENTITY:-}" ]] || { echo "WANDB_ENTITY is required for online logging" >&2; exit 2; }
  logger_args=("logger.offline=false" "+logger.entity=$WANDB_ENTITY")
fi

exec "$PYTHON_BIN" -u tools/train.py   exp=offline/g1_training_set_exp1   exp_name=g1_training_set_exp1   'train_datasets.amass_00.split_ratios=[1.0,0.0,0.0]'   'train_datasets.amass_01.split_ratios=[1.0,0.0,0.0]'   'train_datasets.bedlam2_00.split_ratios=[1.0,0.0,0.0]'   'train_datasets.bedlam2_01.split_ratios=[1.0,0.0,0.0]'   'train_datasets.bedlam2_02.split_ratios=[1.0,0.0,0.0]'   'train_datasets.bedlam_00.split_ratios=[1.0,0.0,0.0]'   'train_datasets.bedlam_01.split_ratios=[1.0,0.0,0.0]'   'train_datasets.h36m_00.split_ratios=[1.0,0.0,0.0]'   'train_datasets.h36m_01.split_ratios=[1.0,0.0,0.0]'   'train_datasets.bones_seed.split_ratios=[1.0,0.0,0.0]'   train_datasets.bones_seed.dof_pos_order=byd   ckpt_path=null   resume_mode=null   pl_trainer.devices="$NUM_DEVICES"   pl_trainer.strategy="$TRAIN_STRATEGY"   pl_trainer.max_epochs=10000   pl_trainer.log_every_n_steps=1   pl_trainer.use_distributed_sampler=false   data.loader_opts.train.batch_size=32   data.loader_opts.train.num_workers=1   data.loader_opts.train.source_homogeneous_batches=false   +data.loader_opts.train.source_mixed_batches=true   "${logger_args[@]}"   "$@"
