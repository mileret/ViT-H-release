# Provenance

## Experiment identity

- Release name: `g1_training_set_exp1`
- Launch date: 2026-08-06
- Distributed training processes: 8
- Precision: bfloat16 mixed precision
- Per-process batch size: 32
- Global batch size: 256
- Network: 16 layers, latent dimension 1024, 16 attention heads
- Dataset batching: mixed-source, not source-homogeneous
- Initialization: scratch
- Seed: 42

The launch arguments needed to reproduce the released configuration are
preserved in
`experiments/g1_training_set_exp1/hydra_overrides.yaml`. The fully
composed configuration is preserved in `hydra_config.sanitized.yaml`.
Account identifiers and absolute cluster paths were replaced with portable
placeholders.

## Source snapshot limitation

The source directory was not a Git repository. The training process loaded its
modules on 2026-08-06, while a few files in that directory were later edited.
There is consequently no verifiable launch-time commit.

For the main pipeline module, the closest preserved copy is the backup made on
2026-08-12 immediately before later inference work. That copy is used as the
canonical `hmr4d/model/offline/pipeline/offline_pipeline.py` in this
release. Other included files are cleaned copies of the source tree available
when the release was assembled.

The emitted Hydra configuration records the training setup; the source snapshot
is the closest recoverable version, not a claimed bit-for-bit launch checkout.

## BONES-SEED preprocessing

`tools/bones_seed_preprocessing/` contains the preprocessing entry points,
minimal SONIC MotionLib dependencies, G1 model assets, and source manifest used
to produce the paired BONES-SEED representation consumed by training. Upstream
licenses, citation metadata, and third-party notices are preserved inside its
`GR00T-WholeBodyControl/` directory. Raw BONES-SEED motions and generated
training shards are not included.

## Excluded artifacts

- the remainder of the experiment output directory
- all training checkpoints
- W&B binary run files and logs
- private training datasets
- generated videos, Blender assets, and inference workbenches
- caches, bytecode, editor files, and historical backups
