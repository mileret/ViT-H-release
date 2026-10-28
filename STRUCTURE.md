# Repository structure

```text
ViT-H-release/
|-- README.md
|-- LICENSE
|-- STRUCTURE.md
|-- PROVENANCE.md
|-- requirements.txt
|-- setup.py
|-- pyproject.toml
|-- .env.example
|-- .gitignore
|-- .gitattributes
|-- canonicalize_motion.py
|-- roma.py
|-- assets/
|   `-- teaser.png
|-- data/
|   |-- g1_training_set_exp1_stats.npz
|   |-- g1_training_set_exp1_pred_cam_stats.npz
|   `-- bones_seed_g1_soma_pair_index.json
|-- docs/
|   |-- DATA_LAYOUT.md
|   `-- INFERENCE.md
|-- experiments/
|   `-- g1_training_set_exp1/
|       |-- hydra_config.sanitized.yaml
|       |-- hydra_overrides.yaml
|       `-- SHA256SUMS
|-- hmr4d/
|   |-- configs/
|   |-- datamodule/
|   |-- dataset/
|   |-- model/
|   |-- network/
|   `-- utils/
|-- inputs/
|   `-- README.md
|-- motion_lib/
|-- scripts/
|   |-- train.sh
|   |-- stage_training_set.sh
|   `-- check_release.sh
|-- slurm/
|   `-- train.sbatch
|-- tools/
|   |-- train.py
|   |-- prepare_bones_seed.py
|   |-- infer_video.py
|   |-- inference_utils.py
|   `-- video_preprocess.py
`-- unitree_description/
    `-- mjcf/
        `-- g1.xml
```

## Responsibilities

- `hmr4d/`: model architecture, training datasets, shared utilities,
  and Hydra configuration required by the released experiment and inference.
- `hmr4d/configs/exp/offline/`: the single released experiment
  configuration; its direct inheritance fragments are under
  `hmr4d/configs/base/`.
- `tools/train.py`: Lightning/Hydra training entry point.
- `tools/prepare_bones_seed.py`: converts the official proportional SOMA BVH
  and G1 CSV pairs into the BONES-SEED training shards and index.
- `tools/infer_video.py`: video-to-G1 offline inference entry point.
- `tools/inference_utils.py`: checkpoint loading and output-angle utilities.
- `tools/video_preprocess.py`: tracking, pose, feature, and camera preprocessing.
- `data/`: small metadata required by this experiment; no raw datasets.
- `experiments/`: sanitized configuration emitted by the training run.
- `motion_lib/` and `unitree_description/`: G1 forward kinematics.
- `scripts/` and `slurm/`: data staging, training, and validation commands.
- `inputs/`: instructions for separately licensed preprocessing assets.

## Deliberately excluded

- all training checkpoints;
- raw training datasets and private records;
- experiment outputs and W&B run data;
- Blender binaries/assets and rendering-only scripts;
- batch orchestration and site-specific job scripts;
- diagnostic, benchmark, migration, and historical backup files;
- Python bytecode, caches, editor settings, and local environment files.
