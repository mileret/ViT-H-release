# Dataset layout

Four preprocessed sources are distributed from the project Hugging Face
dataset as `.tar.zst` archives. BONES-SEED is downloaded from its official
repository and converted separately. Place all five sources under one training
directory and set `VITH_DATASET_ROOT` to that directory.

```text
training_set/
|-- amass/
|   |-- part_00/
|   `-- part_01/
|-- bedlam/
|   |-- part_00/
|   `-- part_01/
|-- bedlam2/
|   |-- part_00/
|   |-- part_01/
|   `-- part_02/
|-- h36m/
|   |-- part_00/
|   `-- part_01/
|-- bones_seed/
|   |-- soma/                 # 30 FPS SOMA intermediates
|   |-- soma_filtered/        # selected SOMA intermediates
|   |-- robot/                # 30 FPS G1 intermediates
|   |-- robot_filtered/       # selected G1 intermediates
|   `-- paired/
|       |-- <session>.pkl
|       |-- bones_seed_index_v1.json
|       `-- export_errors.log
`-- metadata/
    |-- normalization_stats.npz
    `-- camera_stats.npz
```

Each motion part contains `motion.pth` and `g1.pth`. Video-derived parts
also contain bounding boxes, 2D keypoints, and image features. The numbered
parts preserve the published four-source portion of the training composition.

Extract the archives:

```bash
for archive in amass bedlam bedlam2 h36m; do
  tar --use-compress-program=unzstd -xf "${archive}.tar.zst"
done
```

The official BONES-SEED download has matched proportional SOMA BVH and G1 CSV
files at 120 FPS:

```text
bones-seed/
|-- soma_proportional/bvh/<session>/<motion>.bvh
`-- g1/csv/<session>/<motion>.csv
```

Create a dedicated preprocessing environment and convert the data to the layout
consumed by `BonesSeedG1SomaDataset`:

```bash
conda create -n bones-seed-preprocess python=3.10 -y
conda activate bones-seed-preprocess
pip install -r tools/bones_seed_preprocessing/requirements.txt

bash tools/bones_seed_preprocessing/run_preprocessing.sh \
  /path/to/bones-seed \
  /path/to/training_set/bones_seed \
  16
```

Every converted sequence contains paired `g1` and `soma` entries. G1 arrays
include root pose, 29 BYD-ordered joint angles, and 30-body forward-kinematics
results. SOMA arrays contain root orientation and the 28 joints expected by the
training loader. Both sides are aligned and sampled at 30 FPS. The final
exporter writes `paired/bones_seed_index_v1.json`; no separate indexing step is
required. Only `paired/` is needed for training after a successful conversion.
The preceding directories are reproducible intermediate products.

To copy the extracted dataset to node-local storage without modifying the
source:

```bash
export VITH_DATASET_ROOT=/path/to/training_set
bash scripts/stage_training_set.sh /path/to/local/training_set
```
