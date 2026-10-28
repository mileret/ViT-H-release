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
|   |-- samples/
|   |   `-- <session>.pkl
|   |-- manifests/
|   |   `-- <session>.json
|   |-- index.json
|   `-- conversion_report.json
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

Convert it to the layout consumed by `BonesSeedG1SomaDataset`:

```bash
python tools/prepare_bones_seed.py \
  --source /path/to/bones-seed \
  --output /path/to/training_set/bones_seed \
  --workers 4
```

Every converted sequence contains paired `g1` and `soma` entries. G1 arrays
include root pose, 29 BYD-ordered joint angles, and 30-body forward-kinematics
results. SOMA arrays contain root orientation and the 28 joints expected by the
training loader. Both sides are aligned and sampled at 30 FPS. The converter
also writes `index.json`; no separate indexing step is required.

To copy the extracted dataset to node-local storage without modifying the
source:

```bash
export VITH_DATASET_ROOT=/path/to/training_set
bash scripts/stage_training_set.sh /path/to/local/training_set
```
