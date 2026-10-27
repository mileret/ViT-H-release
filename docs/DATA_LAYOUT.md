# Dataset layout

The Hugging Face dataset is distributed as five `.tar.zst` archives. Extract
all archives into one directory and set `VITH_DATASET_ROOT` to that directory.

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
|   `-- index.json
`-- metadata/
    |-- normalization_stats.npz
    `-- camera_stats.npz
```

Each motion part contains `motion.pth` and `g1.pth`. Video-derived parts
also contain bounding boxes, 2D keypoints, and image features. The numbered
parts preserve the full training composition.

Extract the archives:

```bash
for archive in amass bedlam bedlam2 h36m bones_seed; do
  tar --use-compress-program=unzstd -xf "${archive}.tar.zst"
done
```

To copy the extracted dataset to node-local storage without modifying the
source:

```bash
export VITH_DATASET_ROOT=/path/to/training_set
bash scripts/stage_training_set.sh /path/to/local/training_set
```
