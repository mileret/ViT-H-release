# Offline video inference

## External files

The trained checkpoint is not distributed in this repository. Store it outside
the repository and pass its path through `--ckpt`.

The released training and inference pipeline uses SAM3D image features. Download
the ViT-H model and preprocessing weights from the
[checkpoint repository](https://huggingface.co/datasets/eerrr1/ViT-H-checkpoints):

```bash
hf download eerrr1/ViT-H-checkpoints \
  vith_offline_g1.ckpt \
  inputs/checkpoints/dpvo/dpvo.pth \
  inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt \
  inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth \
  inputs/checkpoints/yolo/yolov8x.pt \
  --repo-type dataset \
  --local-dir .
```

This creates the following files:

```text
vith_offline_g1.ckpt
inputs/checkpoints/dpvo/dpvo.pth
inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
inputs/checkpoints/yolo/yolov8x.pt
inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
```

The DPVO checkpoint is only used when `--use-dpvo` is passed. The default
SimpleVO path does not require DPVO.

The released `vith_offline_g1.ckpt` uses SAM3D. Its encoder is selected
automatically from the documented release compatibility rule, and inference
does not silently fall back to HMR2. The HMR2 asset is retained only for
custom or experimental checkpoints that explicitly use HMR2 features. A
known SAM3D checkpoint rejects `--imgfeat-encoder hmr2` unless the explicitly
experimental `--allow-encoder-mismatch` option is also supplied; results from
that override are not guaranteed.

Clone [Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body) into
`third-party/Fast-SAM-3D-Body` and follow its setup instructions. Download the
official SAM3D files into that checkout:

```text
third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model.ckpt
third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model_config.yaml
third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

`model.ckpt` contains the SAM3D network weights. `mhr_model.pt` contains the
MHR parametric body model used by SAM3D. Both are required but neither is part
of the ViT-H checkpoint repository.

Create the body-model directory and obtain SMPL and SMPL-X from their official
websites after registration: [SMPL](https://smpl.is.tue.mpg.de/) and
[SMPL-X](https://smpl-x.is.tue.mpg.de/).

```bash
mkdir -p inputs/checkpoints/body_models/{smpl,smplx}
```

```text
inputs/checkpoints/body_models/smpl/SMPL_{GENDER}.pkl
inputs/checkpoints/body_models/smplx/SMPLX_{GENDER}.npz
```

Install [DPVO](https://github.com/princeton-vl/DPVO) only if DPVO camera
estimation is needed:

```bash
mkdir -p third-party
git clone --recursive https://github.com/princeton-vl/DPVO.git third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip \
  -O /tmp/eigen-3.4.0.zip
unzip /tmp/eigen-3.4.0.zip -d third-party/DPVO/thirdparty
pip install ./third-party/DPVO
```

The resulting optional DPVO paths are:

```text
third-party/DPVO/config/default.yaml
inputs/checkpoints/dpvo/dpvo.pth
```

## Command

```bash
python tools/infer_video.py \
  --video /path/to/input.mp4 \
  --ckpt /path/to/vith_offline_g1.ckpt \
  --sam3d-ckpt-path third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model.ckpt \
  --sam3d-mhr-path third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --sam3d-root third-party/Fast-SAM-3D-Body \
  --output outputs/inference/input_g1.npz
```

Architecture defaults are fixed to the released experiment:

- 16 transformer layers;
- latent dimension 1024;
- 16 attention heads;
- 120-frame windows;
- seed 42.

Use `--static-cam` only when the input camera is known to be static. Otherwise
SimpleVO estimates camera motion. `--use-dpvo` selects DPVO explicitly.

Custom checkpoints should record `imgfeat_encoder` in their checkpoint
metadata. If a custom checkpoint has no such metadata and does not use the
documented release filename, pass `--imgfeat-encoder sam3d` or
`--imgfeat-encoder hmr2` explicitly.

## Timeline and cache behavior

Video frames are decoded with PyAV timestamps and resampled to a true 30 FPS
timeline before tracking, pose estimation, feature extraction, and camera
estimation. This supports constant- and variable-frame-rate inputs without
changing the action duration. Empty videos, invalid/non-monotonic timestamps,
and inconsistent derived frame counts fail with an error. The exported NPZ
therefore records the same 30 FPS timeline actually seen by the model.

Each preprocessing directory contains `preprocess/cache_manifest.json`. The
cache identity includes a content hash of the source video, the resampling
version and target FPS, preprocessing settings, encoder choice, checkpoint and
MHR identities, detector/pose weights, camera method, and optional camera
calibration. A missing, incomplete, mismatched, or corrupted manifest causes
the managed cache to be rebuilt.

- `--force-recompute` rebuilds the staged video and all managed preprocessing
  artifacts.
- `--force-recompute-imgfeat` rebuilds only image features and their metadata.
- Neither option removes a user-provided `preprocess/camera_K.pt`; it is
  validated against the resampled frame count and image dimensions.

## Output

The NPZ contains:

- `joint_pos`: `(T, 29)`, G1 scalar DoFs in BYD order;
- `root_pos_w`: `(T, 3)`, z-up world translation;
- `root_quat_w`: `(T, 4)`, wxyz z-up world quaternion;
- `body_pos_w`: predicted body points when available;
- camera calibration and provenance metadata.
