# Offline video inference

## External files

The trained checkpoint is not distributed in this repository. Store it outside
the repository and pass its path through `--ckpt`.

The released training and inference pipeline uses SAM3D image features. It
expects the following files from the ViT-H checkpoint repository:

```text
vith_offline_g1.ckpt
inputs/checkpoints/dpvo/dpvo.pth
inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
inputs/checkpoints/yolo/yolov8x.pt
inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
```

The DPVO checkpoint is only used when `--use-dpvo` is passed. The default
SimpleVO path does not require DPVO.

The released configuration uses SAM3D. The bundled HMR2 checkpoint supports
the optional `--imgfeat-encoder hmr2` compatibility path.

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
  --imgfeat-encoder sam3d \
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

## Output

The NPZ contains:

- `joint_pos`: `(T, 29)`, G1 scalar DoFs in BYD order;
- `root_pos_w`: `(T, 3)`, z-up world translation;
- `root_quat_w`: `(T, 4)`, wxyz z-up world quaternion;
- `body_pos_w`: predicted body points when available;
- camera calibration and provenance metadata.
