# Offline video inference

## External files

The trained checkpoint is not distributed in this repository. Store it outside
the repository and pass its path through `--ckpt`.

The default HMR2 preprocessing path additionally expects:

```text
inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
inputs/checkpoints/yolo/yolov8x.pt
inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
```

`SMPLX_NEUTRAL.npz` is separately licensed and must be obtained under the
SMPL-X license. DPVO is optional; when `--use-dpvo` is used, also provide:

```text
inputs/checkpoints/dpvo/dpvo.pth
```

SAM3D is optional and requires the three `--sam3d-*` arguments.

## Command

```bash
python tools/infer_video.py \
  --video /path/to/input.mp4 \
  --ckpt /path/to/model.ckpt \
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
