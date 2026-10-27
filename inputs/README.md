# External model assets

The trained G1 checkpoint is not distributed in this repository. Pass its
external path through `--ckpt`. The preprocessing pipeline additionally needs
these separately obtained files:

```text
inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
inputs/checkpoints/yolo/yolov8x.pt
inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
inputs/checkpoints/dpvo/dpvo.pth                 # optional
```

SMPL-X files are governed by their own license and are not redistributed here.
