# G1 Motion Tracker

This directory is a release-ready overlay for ViT-H. It tracks a G1 reference
motion with the 1570-dimensional SONIC deploy policy in MuJoCo and writes the
same protocol-v4 fields used by the evaluation pipeline.

## What is included

- `vith_tracker/`: resampling, grounding, MuJoCo rollout, and success logic.
- `tools/track_g1.py`: single-file command-line entry point.
- `assets/g1/`: the rollout and grounding XML files with their required meshes.
- `requirements-tracker.txt`: tracker-only Python dependencies.
- `tests/`: lightweight schema and math tests.

The ONNX policy checkpoint is intentionally excluded and must be downloaded
separately. The included Unitree robot assets retain their BSD 3-Clause license
in `assets/g1/LICENSE`.

## Input

The input is an NPZ file with either unbatched or batched arrays:

| Field | Unbatched shape | Batched shape | Convention |
| --- | --- | --- | --- |
| `joint_pos` | `(T, 29)` | `(S, T, 29)` | IsaacLab G1 order, radians |
| `root_pos_w` | `(T, 3)` | `(S, T, 3)` | world frame, Z-up, metres |
| `root_quat_w` | `(T, 4)` | `(S, T, 4)` | world frame, `wxyz` |
| `body_pos_w` | `(T, 30, 3)` | `(S, T, 30, 3)` | world frame, Z-up, metres |

The source frame rate must be supplied with `--input-fps` or stored as a scalar
`fps` field in the NPZ. The tracker resamples to 50 Hz before rollout and
resamples the rollout back to the source timestamps.

## Run

Install the tracker dependencies in the same environment as ViT-H, then run:

```bash
python g1_tracker/tools/track_g1.py \
  --input outputs/example/motion.npz \
  --output outputs/example/tracked_motion.npz \
  --input-fps 30 \
  --policy-onnx path/to/model_step_020000_g1.onnx \
  --xml-path g1_tracker/assets/g1/rollout/scene_29dof_mujoco.xml \
  --grounding-xml g1_tracker/assets/g1/grounding/mjcf/g1_29dof_rev_1_0.xml
```

`--xml-path` is the rollout model. `--grounding-xml` is the 29-DoF FK model
whose initial foot-contact spheres define the support-plane offset. Using a
different robot definition changes the rollout and is not equivalent.

### Optional MuJoCo video

Add `--render-video` to render the tracked result after the rollout:

```bash
MUJOCO_GL=egl python g1_tracker/tools/track_g1.py \
  --input outputs/example/motion.npz \
  --output outputs/example/tracked_motion.npz \
  --input-fps 30 \
  --policy-onnx path/to/model_step_020000_g1.onnx \
  --xml-path g1_tracker/assets/g1/rollout/scene_29dof_mujoco.xml \
  --grounding-xml g1_tracker/assets/g1/grounding/mjcf/g1_29dof_rev_1_0.xml \
  --render-video
```

The default video path is `outputs/example/tracked_motion.mp4`. Use
`--video-output`, `--video-width`, and `--video-height` to override it. Batched
inputs produce one file per sample with `_sample_000`, `_sample_001`, and so on.
`MUJOCO_GL=egl` is recommended on headless Linux nodes; omit it on a desktop if
the default OpenGL backend works.

## Output

The output NPZ contains:

- `joint_pos`, `root_pos_w`, `root_quat_w`, `body_pos_w`
- `is_success`
- `first_failure_frame_50hz` (`-1` means no failure)
- `grounding_offset_m`
- `initial_root_lin_vel_w_m_s`, `initial_root_ang_vel_w_rad_s`
- `initial_joint_vel_rad_s`
- `initial_joint_clip_count`, `initial_joint_clip_max_abs_rad`
- `tracker_protocol_version` (value `4`)

The success criterion is evaluated at 50 Hz. It fails when pelvis height error
exceeds 0.25 m, squared pelvis orientation error exceeds 1.0 radian squared, or
world-Z error of an ankle or wrist exceeds 0.25 m.
