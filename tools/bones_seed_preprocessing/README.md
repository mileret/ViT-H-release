# BONES-SEED G1/SOMA preprocessing

This directory contains the four Python entry points used in the preprocessing
workflow, together with their local SONIC FK/MotionLib dependencies, G1 MJCF,
and referenced meshes. `SOURCE_MANIFEST.json` records their paths and SHA-256
hashes. The exporter uses portable paths and additionally writes the index
consumed by the ViT-H training loader.

Workflow (four scripts, five invocations)
---------------------------------------
1. extract_soma_joints_with_eyes_global_from_bvh.py:
   SOMA BVH -> 30 fps PKLs, 28 global joints including both eyes, meters, Z-up.
   Intermediate soma_root_quat is wxyz in the original Y-up world frame.
2. filter_and_copy_bones_data.py: filter SOMA by built-in filename keywords.
3. convert_soma_csv_to_motion_lib.py:
   Already-retargeted G1 CSV (120 fps) -> SONIC motion PKLs (30 fps).
   This does not retarget human BVH to G1; the downloaded G1 CSV is required.
4. filter_and_copy_bones_data.py: apply the same keyword filter to G1 PKLs.
5. `export_sonic_paired_motions.py`:
   Load 30 fps G1 using SONIC MotionLibRobot/FK, pair by shared motion key,
   copy SOMA joints, and convert SOMA root rotation to Z-up axis-angle.
   No further temporal resampling is requested at this stage. Unequal-length
   pairs and missing inputs are skipped and recorded in `export_errors.log`.
   The exporter also writes `bones_seed_index_v1.json` for training.

Installation and execution
--------------------------
Linux, Python 3.10. Example setup for a new environment:

    conda create -n bones-seed-preprocess python=3.10
    conda activate bones-seed-preprocess
    pip install -r tools/bones_seed_preprocessing/requirements.txt

Select a PyTorch build suitable for your machine. environment_versions.txt is
the local packaging environment snapshot, not a cross-platform dependency lock.
No environment is created or modified by run_preprocessing.sh.

Expected raw input layout:
    bones-seed/soma_proportional/bvh/<session>/*.bvh
    bones-seed/g1/csv/<session>/*.csv

Run all five stages from the ViT-H repository root:

    bash tools/bones_seed_preprocessing/run_preprocessing.sh \
      /path/to/bones-seed \
      /path/to/training_set/bones_seed \
      16

The wrapper explicitly supplies portable paths. In particular, the original
exporter has machine-specific default paths; when invoking it directly, supply
all four: --sonic-root, --motion-file, --soma-motion-file, --output-dir.
Use a fresh output directory: the BVH stage skips existing files, the filter
copies do not remove stale files, and the exporter refuses to overwrite shards.
The exporter uses its original worker defaults; its --num-load-jobs option is
passed through to MotionLib, whose internal scheduling controls actual workers.

Output
------
`paired/<session>.pkl` is a joblib dictionary keyed by shared motion name:
    g1:
        fps: 30
        root_pos: (T, 3) float32, body_pos_w_full[:, 0]
        root_rot: (T, 4) float32, wxyz, body_quat_w_full[:, 0]
        dof_pos: (T, 29) float32
        body_pos_w_full: (T, 30, 3) float32
        body_quat_w_full: (T, 30, 4) float32, wxyz
    soma:
        fps: 30
        global_orient: (T, 3) float32, Z-up axis-angle
        joints: (T, 28, 3) float32, Z-up global positions in meters
All array fields are NumPy arrays. No SMPL or transl field is exported.
Each pair contributes T aligned time steps, counted once across both modalities.
First-frame canonicalization and observation construction are not part of this
offline pipeline; canonicalize_motion.py and training scripts are not included.
The older 26-joint/body-local BVH extractor is not used in this workflow.
`paired/bones_seed_index_v1.json` contains the file fingerprint and sequence
metadata used to avoid rescanning every compressed shard at training startup.

SOMA joint order
----------------
0 Hips, 1 Spine1, 2 Spine2, 3 Chest, 4 Neck1, 5 Head,
6 LeftEye, 7 RightEye, 8 LeftShoulder, 9 LeftArm, 10 LeftForeArm,
11 LeftHand, 12 LeftHandThumb1, 13 LeftHandMiddle1,
14 RightShoulder, 15 RightArm, 16 RightForeArm, 17 RightHand,
18 RightHandThumb1, 19 RightHandMiddle1,
20 LeftLeg, 21 LeftShin, 22 LeftFoot, 23 LeftToeBase,
24 RightLeg, 25 RightShin, 26 RightFoot, 27 RightToeBase.

Provenance
----------
Upstream LICENSE, CITATION.cff and legal notices are preserved under
`GR00T-WholeBodyControl/`. Refer to those notices for bundled code and assets.
No raw motions, generated datasets, checkpoints or local execution logs are
included. Raw data must be obtained separately by the recipient.

Validation at packaging time (2026-11-01)
----------------------------------------
All four entry points passed --help in Python isolated mode. The shell wrapper
passed bash -n. The bundled exporter loaded its bundled FK code and meshes from
an unrelated working directory and exported two real preprocessed motions
(original and mirrored jump_and_land_heavy_001__A001, 732 paired frames total).
Checks passed for exact keys, array shapes, float32 dtypes, finite values,
30 fps, root aliases, and unchanged SOMA joint positions.
The raw BVH/CSV dataset was not present at its previous local path, so the raw
conversion stages were not rerun end-to-end during packaging.
