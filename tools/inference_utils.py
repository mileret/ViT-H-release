"""Shared utilities for loading the offline G1 pipeline."""

import gc

import numpy as np
import torch


def reduce_singular_chains(joint_pos_byd):
    """Rewrite equivalent Euler singularities without changing chain rotations."""
    from scipy.spatial.transform import Rotation as Rot

    chains = [
        ([0, 3, 6], "YXZ"),
        ([1, 4, 7], "YXZ"),
        ([2, 5, 8], "ZXY"),
        ([11, 15, 19], "YXZ"),
        ([12, 16, 20], "YXZ"),
        ([23, 25, 27], "XYZ"),
        ([24, 26, 28], "XYZ"),
    ]
    axis = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }

    out = joint_pos_byd.copy()
    for indices, order in chains:
        rotations = [
            Rot.from_rotvec(axis[name][None, :] * out[:, index, None])
            for index, name in zip(indices, order)
        ]
        equivalent = rotations[0] * rotations[1] * rotations[2]
        angles = equivalent.as_euler(order, degrees=False)
        for column, index in enumerate(indices):
            out[:, index] = angles[:, column]
    return out


def load_pipeline(
    ckpt_path,
    exp="offline/g1_training_set_exp1",
    device="cuda",
    overrides=None,
):
    """Compose the training config, instantiate the pipeline, and load weights."""
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    from hmr4d.configs import register_store_offline

    register_store_offline()
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(
            config_name="train",
            overrides=[f"exp={exp}", *(overrides or [])],
        )
    pipeline = instantiate(cfg.pipeline, _recursive_=False)

    try:
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        raise KeyError("Checkpoint has neither state_dict nor model_state_dict.")

    state_dict = {
        key.removeprefix("pipeline."): value
        for key, value in state_dict.items()
    }
    missing, unexpected = pipeline.load_state_dict(state_dict, strict=False)

    bad_missing = [key for key in missing if key.startswith("denoiser3d.")]
    bad_unexpected = [key for key in unexpected if key.startswith("denoiser3d.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Denoiser checkpoint mismatch. "
            f"missing={bad_missing[:5]}, unexpected={bad_unexpected[:5]}"
        )
    if missing:
        print(f"[warning] Missing non-denoiser keys: {missing[:5]}")
    if unexpected:
        print(f"[warning] Unexpected non-denoiser keys: {unexpected[:5]}")

    del checkpoint, state_dict
    gc.collect()
    return pipeline.eval().to(device)
