"""Shared utilities for loading the offline G1 pipeline."""

import gc
from pathlib import Path

import numpy as np
import torch


_ENCODER_METADATA_KEYS = (
    "imgfeat_encoder",
    "image_feature_encoder",
    "feature_encoder",
)
_KNOWN_RELEASE_ENCODERS = {
    "vith_offline_g1.ckpt": "sam3d",
    "e629-s720720.ckpt": "sam3d",
}


def _load_checkpoint_file(ckpt_path):
    try:
        return torch.load(
            ckpt_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")


def _find_encoder_metadata(value, prefix="checkpoint"):
    if not isinstance(value, dict):
        return None
    for key in _ENCODER_METADATA_KEYS:
        encoder = value.get(key)
        if isinstance(encoder, str):
            encoder = encoder.lower()
            if encoder not in {"sam3d", "hmr2"}:
                raise ValueError(f"Unsupported {prefix}.{key}={encoder!r}")
            return encoder, f"{prefix}.{key}"
    for key in ("metadata", "hyper_parameters", "hparams", "config", "cfg"):
        nested = value.get(key)
        found = _find_encoder_metadata(nested, f"{prefix}.{key}")
        if found is not None:
            return found
    return None


def inspect_checkpoint_imgfeat_encoder(ckpt_path):
    """Return (encoder, provenance) without assuming custom checkpoints are SAM3D."""
    checkpoint = _load_checkpoint_file(ckpt_path)
    try:
        found = _find_encoder_metadata(checkpoint)
        if found is not None:
            return found
    finally:
        del checkpoint
        gc.collect()
    filename = Path(ckpt_path).name.lower()
    if filename in _KNOWN_RELEASE_ENCODERS:
        return (
            _KNOWN_RELEASE_ENCODERS[filename],
            f"documented release compatibility rule for {filename}",
        )
    return None, "checkpoint has no encoder metadata and is not a known release filename"


def resolve_imgfeat_encoder(requested, required, provenance, allow_mismatch=False):
    requested = requested.lower() if isinstance(requested, str) else None
    required = required.lower() if isinstance(required, str) else None
    if requested not in {None, "sam3d", "hmr2"}:
        raise ValueError(f"Unknown requested image-feature encoder: {requested!r}")
    if required not in {None, "sam3d", "hmr2"}:
        raise ValueError(f"Unknown checkpoint image-feature encoder: {required!r}")
    if requested is None:
        if required is None:
            raise ValueError(
                "This custom checkpoint does not declare an image-feature encoder. "
                "Pass --imgfeat-encoder sam3d or --imgfeat-encoder hmr2 explicitly."
            )
        print(f"[Checkpoint] image-feature encoder={required} ({provenance})")
        return required
    if required is not None and requested != required:
        message = (
            f"Checkpoint requires {required} features ({provenance}), but {requested} "
            "was requested. Feature dimensions alone do not make the encoders compatible."
        )
        if not allow_mismatch:
            raise ValueError(
                message
                + " Use --allow-encoder-mismatch only for an explicit experiment."
            )
        print(f"[warning] {message} Experimental override enabled; results are not guaranteed.")
    elif required is not None:
        print(f"[Checkpoint] verified image-feature encoder={requested} ({provenance})")
    else:
        print(f"[Checkpoint] explicit custom-checkpoint encoder={requested}; no metadata available")
    return requested


def _extract_pipeline_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        raise KeyError("Checkpoint has neither state_dict nor model_state_dict.")
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError("Checkpoint state dictionary is empty or invalid.")

    keys = list(state_dict)
    prefixed = [key.startswith("pipeline.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError(
            "Checkpoint mixes 'pipeline.'-prefixed and unprefixed keys; refusing "
            "ambiguous structural compatibility."
        )
    if all(prefixed):
        state_dict = {key[len("pipeline.") :]: value for key, value in state_dict.items()}
    return state_dict


def validate_and_load_model_state(model, state_dict):
    """Strictly validate every persistent model state before mutating the model."""
    expected = model.state_dict()
    expected_keys = set(expected)
    supplied_keys = set(state_dict)
    missing = sorted(expected_keys - supplied_keys)
    unexpected = sorted(supplied_keys - expected_keys)
    shape_mismatches = []
    for key in sorted(expected_keys & supplied_keys):
        expected_value = expected[key]
        supplied_value = state_dict[key]
        expected_shape = tuple(getattr(expected_value, "shape", ()))
        supplied_shape = tuple(getattr(supplied_value, "shape", ()))
        if expected_shape != supplied_shape:
            shape_mismatches.append((key, supplied_shape, expected_shape))
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Checkpoint is incompatible with the complete inference pipeline. "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatches={shape_mismatches[:20]}"
        )
    model.load_state_dict(state_dict, strict=True)
    return model


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

    checkpoint = _load_checkpoint_file(ckpt_path)
    state_dict = _extract_pipeline_state_dict(checkpoint)
    validate_and_load_model_state(pipeline, state_dict)

    del checkpoint, state_dict
    gc.collect()
    return pipeline.eval().to(device)
