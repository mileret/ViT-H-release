"""Shared SAM3D configuration lookup and effective inference settings."""

import os
from pathlib import Path

from omegaconf import OmegaConf


FAST_SAM3D_DEFAULTS = {
    "IMG_SIZE": "512",
    "SKIP_KEYPOINT_PROMPT": "1",
    "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
    "BODY_INTERM_PRED_LAYERS": "0,1,2",
    "HAND_INTERM_PRED_LAYERS": "0,1",
    "MHR_NO_CORRECTIVES": "1",
    "LAYER_DTYPE": "fp32",
    "DEBUG_NAN": "0",
    "DEBUG_HAND_PREP": "0",
    "DEBUG_BACKBONE_INPUT": "0",
    "INTERM_TIMING": "0",
}


def is_fast_sam3d(root):
    # Preserve the extractor's existing default-selection rule.
    return bool(root and "fast-sam-3d-body" in str(root).lower())


def setup_fast_sam3d_env():
    for key, value in FAST_SAM3D_DEFAULTS.items():
        os.environ.setdefault(key, value)


def resolve_sam3d_model_config(checkpoint_path):
    """Match load_sam_3d_body: checkpoint sibling, then parent directory."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    candidates = (
        checkpoint.parent / "model_config.yaml",
        checkpoint.parent.parent / "model_config.yaml",
    )
    for candidate in candidates:
        # An existing but invalid sibling must not silently fall back either.
        if candidate.exists():
            if not candidate.is_file():
                raise FileNotFoundError(f"SAM3D model config is not a file: {candidate}")
            return candidate
    raise FileNotFoundError(
        f"SAM3D model_config.yaml not found for checkpoint {checkpoint}; "
        f"checked {candidates[0]} and {candidates[1]}"
    )


def sam3d_runtime_settings(root, checkpoint_path, environ=None):
    """Describe effective values without mutating the process environment.

    Explicit settings equal to defaults have the same identity. IMG_SIZE=0
    (or empty) uses MODEL.IMAGE_SIZE; dtype aliases follow Fast-SAM's loader.
    The selected YAML is separately content-hashed in the resource identity.
    """
    env = dict(os.environ if environ is None else environ)
    if is_fast_sam3d(root):
        for key, value in FAST_SAM3D_DEFAULTS.items():
            env.setdefault(key, value)
    config_path = resolve_sam3d_model_config(checkpoint_path)
    config = OmegaConf.load(config_path)
    size = int(env.get("IMG_SIZE", "0") or "0")
    image_size = [size, size] if size > 0 else OmegaConf.to_container(config.MODEL.IMAGE_SIZE)
    if not isinstance(image_size, list) or len(image_size) != 2 or any(int(v) <= 0 for v in image_size):
        raise ValueError(f"Invalid SAM3D MODEL.IMAGE_SIZE in {config_path}: {image_size}")
    dtype = env.get("LAYER_DTYPE", "").lower()
    dtype = {"fp16": "fp16", "float16": "fp16", "bf16": "bf16", "bfloat16": "bf16"}.get(dtype, "fp32")
    use_compile = env.get("USE_COMPILE", "0").lower() in ("1", "true", "yes")
    settings = {
        "image_size": [int(v) for v in image_size],
        "layer_dtype": dtype,
        "use_compile": use_compile,
        "compile_mode": env.get("COMPILE_MODE", "reduce-overhead") if use_compile else None,
    }
    # Include other computation-affecting Fast-SAM switches, with upstream
    # defaults where the extractor does not override them.
    defaults = {
        "SKIP_KEYPOINT_PROMPT": "0",
        "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
        "KEYPOINT_PROMPT_INTERM_LAYERS": "",
        "BODY_INTERM_PRED_LAYERS": "",
        "HAND_INTERM_PRED_LAYERS": "",
        "MHR_NO_CORRECTIVES": "0",
        "GPU_HAND_PREP": "1",
        "PARALLEL_DECODERS": "1",
    }
    settings["inference_options"] = {key: env.get(key, value) for key, value in defaults.items()}
    if use_compile:
        settings["compile_options"] = {
            key: env.get(key, value)
            for key, value in {
                "DECODER_COMPILE": "1", "INTERM_COMPILE": "1",
                "INTERM_FULLGRAPH": "0", "INTERM_SLIM": "1",
            }.items()
        }
    return settings
