#!/usr/bin/env python3
"""Export paired SONIC G1/SOMA motions after G1 SONIC preprocessing.

The output schema is:

    {
        "<shared_motion_key>": {
            "g1": {
                "fps": 30,
                "root_pos": ...,          # (T, 3) float32
                "root_rot": ...,          # (T, 4) float32, wxyz
                "dof_pos": ...,           # (T, 29) float32
                "body_pos_w_full": ...,   # (T, 30, 3) float32
                "body_quat_w_full": ...,  # (T, 30, 4) float32, wxyz
            },
            "soma": {
                "fps": 30,
                "global_orient": ...,     # (T, 3) float32, Z-up axis-angle
                "joints": ...,            # (T, 28, 3) float32, Z-up global positions
            },
        },
    }

Only complete G1/SOMA pairs are exported. G1 motions are loaded through SONIC's
MotionLibRobot at 30 fps so the G1 cache is produced by the same FK path used
by training, but without interpolation or training augmentations. SOMA motions
are copied from the preprocessed 28-joint global-Z-up PKLs, except the root
quaternion is converted from Y-up wxyz quaternion to Z-up axis-angle.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import os.path as osp
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import easydict
import joblib
import numpy as np
from scipy.spatial.transform import Rotation
import torch


BUNDLE_ROOT = Path(__file__).resolve().parents[3]
SONIC_ROOT = str(BUNDLE_ROOT / "GR00T-WholeBodyControl")
DEFAULT_MOTION_FILE = osp.join(SONIC_ROOT, "data/motion_lib_bones_seed/robot_filtered")
DEFAULT_SOMA_MOTION_FILE = osp.join(SONIC_ROOT, "data/motion_lib_bones_seed/soma_filtered")
DEFAULT_OUTPUT_DIR = osp.join(SONIC_ROOT, "data/sonic_g1_soma_paired_processed")
DEFAULT_ERROR_LOG = "export_errors.log"
DEFAULT_FPS = 30
NUM_SOMA_JOINTS = 28

BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# Copied from gear_sonic.envs.manager_env.robots.g1 to avoid importing
# IsaacLab robot config in a plain Python process, where carb is unavailable.
G1_ISAACLAB_JOINTS = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

G1_ISAACLAB_TO_MUJOCO_DOF = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]

G1_MUJOCO_TO_ISAACLAB_DOF = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]

G1_ISAACLAB_TO_MUJOCO_BODY = [
    0,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    3,
    6,
    9,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
    13,
    17,
    21,
    23,
    25,
    27,
    29,
]

G1_MUJOCO_TO_ISAACLAB_BODY = [
    0,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    6,
    12,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
    22,
    29,
]

G1_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": G1_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": G1_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": G1_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": G1_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": G1_MUJOCO_TO_ISAACLAB_BODY,
}

REQUIRED_G1_KEYS = ("root_trans_offset", "pose_aa", "fps")
REQUIRED_SOMA_KEYS = ("soma_joints", "soma_root_quat", "fps")


@dataclass(frozen=True)
class MotionPair:
    key: str
    g1_path: str
    soma_path: str
    shard_name: str


@dataclass
class FullMotionState:
    motion_data_by_key: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export complete paired SONIC G1/SOMA motions after G1 SONIC preprocessing."
    )
    parser.add_argument("--sonic-root", default=SONIC_ROOT)
    parser.add_argument("--motion-file", default=DEFAULT_MOTION_FILE)
    parser.add_argument("--soma-motion-file", default=DEFAULT_SOMA_MOTION_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-mode",
        choices=("shard-by-folder", "single"),
        default="shard-by-folder",
        help="Default is shard-by-folder to avoid holding the full dataset in memory.",
    )
    parser.add_argument("--target-fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--num-load-jobs", type=int, default=1)
    parser.add_argument("--max-motions", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--compress", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--error-log",
        default=None,
        help="Defaults to <output-dir>/export_errors.log.",
    )
    return parser.parse_args()


def add_sonic_to_path(sonic_root: str) -> None:
    if sonic_root not in sys.path:
        sys.path.insert(0, sonic_root)


def make_motion_lib_cfg(args: argparse.Namespace) -> easydict.EasyDict:
    mapping = G1_ISAACLAB_TO_MUJOCO_MAPPING
    body_indexes_data = [mapping["isaaclab_joints"].index(name) for name in BODY_NAMES]

    return easydict.EasyDict(
        {
            "motion_file": args.motion_file,
            "smpl_motion_file": None,
            "asset": {
                "assetRoot": osp.join(
                    args.sonic_root,
                    "gear_sonic/data/assets/robot_description/mjcf",
                ),
                "assetFileName": "g1_29dof_rev_1_0.xml",
                "urdfFileName": "",
            },
            "extend_config": [],
            "target_fps": args.target_fps,
            "multi_thread": True,
            "num_load_jobs": args.num_load_jobs,
            "filter_motion_keys": None,
            "adaptive_sampling": {"enable": False},
            "mujoco_to_isaaclab_dof": mapping["mujoco_to_isaaclab_dof"],
            "mujoco_to_isaaclab_body": mapping["mujoco_to_isaaclab_body"],
            "isaaclab_to_mujoco_dof": mapping["isaaclab_to_mujoco_dof"],
            "isaaclab_to_mujoco_body": mapping["isaaclab_to_mujoco_body"],
            "body_names": BODY_NAMES,
            "body_indexes_data": body_indexes_data,
            "lower_joint_indices_mujoco": list(range(12)),
            "cat_upper_body_poses": False,
            "cat_upper_body_poses_prob": 0.0,
            "randomize_heading": False,
            "freeze_frame_aug": False,
            "randomize_wrist_poses": False,
            "randomize_wrist_prob": 0.0,
            "randomize_wrist_std": 0.0,
            "debug": False,
        }
    )


def unwrap_single_motion_dict(data: Any, expected_key: str | None = None) -> Any:
    if not isinstance(data, dict):
        return data
    if expected_key is not None and expected_key in data and isinstance(data[expected_key], dict):
        return data[expected_key]
    if any(key in data for key in (*REQUIRED_G1_KEYS, *REQUIRED_SOMA_KEYS)):
        return data
    if len(data) == 1:
        only_value = next(iter(data.values()))
        if isinstance(only_value, dict):
            return only_value
    return data


def has_required_keys(data: Any, required_keys: tuple[str, ...]) -> bool:
    return isinstance(data, dict) and all(key in data and data[key] is not None for key in required_keys)


def get_record_path(record: Any) -> str | None:
    if isinstance(record, dict) and "path" in record:
        return record["path"]
    return None


def write_error(error_log_path: str, motion_key: str, reason: str) -> None:
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(f"{motion_key}\t{reason}\n")
        f.flush()


def load_single_motion(path: str, motion_key: str) -> dict[str, Any]:
    data = unwrap_single_motion_dict(joblib.load(path), motion_key)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict motion record in {path}, got {type(data).__name__}")
    return data


def validate_fps(data: dict[str, Any], target_fps: int, prefix: str) -> tuple[bool, str | None]:
    try:
        fps = int(round(float(data["fps"])))
    except Exception as exc:  # noqa: BLE001
        return False, f"{prefix}_invalid_fps: {exc}"
    if fps != target_fps:
        return False, f"{prefix}_fps_mismatch: got {fps}, expected {target_fps}"
    return True, None


def validate_g1_record(record: Any, motion_key: str, target_fps: int) -> tuple[bool, str | None]:
    g1_path = get_record_path(record)
    if g1_path is None or not osp.isfile(g1_path):
        return False, "missing_g1_file"
    try:
        g1_data = load_single_motion(g1_path, motion_key)
    except Exception as exc:  # noqa: BLE001
        return False, f"g1_load_error: {exc}"
    if not has_required_keys(g1_data, REQUIRED_G1_KEYS):
        missing = [key for key in REQUIRED_G1_KEYS if key not in g1_data or g1_data[key] is None]
        return False, f"missing_g1_keys: {missing}"
    fps_ok, fps_reason = validate_fps(g1_data, target_fps, "g1")
    if not fps_ok:
        return False, fps_reason
    root_trans_offset = np.asarray(g1_data["root_trans_offset"])
    pose_aa = np.asarray(g1_data["pose_aa"])
    if root_trans_offset.shape[0] == 0 or pose_aa.shape[0] == 0:
        return False, "empty_g1_arrays"
    return True, None


def validate_soma_record(soma_path: str | None, motion_key: str, target_fps: int) -> tuple[bool, str | None]:
    if soma_path is None or not osp.isfile(soma_path):
        return False, "missing_soma_file"
    try:
        soma_data = load_single_motion(soma_path, motion_key)
    except Exception as exc:  # noqa: BLE001
        return False, f"soma_load_error: {exc}"
    if not has_required_keys(soma_data, REQUIRED_SOMA_KEYS):
        missing = [key for key in REQUIRED_SOMA_KEYS if key not in soma_data or soma_data[key] is None]
        return False, f"missing_soma_keys: {missing}"
    fps_ok, fps_reason = validate_fps(soma_data, target_fps, "soma")
    if not fps_ok:
        return False, fps_reason

    soma_joints = np.asarray(soma_data["soma_joints"])
    soma_root_quat = np.asarray(soma_data["soma_root_quat"])
    if soma_joints.ndim != 3 or soma_joints.shape[1:] != (NUM_SOMA_JOINTS, 3):
        return False, f"bad_soma_joints_shape: {soma_joints.shape}"
    if soma_root_quat.ndim != 2 or soma_root_quat.shape[1] != 4:
        return False, f"bad_soma_root_quat_shape: {soma_root_quat.shape}"
    if soma_joints.shape[0] == 0 or soma_root_quat.shape[0] == 0:
        return False, "empty_soma_arrays"
    if soma_joints.shape[0] != soma_root_quat.shape[0]:
        return False, f"soma_length_mismatch: joints={soma_joints.shape[0]}, root_quat={soma_root_quat.shape[0]}"
    return True, None


def build_soma_index(soma_motion_file: str) -> dict[str, str]:
    assert osp.isdir(soma_motion_file), f"Expected soma_motion_file to be a directory: {soma_motion_file}"
    pkl_files = sorted(glob.glob(osp.join(soma_motion_file, "**", "*.pkl"), recursive=True))
    return {
        osp.splitext(osp.basename(path))[0]: path
        for path in pkl_files
        if not path.endswith("metadata.pkl")
    }


def build_complete_pairs(
    motion_lib: Any,
    soma_index: dict[str, str],
    error_log_path: str,
    target_fps: int,
    max_pairs: int | None = None,
) -> list[MotionPair]:
    pairs: list[MotionPair] = []
    motion_keys = motion_lib._motion_data_keys.tolist()  # noqa: SLF001
    num_motion_keys = len(motion_keys)
    scanned_count = 0
    for idx, motion_key_raw in enumerate(motion_keys):
        if idx > 0 and idx % 5000 == 0:
            print(
                f"[export] scanned {idx}/{num_motion_keys} motions, complete pairs: {len(pairs)}",
                flush=True,
            )
        scanned_count = idx + 1
        motion_key = str(motion_key_raw)
        g1_record = motion_lib._motion_data_list[idx]  # noqa: SLF001

        g1_ok, g1_reason = validate_g1_record(g1_record, motion_key, target_fps)
        if not g1_ok:
            write_error(error_log_path, motion_key, g1_reason or "invalid_g1")
            continue

        soma_path = soma_index.get(motion_key)
        soma_ok, soma_reason = validate_soma_record(soma_path, motion_key, target_fps)
        if not soma_ok:
            write_error(error_log_path, motion_key, soma_reason or "invalid_soma")
            continue

        g1_path = get_record_path(g1_record)
        assert g1_path is not None
        assert soma_path is not None
        shard_name = osp.basename(osp.dirname(g1_path))
        pairs.append(
            MotionPair(key=motion_key, g1_path=g1_path, soma_path=soma_path, shard_name=shard_name)
        )
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    print(
        f"[export] scanned {scanned_count}/{num_motion_keys} motions, complete pairs: {len(pairs)}",
        flush=True,
    )
    return pairs


def capture_full_motion_state(motion_lib: Any) -> FullMotionState:
    motion_data_by_key = {
        str(key): motion_lib._motion_data_list[idx]  # noqa: SLF001
        for idx, key in enumerate(motion_lib._motion_data_keys.tolist())  # noqa: SLF001
    }
    return FullMotionState(motion_data_by_key=motion_data_by_key)


def reset_motion_lib_subset(motion_lib: Any, full_state: FullMotionState, keys: list[str]) -> None:
    motion_lib._motion_data_keys = np.array(keys, dtype=object)  # noqa: SLF001
    motion_lib._motion_data_list = np.array(  # noqa: SLF001
        [full_state.motion_data_by_key[key] for key in keys],
        dtype=object,
    )
    motion_lib._num_unique_motions = len(keys)  # noqa: SLF001
    motion_lib._termination_history = torch.zeros(len(keys), device=motion_lib._device)  # noqa: SLF001
    motion_lib._success_rate = torch.zeros(len(keys), device=motion_lib._device)  # noqa: SLF001
    motion_lib._sampling_history = torch.zeros(len(keys), device=motion_lib._device)  # noqa: SLF001
    motion_lib._sampling_prob = (  # noqa: SLF001
        torch.ones(len(keys), device=motion_lib._device, dtype=torch.float32) / len(keys)  # noqa: SLF001
    )
    motion_lib.all_motions_loaded = False


def tensor_slice_numpy(tensor: torch.Tensor, start: int, end: int) -> np.ndarray:
    return tensor[start:end].detach().cpu().numpy().copy()


def scalar_tensor_item(tensor: torch.Tensor, index: int) -> int | float:
    value = tensor[index]
    if value.dtype.is_floating_point:
        return float(value.detach().cpu().item())
    return int(value.detach().cpu().item())


def soma_root_quat_wxyz_yup_to_axis_angle_zup(root_quat_wxyz: np.ndarray) -> np.ndarray:
    root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float64)
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    y_to_z_rotation = Rotation.from_rotvec(np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float64))
    root_rotation_zup = y_to_z_rotation * Rotation.from_quat(root_quat_xyzw)
    return root_rotation_zup.as_rotvec().astype(np.float32)


def load_soma_export(
    soma_path: str,
    motion_key: str,
    target_fps: int,
    expected_num_frames: int,
) -> dict[str, np.ndarray | int]:
    soma_data = load_single_motion(soma_path, motion_key)
    soma_ok, soma_reason = validate_soma_record(soma_path, motion_key, target_fps)
    if not soma_ok:
        raise ValueError(soma_reason or "invalid_soma")

    joints = np.asarray(soma_data["soma_joints"], dtype=np.float32)
    root_quat_wxyz = np.asarray(soma_data["soma_root_quat"], dtype=np.float32)
    if joints.shape[0] != expected_num_frames:
        raise ValueError(
            f"soma_g1_length_mismatch: soma={joints.shape[0]}, g1={expected_num_frames}"
        )
    return {
        "fps": target_fps,
        "global_orient": soma_root_quat_wxyz_yup_to_axis_angle_zup(root_quat_wxyz),
        "joints": joints.copy(),
    }


def export_loaded_motion_lib(
    motion_lib: Any,
    expected_keys: list[str],
    soma_path_by_key: dict[str, str],
    error_log_path: str,
    target_fps: int,
) -> dict[str, dict[str, dict[str, np.ndarray | int]]]:
    exported: dict[str, dict[str, dict[str, np.ndarray | int]]] = {}
    keys_to_export = expected_keys
    if len(expected_keys) == 1:
        # load_motions() has a single-sample curr_motion_keys edge case, so the
        # caller loads two clamped copies. Only export the first copy.
        keys_to_export = expected_keys[:1]

    for motion_idx, motion_key in enumerate(keys_to_export):
        start = int(motion_lib.length_starts[motion_idx].detach().cpu().item())
        num_frames = int(motion_lib._motion_num_frames[motion_idx].detach().cpu().item())  # noqa: SLF001
        end = start + num_frames
        motion_fps = int(round(scalar_tensor_item(motion_lib._motion_fps, motion_idx)))  # noqa: SLF001
        if motion_fps != target_fps:
            write_error(
                error_log_path,
                motion_key,
                f"g1_cache_fps_mismatch: got {motion_fps}, expected {target_fps}",
            )
            continue

        body_pos_w_full = tensor_slice_numpy(motion_lib.body_pos_w_full, start, end)
        body_quat_w_full = tensor_slice_numpy(motion_lib.body_quat_w_full, start, end)
        g1: dict[str, np.ndarray | int] = {
            "fps": target_fps,
            "root_pos": body_pos_w_full[:, 0, :].copy(),
            "root_rot": body_quat_w_full[:, 0, :].copy(),
            "dof_pos": tensor_slice_numpy(motion_lib.dof_pos, start, end),
            "body_pos_w_full": body_pos_w_full,
            "body_quat_w_full": body_quat_w_full,
        }

        try:
            soma = load_soma_export(
                soma_path_by_key[motion_key],
                motion_key,
                target_fps,
                expected_num_frames=num_frames,
            )
        except Exception as exc:  # noqa: BLE001
            write_error(error_log_path, motion_key, f"soma_export_error: {exc}")
            continue

        exported[motion_key] = {"g1": g1, "soma": soma}
    return exported


def load_and_export_keys(
    motion_lib: Any,
    full_state: FullMotionState,
    keys: list[str],
    soma_path_by_key: dict[str, str],
    error_log_path: str,
    target_fps: int,
) -> dict[str, dict[str, dict[str, np.ndarray | int]]]:
    if len(keys) == 0:
        return {}

    try:
        reset_motion_lib_subset(motion_lib, full_state, keys)
        num_motions_to_load = len(keys)
        if len(keys) == 1:
            # Work around the upstream single-motion curr_motion_keys edge case.
            num_motions_to_load = 2
        motion_lib.load_motions(
            random_sample=False,
            start_idx=0,
            num_motions_to_load=num_motions_to_load,
            is_evaluation=True,
        )
        return export_loaded_motion_lib(
            motion_lib,
            keys,
            soma_path_by_key,
            error_log_path,
            target_fps,
        )
    except Exception as exc:  # noqa: BLE001
        if len(keys) == 1:
            write_error(error_log_path, keys[0], f"sonic_preprocess_error: {exc}")
            write_error(error_log_path, keys[0], traceback.format_exc())
            return {}
        midpoint = len(keys) // 2
        left = load_and_export_keys(
            motion_lib,
            full_state,
            keys[:midpoint],
            soma_path_by_key,
            error_log_path,
            target_fps,
        )
        right = load_and_export_keys(
            motion_lib,
            full_state,
            keys[midpoint:],
            soma_path_by_key,
            error_log_path,
            target_fps,
        )
        left.update(right)
        return left
    finally:
        gc.collect()


def dump_joblib(data: Any, output_path: str, compress: int, overwrite: bool) -> None:
    if osp.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    joblib.dump(data, output_path, compress=compress)


def append_index_records(
    records: list[dict[str, Any]],
    data: dict[str, dict[str, dict[str, np.ndarray | int]]],
    file_name: str,
) -> None:
    file_stem = Path(file_name).stem
    for sequence_name, payload in sorted(data.items()):
        g1_length = int(np.asarray(payload["g1"]["root_pos"]).shape[0])
        soma_length = int(np.asarray(payload["soma"]["joints"]).shape[0])
        if g1_length != soma_length:
            raise ValueError(
                f"Unpaired exported lengths for {sequence_name}: "
                f"G1={g1_length}, SOMA={soma_length}"
            )
        records.append(
            {
                "key": f"{file_stem}/{sequence_name}",
                "file_name": file_name,
                "seq_name": sequence_name,
                "length": g1_length,
            }
        )


def write_training_index(output_dir: str, records: list[dict[str, Any]]) -> str:
    fingerprint = []
    for path in sorted(Path(output_dir).glob("*.pkl")):
        stat = path.stat()
        fingerprint.append(
            {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )

    index_path = Path(output_dir) / "bones_seed_index_v1.json"
    tmp_path = index_path.with_name(f".{index_path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        payload = {
            "version": 1,
            "fingerprint": fingerprint,
            "records": sorted(records, key=lambda item: item["key"]),
        }
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
        )
    os.replace(tmp_path, index_path)
    return str(index_path)


def group_pairs_by_shard(pairs: list[MotionPair]) -> dict[str, list[str]]:
    shard_to_keys: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        shard_to_keys[pair.shard_name].append(pair.key)
    return {shard: sorted(keys) for shard, keys in sorted(shard_to_keys.items())}


def main() -> None:
    args = parse_args()
    args.sonic_root = osp.abspath(args.sonic_root)
    args.motion_file = osp.abspath(args.motion_file)
    args.soma_motion_file = osp.abspath(args.soma_motion_file)
    args.output_dir = osp.abspath(args.output_dir)
    error_log_path = args.error_log or osp.join(args.output_dir, DEFAULT_ERROR_LOG)

    os.makedirs(args.output_dir, exist_ok=True)
    if osp.exists(error_log_path) and args.overwrite:
        os.remove(error_log_path)

    add_sonic_to_path(args.sonic_root)

    from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot

    motion_lib_cfg = make_motion_lib_cfg(args)
    motion_lib = MotionLibRobot(motion_lib_cfg, num_envs=1, device=torch.device("cpu"))
    full_state = capture_full_motion_state(motion_lib)
    soma_index = build_soma_index(args.soma_motion_file)

    print("[export] Scanning complete G1/SOMA pairs...")  # noqa: T201
    pairs = build_complete_pairs(
        motion_lib,
        soma_index,
        error_log_path,
        target_fps=args.target_fps,
        max_pairs=args.max_motions,
    )
    pairs.sort(key=lambda pair: (pair.shard_name, pair.key))
    soma_path_by_key = {pair.key: pair.soma_path for pair in pairs}
    print(f"[export] Complete pairs selected: {len(pairs)}")  # noqa: T201

    if len(pairs) == 0:
        print(f"[export] No complete pairs found. See {error_log_path}")  # noqa: T201
        return

    index_records: list[dict[str, Any]] = []
    if args.output_mode == "single":
        keys = [pair.key for pair in pairs]
        data = load_and_export_keys(
            motion_lib,
            full_state,
            keys,
            soma_path_by_key,
            error_log_path,
            target_fps=args.target_fps,
        )
        output_path = osp.join(args.output_dir, "sonic_g1_soma_paired_processed.pkl")
        dump_joblib(data, output_path, args.compress, args.overwrite)
        append_index_records(index_records, data, osp.basename(output_path))
        index_path = write_training_index(args.output_dir, index_records)
        print(f"[export] Wrote {len(data)} motions to {output_path}")  # noqa: T201
        print(f"[export] Wrote training index to {index_path}")  # noqa: T201
        return

    shard_to_keys = group_pairs_by_shard(pairs)
    shard_items = list(shard_to_keys.items())
    if args.max_shards is not None:
        shard_items = shard_items[: args.max_shards]

    total_written = 0
    for shard_idx, (shard_name, keys) in enumerate(shard_items, start=1):
        output_path = osp.join(args.output_dir, f"{shard_name}.pkl")
        print(  # noqa: T201
            f"[export] Shard {shard_idx}/{len(shard_items)}: {shard_name} ({len(keys)} motions)"
        )
        data = load_and_export_keys(
            motion_lib,
            full_state,
            keys,
            soma_path_by_key,
            error_log_path,
            target_fps=args.target_fps,
        )
        dump_joblib(data, output_path, args.compress, args.overwrite)
        append_index_records(index_records, data, osp.basename(output_path))
        total_written += len(data)
        print(f"[export] Wrote {len(data)} motions to {output_path}")  # noqa: T201

    index_path = write_training_index(args.output_dir, index_records)
    print(f"[export] Done. Wrote {total_written} motions across {len(shard_items)} shards.")  # noqa: T201
    print(f"[export] Training index: {index_path}")  # noqa: T201
    print(f"[export] Error log: {error_log_path}")  # noqa: T201


if __name__ == "__main__":
    main()
