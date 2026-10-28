#!/usr/bin/env python3
"""Convert the official BONES-SEED release to the paired training format."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_FPS = 120
TARGET_FPS = 30

MJC_JOINT_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)
BYD_JOINT_NAMES = (
    "left_hip_pitch", "right_hip_pitch", "waist_yaw",
    "left_hip_roll", "right_hip_roll", "waist_roll",
    "left_hip_yaw", "right_hip_yaw", "waist_pitch",
    "left_knee", "right_knee",
    "left_shoulder_pitch", "right_shoulder_pitch",
    "left_ankle_pitch", "right_ankle_pitch",
    "left_shoulder_roll", "right_shoulder_roll",
    "left_ankle_roll", "right_ankle_roll",
    "left_shoulder_yaw", "right_shoulder_yaw",
    "left_elbow", "right_elbow",
    "left_wrist_roll", "right_wrist_roll",
    "left_wrist_pitch", "right_wrist_pitch",
    "left_wrist_yaw", "right_wrist_yaw",
)
MJC_TO_BYD = tuple(MJC_JOINT_NAMES.index(name) for name in BYD_JOINT_NAMES)

# This is the 28-joint representation consumed by BonesSeedG1SomaDataset.
# It is the compact SOMA body skeleton without Neck2 and Jaw.
SOMA_JOINT_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Head",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumbEnd",
    "LeftHandMiddleEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumbEnd",
    "RightHandMiddleEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)

# SOMA BVH uses y-up/+z-forward; the paired training files use z-up/+x-forward.
Y_UP_TO_Z_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


@dataclass
class BvhJoint:
    name: str
    parent: int
    offset_cm: np.ndarray
    channels: tuple[str, ...]
    channel_start: int


def _parse_bvh_hierarchy(lines: list[str]) -> tuple[list[BvhJoint], int]:
    joints: list[BvhJoint] = []
    stack: list[int | None] = []
    pending: int | None = None
    channel_count = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        keyword = fields[0]

        if keyword in ("ROOT", "JOINT"):
            parent = next((idx for idx in reversed(stack) if idx is not None), -1)
            joints.append(
                BvhJoint(
                    name=fields[1],
                    parent=parent,
                    offset_cm=np.zeros(3, dtype=np.float32),
                    channels=(),
                    channel_start=channel_count,
                )
            )
            pending = len(joints) - 1
        elif keyword == "End" and len(fields) > 1 and fields[1] == "Site":
            pending = None
        elif keyword == "{":
            stack.append(pending)
            pending = None
        elif keyword == "}":
            if not stack:
                raise ValueError("Unbalanced closing brace in BVH hierarchy")
            stack.pop()
        elif keyword == "OFFSET":
            current = stack[-1] if stack else None
            if current is not None:
                joints[current].offset_cm = np.asarray(fields[1:4], dtype=np.float32)
        elif keyword == "CHANNELS":
            current = stack[-1] if stack else None
            if current is None:
                raise ValueError("BVH CHANNELS entry is not attached to a joint")
            count = int(fields[1])
            channels = tuple(fields[2 : 2 + count])
            if len(channels) != count:
                raise ValueError(f"Invalid CHANNELS entry for {joints[current].name}")
            joints[current].channels = channels
            joints[current].channel_start = channel_count
            channel_count += count

    if stack:
        raise ValueError("Unbalanced opening brace in BVH hierarchy")
    if not joints or channel_count == 0:
        raise ValueError("BVH hierarchy contains no animated joints")
    return joints, channel_count


def _load_bvh_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return z-up root orientation and the 28 z-up SOMA joint positions."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        motion_line = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    except StopIteration as exc:
        raise ValueError(f"Missing MOTION section in {path}") from exc

    joints, channel_count = _parse_bvh_hierarchy(lines[:motion_line])
    frame_count = None
    frame_time = None
    data_start = None
    for i in range(motion_line + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith("Frames:"):
            frame_count = int(line.split(":", 1)[1].strip())
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split(":", 1)[1].strip())
            data_start = i + 1
            break
    if frame_count is None or frame_time is None or data_start is None:
        raise ValueError(f"Incomplete BVH motion header in {path}")

    values = np.fromstring(" ".join(lines[data_start:]), sep=" ", dtype=np.float32)
    expected = frame_count * channel_count
    if values.size != expected:
        raise ValueError(
            f"BVH channel count mismatch in {path}: expected {expected}, got {values.size}"
        )
    frames = values.reshape(frame_count, channel_count)

    measured_fps = 1.0 / frame_time
    if abs(measured_fps - SOURCE_FPS) > 0.5:
        raise ValueError(f"Expected {SOURCE_FPS} FPS BVH, found {measured_fps:.4f} FPS: {path}")

    joint_names = [joint.name for joint in joints]
    missing = [name for name in SOMA_JOINT_NAMES if name not in joint_names]
    if missing:
        raise ValueError(f"Missing SOMA joints in {path}: {missing}")

    num_frames = frames.shape[0]
    num_joints = len(joints)
    local_rot = np.broadcast_to(np.eye(3, dtype=np.float32), (num_frames, num_joints, 3, 3)).copy()
    local_pos = np.broadcast_to(
        np.stack([joint.offset_cm for joint in joints], axis=0)[None] * 0.01,
        (num_frames, num_joints, 3),
    ).copy()

    for joint_idx, joint in enumerate(joints):
        position_channels = [name for name in joint.channels if name.lower().endswith("position")]
        rotation_channels = [name for name in joint.channels if name.lower().endswith("rotation")]
        channel_to_index = {
            name: joint.channel_start + offset for offset, name in enumerate(joint.channels)
        }

        for channel in position_channels:
            axis = "XYZ".index(channel[0].upper())
            local_pos[:, joint_idx, axis] += frames[:, channel_to_index[channel]] * 0.01

        if rotation_channels:
            if len(rotation_channels) != 3:
                raise ValueError(
                    f"Expected three rotation channels for {joint.name} in {path}, "
                    f"found {rotation_channels}"
                )
            order = "".join(channel[0].upper() for channel in rotation_channels)
            euler_deg = frames[:, [channel_to_index[channel] for channel in rotation_channels]]
            local_rot[:, joint_idx] = Rotation.from_euler(
                order, euler_deg, degrees=True
            ).as_matrix().astype(np.float32)

    global_rot = np.empty_like(local_rot)
    global_pos = np.empty_like(local_pos)
    for joint_idx, joint in enumerate(joints):
        if joint.parent < 0:
            global_rot[:, joint_idx] = local_rot[:, joint_idx]
            global_pos[:, joint_idx] = local_pos[:, joint_idx]
        else:
            parent_rot = global_rot[:, joint.parent]
            global_rot[:, joint_idx] = parent_rot @ local_rot[:, joint_idx]
            global_pos[:, joint_idx] = (
                global_pos[:, joint.parent]
                + np.einsum("tij,tj->ti", parent_rot, local_pos[:, joint_idx])
            )

    selection = [joint_names.index(name) for name in SOMA_JOINT_NAMES]
    joints_y_up = global_pos[:, selection]
    joints_z_up = np.einsum("ij,tkj->tki", Y_UP_TO_Z_UP, joints_y_up).astype(np.float32)

    hips_idx = joint_names.index("Hips")
    root_rot_z_up = Y_UP_TO_Z_UP[None] @ global_rot[:, hips_idx]
    global_orient = Rotation.from_matrix(root_rot_z_up).as_rotvec().astype(np.float32)
    return global_orient, joints_z_up


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> int:
    normalized = {name.strip().lower(): idx for idx, name in enumerate(columns)}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(f"Missing CSV column; expected one of {candidates}")


def _load_g1_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig") as handle:
        columns = [item.strip() for item in handle.readline().strip().split(",")]
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if values.ndim == 1:
        values = values[None]
    if values.shape[1] != len(columns):
        raise ValueError(
            f"CSV width mismatch in {path}: header={len(columns)}, data={values.shape[1]}"
        )

    root_pos_indices = [
        _find_column(columns, (f"root_translate{axis}", f"root_position{axis}"))
        for axis in "XYZ"
    ]
    root_rot_indices = [
        _find_column(columns, (f"root_rotate{axis}", f"root_rotation{axis}"))
        for axis in "XYZ"
    ]
    root_pos = values[:, root_pos_indices] * 0.01
    root_euler_deg = values[:, root_rot_indices]
    root_xyzw = Rotation.from_euler("xyz", root_euler_deg, degrees=True).as_quat()
    root_wxyz = np.concatenate([root_xyzw[:, 3:4], root_xyzw[:, :3]], axis=-1)

    normalized = {name.strip().lower(): idx for idx, name in enumerate(columns)}
    joint_indices = []
    for joint_name in MJC_JOINT_NAMES:
        candidates = (
            f"{joint_name}_dof",
            f"{joint_name}_joint_dof",
            joint_name,
            f"{joint_name}_joint",
        )
        index = next((normalized[name] for name in candidates if name in normalized), None)
        if index is None:
            joint_indices = []
            break
        joint_indices.append(index)

    if not joint_indices:
        reserved = {0, *root_pos_indices, *root_rot_indices}
        joint_indices = [idx for idx in range(values.shape[1]) if idx not in reserved]
        if len(joint_indices) != len(MJC_JOINT_NAMES):
            raise ValueError(
                f"Expected {len(MJC_JOINT_NAMES)} G1 joint columns in {path}, "
                f"found {len(joint_indices)}"
            )

    dof_mjc = np.deg2rad(values[:, joint_indices]).astype(np.float32)
    dof_byd = dof_mjc[:, MJC_TO_BYD]
    return root_pos.astype(np.float32), root_wxyz.astype(np.float32), dof_byd


def _convert_pair(csv_path: Path, bvh_path: Path) -> dict[str, dict[str, np.ndarray | int]]:
    from hmr4d.dataset.pure_motion.g1_amass import fk_g1_pth_slice

    root_pos, root_rot, dof_pos = _load_g1_csv(csv_path)
    soma_global_orient, soma_joints = _load_bvh_motion(bvh_path)

    source_lengths = (len(root_pos), len(soma_joints))
    if abs(source_lengths[0] - source_lengths[1]) > 1:
        raise ValueError(
            f"Unpaired frame counts for {csv_path.stem}: G1={source_lengths[0]}, "
            f"SOMA={source_lengths[1]}"
        )
    source_length = min(source_lengths)
    step = SOURCE_FPS // TARGET_FPS
    frame_ids = np.arange(0, source_length, step, dtype=np.int64)

    root_pos = np.ascontiguousarray(root_pos[frame_ids])
    root_rot = np.ascontiguousarray(root_rot[frame_ids])
    dof_pos = np.ascontiguousarray(dof_pos[frame_ids])
    soma_global_orient = np.ascontiguousarray(soma_global_orient[frame_ids])
    soma_joints = np.ascontiguousarray(soma_joints[frame_ids])

    with torch.no_grad():
        body_pos, body_quat = fk_g1_pth_slice(
            torch.from_numpy(dof_pos),
            torch.from_numpy(root_pos),
            torch.from_numpy(root_rot),
            device=torch.device("cpu"),
        )

    return {
        "g1": {
            "fps": TARGET_FPS,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "body_pos_w_full": body_pos.cpu().numpy().astype(np.float32),
            "body_quat_w_full": body_quat.cpu().numpy().astype(np.float32),
        },
        "soma": {
            "fps": TARGET_FPS,
            "global_orient": soma_global_orient,
            "joints": soma_joints,
        },
    }


def _safe_session_name(relative_parent: Path) -> str:
    if not relative_parent.parts:
        return "root"
    name = "__".join(relative_parent.parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _process_session(
    session_name: str,
    pairs: list[tuple[str, str]],
    samples_dir: str,
    manifests_dir: str,
    overwrite: bool,
    skip_errors: bool,
    compression: int,
) -> dict:
    samples_path = Path(samples_dir) / f"{session_name}.pkl"
    manifest_path = Path(manifests_dir) / f"{session_name}.json"
    if samples_path.exists() and manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_pair_count") != len(pairs):
            raise RuntimeError(
                f"Existing session {session_name} contains a different source-pair count; "
                "rerun with --overwrite"
            )
        return manifest
    if samples_path.exists() and not overwrite:
        raise FileExistsError(
            f"{samples_path} exists without its manifest; rerun with --overwrite"
        )

    payloads = {}
    records = []
    failures = []
    for csv_name, bvh_name in pairs:
        csv_path = Path(csv_name)
        bvh_path = Path(bvh_name)
        sequence_name = csv_path.stem
        try:
            payload = _convert_pair(csv_path, bvh_path)
        except Exception as exc:
            if not skip_errors:
                raise RuntimeError(f"Failed to convert {sequence_name}: {exc}") from exc
            failures.append({"sequence": sequence_name, "error": str(exc)})
            continue
        payloads[sequence_name] = payload
        records.append(
            {"seq_name": sequence_name, "length": int(payload["g1"]["root_pos"].shape[0])}
        )

    if not payloads:
        raise RuntimeError(f"No valid motion pairs were produced for session {session_name}")

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_samples = samples_path.with_name(f".{samples_path.name}.tmp.{os.getpid()}")
    tmp_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    joblib.dump(payloads, tmp_samples, compress=compression)
    os.replace(tmp_samples, samples_path)

    manifest = {
        "session": session_name,
        "file_name": samples_path.name,
        "source_pair_count": len(pairs),
        "records": records,
        "failures": failures,
    }
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    return manifest


def _build_index(output_root: Path, manifests: list[dict]) -> None:
    records = []
    for manifest in manifests:
        file_name = manifest["file_name"]
        file_stem = Path(file_name).stem
        for record in manifest["records"]:
            records.append(
                {
                    "key": f"{file_stem}/{record['seq_name']}",
                    "file_name": file_name,
                    "seq_name": record["seq_name"],
                    "length": int(record["length"]),
                }
            )
    records.sort(key=lambda item: item["key"])

    samples = sorted((output_root / "samples").glob("*.pkl"))
    fingerprint = [
        {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in samples
    ]
    index = {"version": 1, "fingerprint": fingerprint, "records": records}
    index_path = output_root / "index.json"
    tmp_path = index_path.with_name(f".{index_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(index), encoding="utf-8")
    os.replace(tmp_path, index_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Extracted official BONES-SEED root containing g1/ and soma_proportional/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output bones_seed directory containing samples/ and index.json.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of session workers.")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N pairs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing session files.")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Record invalid pairs in conversion_report.json and continue.",
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=3,
        choices=range(0, 10),
        metavar="0-9",
        help="Joblib compression level for session files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if SOURCE_FPS % TARGET_FPS != 0:
        raise RuntimeError("SOURCE_FPS must be divisible by TARGET_FPS")
    if args.workers < 1:
        raise ValueError("--workers must be at least one")

    source_root = args.source.resolve()
    output_root = args.output.resolve()
    g1_root = source_root / "g1" / "csv"
    soma_root = source_root / "soma_proportional" / "bvh"
    if not g1_root.is_dir():
        raise FileNotFoundError(f"Missing extracted G1 directory: {g1_root}")
    if not soma_root.is_dir():
        raise FileNotFoundError(f"Missing extracted SOMA directory: {soma_root}")

    csv_files = sorted(g1_root.rglob("*.csv"))
    if args.limit is not None:
        csv_files = csv_files[: args.limit]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {g1_root}")

    grouped: dict[str, list[tuple[str, str]]] = {}
    missing_bvh = []
    for csv_path in csv_files:
        relative = csv_path.relative_to(g1_root)
        bvh_path = (soma_root / relative).with_suffix(".bvh")
        if not bvh_path.is_file():
            missing_bvh.append(str(relative.with_suffix(".bvh")))
            continue
        session_name = _safe_session_name(relative.parent)
        grouped.setdefault(session_name, []).append((str(csv_path), str(bvh_path)))

    if missing_bvh and not args.skip_errors:
        examples = ", ".join(missing_bvh[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_bvh)} proportional SOMA pairs; examples: {examples}"
        )
    if not grouped:
        raise RuntimeError("No matching G1/SOMA motion pairs were found")

    samples_dir = output_root / "samples"
    manifests_dir = output_root / "manifests"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    tasks = sorted(grouped.items())
    manifests = []
    if args.workers == 1:
        for index, (session_name, pairs) in enumerate(tasks, start=1):
            print(f"[{index}/{len(tasks)}] {session_name}: {len(pairs)} pairs", flush=True)
            manifests.append(
                _process_session(
                    session_name,
                    pairs,
                    str(samples_dir),
                    str(manifests_dir),
                    args.overwrite,
                    args.skip_errors,
                    args.compression,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _process_session,
                    session_name,
                    pairs,
                    str(samples_dir),
                    str(manifests_dir),
                    args.overwrite,
                    args.skip_errors,
                    args.compression,
                ): session_name
                for session_name, pairs in tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                session_name = futures[future]
                manifest = future.result()
                manifests.append(manifest)
                print(f"[{index}/{len(tasks)}] finished {session_name}", flush=True)

    manifests.sort(key=lambda item: item["session"])
    _build_index(output_root, manifests)
    failures = [failure for manifest in manifests for failure in manifest["failures"]]
    report = {
        "source": str(source_root),
        "output": str(output_root),
        "sessions": len(manifests),
        "converted_sequences": sum(len(item["records"]) for item in manifests),
        "failures": failures,
        "missing_bvh": missing_bvh,
    }
    (output_root / "conversion_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, indent=2))


if __name__ == "__main__":
    main()
