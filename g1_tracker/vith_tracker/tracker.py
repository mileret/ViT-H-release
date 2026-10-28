"""G1 motion tracking with the SONIC deploy policy and MuJoCo."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d

from .observation import GlobalObservationMixin
from .runtime import (
    ACTION_SCALE,
    ARMATURE_4010,
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    DEFAULT_ANGLES,
    HW_DOF,
    ISAACLAB_TO_MUJOCO,
    KD,
    KP,
    MUJOCO_TO_ISAACLAB,
    G1SonicEnv,
    normalize_quat,
    pd_control,
    quat_conj,
    quat_rotate,
)
from .success import SONIC_END_EFFECTOR_BODY_NAMES, SonicStandardizedSuccessAccumulator

BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
)
END_EFFECTOR_INDICES = tuple(BODY_NAMES.index(name) for name in SONIC_END_EFFECTOR_BODY_NAMES)

# These are the four spherical contact proxies attached to each ankle-roll
# link in g1_29dof_rev_1_0.xml. They intentionally match Isaac's
# _compute_initial_support_grounding_offsets() implementation exactly.
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_CONTACT_SPHERE_CENTERS_M = np.asarray(
    (
        (-0.05, 0.025, -0.03),
        (-0.05, -0.025, -0.03),
        (0.12, 0.03, -0.03),
        (0.12, -0.03, -0.03),
    ),
    dtype=np.float64,
)
FOOT_CONTACT_SPHERE_RADIUS_M = 0.005
ISAAC_SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
TRACKER_PROTOCOL_VERSION = 4
# Collision primitives in robot_description/urdf/g1/main.urdf. Isaac's URDF
# importer converts these cylinders to capsules. Both feet use the same seven
# longitudinal primitives; their lowest point is z=-0.035 m, consistent with
# the support-grounding proxy above.
ISAAC_FOOT_CAPSULES = (
    ((0.075, -0.026, -0.025), 0.01, 0.05, np.pi / 2.0),
    ((0.0395, -0.018, -0.025), 0.008, 0.167, -np.pi / 2.0),
    ((0.039, -0.01, -0.025), 0.01, 0.182, -np.pi / 2.0),
    ((0.039, 0.0, -0.025), 0.01, 0.186, -np.pi / 2.0),
    ((0.039, 0.01, -0.025), 0.01, 0.182, -np.pi / 2.0),
    ((0.0395, 0.018, -0.025), 0.008, 0.167, -np.pi / 2.0),
    ((0.075, 0.026, -0.025), 0.01, 0.05, np.pi / 2.0),
)

def isaac_effort_limit_for_joint(joint_name: str) -> float:
    """Return the Isaac G1 actuator effort limit for one 29-DoF joint."""
    if "hip_pitch" in joint_name or "hip_roll" in joint_name or "knee" in joint_name:
        return 139.0
    if "hip_yaw" in joint_name or joint_name == "waist_yaw_joint":
        return 88.0
    if "ankle_" in joint_name or joint_name in {
        "waist_roll_joint",
        "waist_pitch_joint",
    }:
        return 50.0
    if "wrist_pitch" in joint_name or "wrist_yaw" in joint_name:
        return 5.0
    return 25.0


def isaac_armature_for_joint(joint_name: str) -> float:
    """Return the Isaac G1 implicit-actuator armature for one joint."""
    if "hip_pitch" in joint_name or "hip_roll" in joint_name or "knee" in joint_name:
        return ARMATURE_7520_22
    if "hip_yaw" in joint_name or joint_name == "waist_yaw_joint":
        return ARMATURE_7520_14
    if "ankle_" in joint_name or joint_name in {
        "waist_roll_joint",
        "waist_pitch_joint",
    }:
        return 2.0 * ARMATURE_5020
    if "wrist_pitch" in joint_name or "wrist_yaw" in joint_name:
        return ARMATURE_4010
    return ARMATURE_5020


def load_isaac_aligned_mujoco_model(
    xml_path: Path,
) -> tuple[mujoco.MjModel, dict[str, int]]:
    """Compile a MuJoCo model with all directly mappable Isaac parameters.

    PD stiffness/damping are already applied explicitly by ``pd_control`` with
    the same Isaac gains. Consequently passive MuJoCo joint damping and dry
    friction must be zero to avoid applying extra resistance a second time.
    Isaac's URDF importer also replaces collision cylinders with capsules.
    """
    spec = mujoco.MjSpec.from_file(str(xml_path))
    hinge_joints = [
        joint for joint in spec.joints if joint.type == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if len(hinge_joints) != 29:
        raise ValueError(f"expected 29 hinge joints, got {len(hinge_joints)}")
    actuators_by_target = {
        actuator.target: actuator
        for actuator in spec.actuators
        if actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT
    }
    for joint in hinge_joints:
        joint_name = joint.name
        effort_limit = isaac_effort_limit_for_joint(joint_name)
        joint.armature = isaac_armature_for_joint(joint_name)
        joint.damping = np.zeros_like(joint.damping)
        joint.frictionloss = 0.0
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.actfrcrange = np.asarray((-effort_limit, effort_limit))
        actuator = actuators_by_target.get(joint_name)
        if actuator is None:
            raise ValueError(f"no actuator found for {joint_name}")
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator.ctrlrange = np.asarray((-effort_limit, effort_limit))

    removed_foot_spheres = 0
    added_foot_capsules = 0
    for body_name in FOOT_BODY_NAMES:
        body = spec.body(body_name)
        old_collisions = [
            geom
            for geom in body.geoms
            if geom.contype != 0 or geom.conaffinity != 0
        ]
        if len(old_collisions) != 4 or any(
            geom.type != mujoco.mjtGeom.mjGEOM_SPHERE
            for geom in old_collisions
        ):
            raise ValueError(
                f"expected four support spheres on {body_name}, got "
                f"{[(geom.type, geom.contype, geom.conaffinity) for geom in old_collisions]}"
            )
        template = old_collisions[0]
        contact_properties = {
            "contype": int(template.contype),
            "conaffinity": int(template.conaffinity),
            "condim": int(template.condim),
            "priority": int(template.priority),
            "friction": np.array(template.friction, copy=True),
            "solmix": float(template.solmix),
            "solref": np.array(template.solref, copy=True),
            "solimp": np.array(template.solimp, copy=True),
            "margin": float(template.margin),
            "gap": float(template.gap),
            "group": int(template.group),
            "rgba": np.array(template.rgba, copy=True),
        }
        for geom in old_collisions:
            spec.delete(geom)
            removed_foot_spheres += 1
        for capsule_index, (position, radius, length, pitch) in enumerate(
            ISAAC_FOOT_CAPSULES
        ):
            body.add_geom(
                name=f"{body_name}_isaac_capsule_{capsule_index}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                pos=position,
                quat=(np.cos(pitch / 2.0), 0.0, np.sin(pitch / 2.0), 0.0),
                size=(radius, length / 2.0, 0.0),
                **contact_properties,
            )
            added_foot_capsules += 1

    collision_cylinders = [
        geom
        for geom in spec.geoms
        if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER
        and (geom.contype != 0 or geom.conaffinity != 0)
    ]
    for geom in collision_cylinders:
        geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE

    model = spec.compile()
    return model, {
        "aligned_hinge_joints": int(len(hinge_joints)),
        "cylinders_replaced_with_capsules": int(len(collision_cylinders)),
        "support_spheres_removed": int(removed_foot_spheres),
        "isaac_foot_capsules_added": int(added_foot_capsules),
    }


@dataclass
class Motion:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    body_pos: np.ndarray

    @property
    def timesteps(self) -> int:
        return int(self.joint_pos.shape[0])

def normalize_quaternions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-10):
        raise ValueError("zero quaternion")
    return values / norms


def slerp_rows(values: np.ndarray, old_t: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    values = normalize_quaternions(values)
    indices = np.searchsorted(old_t, new_t, side="right") - 1
    indices = np.clip(indices, 0, len(old_t) - 2)
    fractions = (new_t - old_t[indices]) / np.maximum(
        old_t[indices + 1] - old_t[indices], 1.0e-12
    )
    first = values[indices]
    second = values[indices + 1].copy()
    dots = np.sum(first * second, axis=-1)
    second[dots < 0.0] *= -1.0
    dots = np.abs(dots).clip(-1.0, 1.0)
    result = np.empty_like(first)
    linear = dots > 0.9995
    result[linear] = (
        first[linear] + fractions[linear, None] * (second[linear] - first[linear])
    )
    nonlinear = ~linear
    if np.any(nonlinear):
        angles = np.arccos(dots[nonlinear])
        denominator = np.sin(angles)
        weight_first = np.sin((1.0 - fractions[nonlinear]) * angles) / denominator
        weight_second = np.sin(fractions[nonlinear] * angles) / denominator
        result[nonlinear] = (
            weight_first[:, None] * first[nonlinear]
            + weight_second[:, None] * second[nonlinear]
        )
    return normalize_quaternions(result).astype(np.float32)


def interpolate_linear(values: np.ndarray, old_t: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(len(old_t), -1)
    out = np.stack([np.interp(new_t, old_t, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return out.reshape((len(new_t),) + values.shape[1:]).astype(np.float32)


def isaac_resample_times(
    frame_count: int,
    input_fps: float,
    output_fps: float,
) -> np.ndarray:
    """Match Isaac MotionLib's endpoint-exclusive target timeline."""
    if frame_count < 2 or input_fps <= 0.0 or output_fps <= 0.0:
        raise ValueError(
            f"invalid resampling request: frames={frame_count} "
            f"input_fps={input_fps} output_fps={output_fps}"
        )
    # Isaac constructs both duration and target times as float32 tensors.
    # Preserve that numerical convention before promoting to float64 for the
    # NumPy interpolation helpers below.
    duration = np.float32((frame_count - 1) / input_fps)
    times = np.arange(
        np.float32(0.0),
        duration,
        np.float32(1.0 / output_fps),
        dtype=np.float32,
    )
    return times.astype(np.float64)


def quaternion_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Vectorized Hamilton product for matching ``[..., 4]`` arrays."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_w, first_xyz = first[..., :1], first[..., 1:]
    second_w, second_xyz = second[..., :1], second[..., 1:]
    return np.concatenate(
        (
            first_w * second_w - np.sum(first_xyz * second_xyz, axis=-1, keepdims=True),
            first_w * second_xyz
            + second_w * first_xyz
            + np.cross(first_xyz, second_xyz),
        ),
        axis=-1,
    )


def reference_angular_velocity_wxyz(
    quaternions_wxyz: np.ndarray,
    time_delta: float,
) -> np.ndarray:
    """Match Isaac FK's world angular velocity and Gaussian smoothing."""
    quaternions = normalize_quaternions(quaternions_wxyz)
    conjugates = quaternions[:-1].copy()
    conjugates[:, 1:] *= -1.0
    differences = normalize_quaternions(
        quaternion_multiply_wxyz(quaternions[1:], conjugates)
    )
    differences[differences[:, 0] < 0.0] *= -1.0
    vector_norm = np.linalg.norm(differences[:, 1:], axis=-1)
    angles = 2.0 * np.arctan2(vector_norm, np.clip(differences[:, 0], -1.0, 1.0))
    axes = np.zeros_like(differences[:, 1:])
    nonzero = vector_norm > 1.0e-10
    axes[nonzero] = differences[nonzero, 1:] / vector_norm[nonzero, None]
    angular_velocity = np.zeros((len(quaternions), 3), dtype=np.float64)
    angular_velocity[:-1] = axes * angles[:, None] / time_delta
    return gaussian_filter1d(
        angular_velocity, sigma=2, axis=0, mode="nearest"
    ).astype(np.float32)


def resample_motion(
    values: dict[str, np.ndarray],
    input_fps: float,
    output_fps: float,
    sample_index: int = 0,
) -> Motion:
    joint_pos = np.asarray(values["joint_pos"], dtype=np.float32)
    if joint_pos.ndim == 3:
        if not 0 <= sample_index < joint_pos.shape[0]:
            raise IndexError(sample_index)
        joint_pos = joint_pos[sample_index]
    root_pos = np.asarray(values["root_pos_w"], dtype=np.float32)[sample_index]
    root_quat = np.asarray(values["root_quat_w"], dtype=np.float32)[sample_index]
    body_pos = np.asarray(values["body_pos_w"], dtype=np.float32)[sample_index]
    frame_count = len(joint_pos)
    old_t = np.arange(frame_count, dtype=np.float64) / input_fps
    new_t = isaac_resample_times(frame_count, input_fps, output_fps)
    joint_50 = interpolate_linear(joint_pos, old_t, new_t)
    if len(joint_50) < 3:
        raise ValueError("at least three resampled frames are required for velocity")
    joint_diff = np.diff(joint_50.astype(np.float64), axis=0) * output_fps
    # Match torch_humanoid_batch.py exactly: the final DoF velocity repeats
    # the penultimate finite difference (dof_vel[:, -2:-1]), not the last.
    joint_vel = np.concatenate((joint_diff, joint_diff[-2:-1]), axis=0).astype(
        np.float32
    )
    root_pos_50 = interpolate_linear(root_pos, old_t, new_t)
    root_quat_50 = slerp_rows(root_quat, old_t, new_t)
    root_lin_vel_w = gaussian_filter1d(
        np.gradient(root_pos_50.astype(np.float64), axis=0) * output_fps,
        sigma=2,
        axis=0,
        mode="nearest",
    ).astype(np.float32)
    root_ang_vel_w = reference_angular_velocity_wxyz(
        root_quat_50, 1.0 / output_fps
    )
    return Motion(
        joint_pos=joint_50,
        joint_vel=joint_vel,
        root_pos=root_pos_50,
        root_quat=root_quat_50,
        root_lin_vel_w=root_lin_vel_w,
        root_ang_vel_w=root_ang_vel_w,
        body_pos=interpolate_linear(body_pos, old_t, new_t),
    )


class InitialSupportGrounder:
    """Apply Isaac's per-motion initial support-grounding convention."""

    def __init__(self, xml_path: Path):
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 36:
            raise ValueError(
                f"grounding FK proxy must have 7+29 qpos, got nq={self.model.nq}"
            )
        self.foot_body_ids = np.asarray(
            [int(self.model.body(name).id) for name in FOOT_BODY_NAMES],
            dtype=np.int64,
        )
        self.body_ids = np.asarray(
            [int(self.model.body(name).id) for name in BODY_NAMES], dtype=np.int64
        )
        hinge_joint_ids = np.flatnonzero(
            self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
        )
        if hinge_joint_ids.shape != (29,):
            raise ValueError(
                f"grounding FK proxy must have 29 hinge joints, got {len(hinge_joint_ids)}"
            )
        hard_limits_mujoco = np.asarray(
            self.model.jnt_range[hinge_joint_ids], dtype=np.float64
        )
        centers = hard_limits_mujoco.mean(axis=-1)
        half_ranges = (
            np.diff(hard_limits_mujoco, axis=-1).reshape(-1)
            * 0.5
            * ISAAC_SOFT_JOINT_POS_LIMIT_FACTOR
        )
        soft_limits_mujoco = np.stack(
            (centers - half_ranges, centers + half_ranges), axis=-1
        )
        self.soft_joint_limits_isaaclab = soft_limits_mujoco[
            MUJOCO_TO_ISAACLAB
        ]

    def forward_body_positions(self, motion: Motion) -> np.ndarray:
        """Rebuild 50 Hz body positions after joint/root interpolation.

        Isaac interpolates root translation and joint rotations first, then
        runs G1 FK. Linearly interpolating already-computed 30 Hz body points
        is not the same operation for rotating articulated links.
        """
        positions = np.empty((motion.timesteps, len(BODY_NAMES), 3), dtype=np.float32)
        for frame in range(motion.timesteps):
            self.data.qpos[:3] = np.asarray(motion.root_pos[frame], dtype=np.float64)
            self.data.qpos[3:7] = normalize_quat(
                np.asarray(motion.root_quat[frame], dtype=np.float64)
            )
            self.data.qpos[7:] = np.asarray(
                motion.joint_pos[frame, ISAACLAB_TO_MUJOCO], dtype=np.float64
            )
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            positions[frame] = self.data.xpos[self.body_ids]
        return positions

    def initial_lowest_contact_z(self, motion: Motion) -> float:
        """Return the lowest initial foot-contact-sphere point in world Z."""
        self.data.qpos[:3] = np.asarray(motion.root_pos[0], dtype=np.float64)
        self.data.qpos[3:7] = normalize_quat(
            np.asarray(motion.root_quat[0], dtype=np.float64)
        )
        self.data.qpos[7:] = np.asarray(
            motion.joint_pos[0, ISAACLAB_TO_MUJOCO], dtype=np.float64
        )
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        foot_positions = np.asarray(
            self.data.xpos[self.foot_body_ids], dtype=np.float64
        )
        foot_rotations = np.asarray(
            self.data.xmat[self.foot_body_ids], dtype=np.float64
        ).reshape(2, 3, 3)
        contact_centers_world = foot_positions[:, None, :] + np.einsum(
            "fij,pj->fpi", foot_rotations, FOOT_CONTACT_SPHERE_CENTERS_M
        )
        return float(
            contact_centers_world[..., 2].min() - FOOT_CONTACT_SPHERE_RADIUS_M
        )

    def ground(self, motion: Motion) -> tuple[Motion, float]:
        """Shift root and body reference trajectories so initial support is Z=0."""
        offset = -self.initial_lowest_contact_z(motion)
        root_pos = np.array(motion.root_pos, dtype=np.float32, copy=True)
        body_pos = self.forward_body_positions(motion)
        root_pos[:, 2] += offset
        body_pos[..., 2] += offset
        return (
            Motion(
                joint_pos=motion.joint_pos,
                joint_vel=motion.joint_vel,
                root_pos=root_pos,
                root_quat=motion.root_quat,
                root_lin_vel_w=motion.root_lin_vel_w,
                root_ang_vel_w=motion.root_ang_vel_w,
                body_pos=body_pos,
            ),
            float(offset),
        )

    def clip_initial_joint_pos(
        self, joint_pos_isaaclab: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clip one IsaacLab-ordered pose to its 0.9-factor soft limits."""
        joint_pos = np.asarray(joint_pos_isaaclab, dtype=np.float64)
        if joint_pos.shape != (29,):
            raise ValueError(f"expected initial joint pose [29], got {joint_pos.shape}")
        clipped = np.clip(
            joint_pos,
            self.soft_joint_limits_isaaclab[:, 0],
            self.soft_joint_limits_isaaclab[:, 1],
        )
        return clipped.astype(np.float32), (clipped - joint_pos).astype(np.float32)


class DeployPolicy:
    def __init__(self, path: Path, threads: int):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        if self.input.shape[-1] != 1570 or self.output.shape[-1] != 29:
            raise ValueError(
                f"deploy policy must be 1570->29, got {self.input.shape}->{self.output.shape}"
            )

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.session.run(
            [self.output.name],
            {self.input.name: observation.astype(np.float32, copy=False)},
        )[0]


class DeployRolloutRunner(GlobalObservationMixin):
    def __init__(self, xml_path: Path, policy: DeployPolicy, control_hz: float):
        self.dt = 1.0 / control_hz
        self.sim_substeps = max(1, round(self.dt / 0.001))
        # G1SonicEnv provides the deploy observation/control plumbing. Replace
        # its stock model with a model compiled from the Isaac-aligned MjSpec
        # before any state is initialized or stepped.
        self.env = G1SonicEnv(xml_path, viewer=False)
        aligned_model, self.isaac_model_alignment = (
            load_isaac_aligned_mujoco_model(xml_path)
        )
        self.env.model = aligned_model
        self.env.data = mujoco.MjData(aligned_model)
        self.env.model.opt.timestep = 0.001
        self.env.ctrl_start = self.env._resolve_policy_actuator_start()
        self.policy = policy
        self.body_ids = np.asarray(
            [int(self.env.model.body(name).id) for name in BODY_NAMES], dtype=np.int64
        )

    def close(self) -> None:
        self.env.close()

    def _read_state(self) -> None:
        self.base_pos = self.env.data.qpos[:3].copy()
        self.base_quat = normalize_quat(self.env.data.qpos[3:7].copy())
        self.base_ang_vel = self.env.data.qvel[3:6].copy()
        self.joint_pos_mj = self.env.data.qpos[7:].copy()
        self.joint_vel_mj = self.env.data.qvel[6:].copy()

    def compute_deploy_observation(self) -> np.ndarray:
        tokenizer = np.concatenate(
            [self._command_multi_future(), self._motion_anchor_orientation_10frame_step5()]
        ).astype(np.float32)
        actor = self._normal_actor_observation()
        result = np.concatenate([tokenizer, actor]).astype(np.float32)
        if result.shape != (1570,):
            raise RuntimeError(f"deploy observation has shape {result.shape}")
        return result.reshape(1, -1)

    def _target_from_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        target = DEFAULT_ANGLES.copy()
        for index in range(HW_DOF):
            target[index] = (
                DEFAULT_ANGLES[index]
                + action[ISAACLAB_TO_MUJOCO[index]] * ACTION_SCALE[index]
            )
        return target

    def _initialize_state(
        self,
        motion: Motion,
        initial_joint_pos_isaaclab: np.ndarray,
    ) -> None:
        """Reset from Isaac-equivalent reference position and velocity state."""
        initial_joint_pos_isaaclab = np.asarray(
            initial_joint_pos_isaaclab, dtype=np.float64
        )
        if initial_joint_pos_isaaclab.shape != (29,):
            raise ValueError(
                f"expected clipped initial joint pose [29], got "
                f"{initial_joint_pos_isaaclab.shape}"
            )
        root_quat = normalize_quat(motion.root_quat[0].astype(np.float64))
        self.env.data.qpos[:3] = motion.root_pos[0].astype(np.float64)
        self.env.data.qpos[3:7] = root_quat
        self.env.data.qpos[7:] = initial_joint_pos_isaaclab[
            ISAACLAB_TO_MUJOCO
        ]

        # MuJoCo free-joint linear velocity is world-frame, while its angular
        # qvel is body-frame. Isaac's reference angular velocity is world-frame.
        self.env.data.qvel[:3] = motion.root_lin_vel_w[0].astype(np.float64)
        self.env.data.qvel[3:6] = quat_rotate(
            quat_conj(root_quat), motion.root_ang_vel_w[0].astype(np.float64)
        )
        self.env.data.qvel[6:] = motion.joint_vel[
            0, ISAACLAB_TO_MUJOCO
        ].astype(np.float64)
        mujoco.mj_forward(self.env.model, self.env.data)

    def rollout(
        self,
        motion: Motion,
        initial_joint_pos_isaaclab: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], bool, int | None]:
        self.motion = motion
        self.current_frame = 0
        self._initialize_state(motion, initial_joint_pos_isaaclab)
        self._read_state()
        self.init_base_pos = self.base_pos.copy()
        self.init_base_quat = self.base_quat.copy()
        self.init_ref_root_pos = motion.root_pos[0].astype(np.float64).copy()
        self.init_ref_root_quat = normalize_quat(motion.root_quat[0].astype(np.float64))
        self.last_action = np.zeros(HW_DOF, dtype=np.float64)
        self.history = deque(maxlen=10)
        for _ in range(10):
            self._append_history()

        frames = motion.timesteps
        result = {
            "joint_pos": np.empty((frames, 29), dtype=np.float32),
            "root_pos_w": np.empty((frames, 3), dtype=np.float32),
            "root_quat_w": np.empty((frames, 4), dtype=np.float32),
            "body_pos_w": np.empty((frames, 30, 3), dtype=np.float32),
        }
        success = SonicStandardizedSuccessAccumulator()
        nonfinite_frame: int | None = None
        for frame in range(frames):
            self.current_frame = frame
            self._read_state()
            self._append_history()
            robot_body_pos = np.asarray(self.env.data.xpos[self.body_ids], dtype=np.float64)
            robot_root_quat = normalize_quat(
                np.asarray(self.env.data.xquat[self.body_ids[0]], dtype=np.float64)
            )
            result["joint_pos"][frame] = self.joint_pos_mj[MUJOCO_TO_ISAACLAB]
            result["root_pos_w"][frame] = robot_body_pos[0]
            result["root_quat_w"][frame] = robot_root_quat
            result["body_pos_w"][frame] = robot_body_pos
            success.update(
                reference_anchor_position_w=motion.body_pos[frame, 0],
                robot_anchor_position_w=robot_body_pos[0],
                reference_anchor_quaternion_wxyz=motion.root_quat[frame],
                robot_anchor_quaternion_wxyz=robot_root_quat,
                reference_end_effector_positions_w=motion.body_pos[frame, list(END_EFFECTOR_INDICES)],
                robot_end_effector_positions_w=robot_body_pos[list(END_EFFECTOR_INDICES)],
            )
            action = self.policy.act(self.compute_deploy_observation()).reshape(-1)
            if not np.all(np.isfinite(action)):
                nonfinite_frame = frame
                break
            self.last_action = action.astype(np.float64)
            target = self._target_from_action(action)
            for _ in range(self.sim_substeps):
                tau = pd_control(
                    target, self.env.data.qpos[7:], KP,
                    np.zeros(HW_DOF), self.env.data.qvel[6:], KD,
                )
                self.env.data.ctrl[:] = 0.0
                start = self.env.ctrl_start
                self.env.data.ctrl[start : start + HW_DOF] = tau
                mujoco.mj_step(self.env.model, self.env.data)
        if nonfinite_frame is not None:
            for key in result:
                result[key][nonfinite_frame + 1 :] = result[key][nonfinite_frame]
        return result, bool(success.is_success and nonfinite_frame is None), success.first_failure_frame_index


def resample_rollout(
    rollout: dict[str, np.ndarray], output_frames: int, control_hz: float, output_fps: float
) -> dict[str, np.ndarray]:
    old_t = np.arange(len(rollout["joint_pos"]), dtype=np.float64) / control_hz
    new_t = np.minimum(
        np.arange(output_frames, dtype=np.float64) / output_fps, old_t[-1]
    )
    return {
        "joint_pos": interpolate_linear(rollout["joint_pos"], old_t, new_t)[None],
        "root_pos_w": interpolate_linear(rollout["root_pos_w"], old_t, new_t)[None],
        "root_quat_w": slerp_rows(rollout["root_quat_w"], old_t, new_t)[None],
        "body_pos_w": interpolate_linear(rollout["body_pos_w"], old_t, new_t)[None],
    }


def atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _sample_video_path(path: Path, sample_index: int, sample_count: int) -> Path:
    if sample_count == 1:
        return path
    return path.with_name(f"{path.stem}_sample_{sample_index:03d}{path.suffix}")


def render_rollout_video(
    rollout: dict[str, np.ndarray],
    xml_path: Path,
    output_path: Path,
    fps: float,
    width: int = 640,
    height: int = 480,
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"video dimensions must be positive, got {width}x{height}")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("video output must use the .mp4 extension")

    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError(
            "video rendering requires imageio and imageio-ffmpeg"
        ) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    model, _ = load_isaac_aligned_mujoco_model(xml_path)
    data = mujoco.MjData(model)
    renderer = None
    writer = None
    failed = False
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.azimuth = 135.0
        camera.elevation = -15.0
        camera.distance = 3.0
        writer = imageio.get_writer(
            temporary,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        frames = int(rollout["joint_pos"].shape[0])
        for frame in range(frames):
            data.qpos[:3] = rollout["root_pos_w"][frame]
            data.qpos[3:7] = normalize_quat(rollout["root_quat_w"][frame])
            data.qpos[7:] = rollout["joint_pos"][frame, ISAACLAB_TO_MUJOCO]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    except Exception:
        failed = True
        raise
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if failed and temporary.exists():
            temporary.unlink()
    os.replace(temporary, output_path)

REQUIRED_INPUT_FIELDS = ("joint_pos", "root_pos_w", "root_quat_w", "body_pos_w")


def _batched_payload(values: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int, int]:
    missing = [key for key in REQUIRED_INPUT_FIELDS if key not in values]
    if missing:
        raise KeyError(f"missing required fields: {', '.join(missing)}")
    result = {key: np.asarray(values[key]) for key in REQUIRED_INPUT_FIELDS}
    expected_tail = {
        "joint_pos": (29,),
        "root_pos_w": (3,),
        "root_quat_w": (4,),
        "body_pos_w": (30, 3),
    }
    for key, tail in expected_tail.items():
        value = result[key]
        unbatched_rank = len(tail) + 1
        if value.ndim == unbatched_rank:
            value = value[None]
        if value.ndim != unbatched_rank + 1 or value.shape[2:] != tail:
            raise ValueError(
                f"{key} must have shape [T, {', '.join(map(str, tail))}] or "
                f"[S, T, {', '.join(map(str, tail))}], got {value.shape}"
            )
        result[key] = value.astype(np.float32, copy=False)
    shapes = {value.shape[:2] for value in result.values()}
    if len(shapes) != 1:
        raise ValueError(f"inconsistent sample/frame dimensions: {sorted(shapes)}")
    samples, frames = next(iter(shapes))
    if frames < 3:
        raise ValueError("at least three input frames are required")
    return result, samples, frames


def _resolve_fps(values: dict[str, np.ndarray], requested: float | None) -> float:
    if requested is not None:
        fps = float(requested)
    elif "fps" in values:
        fps = float(np.asarray(values["fps"]).reshape(()))
    else:
        raise ValueError("input FPS is required via --input-fps or an NPZ 'fps' scalar")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid input FPS: {fps}")
    return fps


def track_file(
    input_path: Path,
    output_path: Path,
    policy_onnx: Path,
    xml_path: Path,
    grounding_xml: Path,
    input_fps: float | None = None,
    control_hz: float = 50.0,
    ort_threads: int = 1,
    render_video: bool = False,
    video_output: Path | None = None,
    video_width: int = 640,
    video_height: int = 480,
) -> dict[str, np.ndarray]:
    paths = [Path(path).expanduser().resolve() for path in (
        input_path, policy_onnx, xml_path, grounding_xml
    )]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_path, policy_onnx, xml_path, grounding_xml = paths
    output_path = Path(output_path).expanduser().resolve()
    if video_output is not None and not render_video:
        raise ValueError("video_output requires render_video=True")
    with np.load(input_path, allow_pickle=False) as payload:
        raw_values = {key: np.asarray(payload[key]).copy() for key in payload.files}
    fps = _resolve_fps(raw_values, input_fps)
    values, samples, frames = _batched_payload(raw_values)

    policy = DeployPolicy(policy_onnx, ort_threads)
    runner = DeployRolloutRunner(xml_path, policy, control_hz)
    grounder = InitialSupportGrounder(grounding_xml)
    collected = {key: [] for key in REQUIRED_INPUT_FIELDS}
    successes = []
    failures = []
    grounding_offsets = []
    initial_root_linear = []
    initial_root_angular = []
    initial_joint_velocity = []
    clip_counts = []
    clip_maxima = []
    try:
        for sample_index in range(samples):
            reference = resample_motion(values, fps, control_hz, sample_index)
            reference, offset = grounder.ground(reference)
            initial_joint_pos, clip_delta = grounder.clip_initial_joint_pos(
                reference.joint_pos[0]
            )
            rollout_50, is_success, first_failure = runner.rollout(
                reference, initial_joint_pos
            )
            sampled = resample_rollout(rollout_50, frames, control_hz, fps)
            for key in collected:
                collected[key].append(sampled[key])
            successes.append(is_success)
            failures.append(-1 if first_failure is None else first_failure)
            grounding_offsets.append(offset)
            initial_root_linear.append(reference.root_lin_vel_w[0])
            initial_root_angular.append(reference.root_ang_vel_w[0])
            initial_joint_velocity.append(reference.joint_vel[0])
            clipped = np.abs(clip_delta) > 1.0e-8
            clip_counts.append(int(clipped.sum()))
            clip_maxima.append(float(np.abs(clip_delta).max(initial=0.0)))
    finally:
        runner.close()

    result = {key: np.concatenate(items, axis=0) for key, items in collected.items()}
    result.update(
        is_success=np.asarray(successes, dtype=np.bool_),
        first_failure_frame_50hz=np.asarray(failures, dtype=np.int32),
        grounding_offset_m=np.asarray(grounding_offsets, dtype=np.float32),
        initial_root_lin_vel_w_m_s=np.asarray(initial_root_linear, dtype=np.float32),
        initial_root_ang_vel_w_rad_s=np.asarray(initial_root_angular, dtype=np.float32),
        initial_joint_vel_rad_s=np.asarray(initial_joint_velocity, dtype=np.float32),
        initial_joint_clip_count=np.asarray(clip_counts, dtype=np.int16),
        initial_joint_clip_max_abs_rad=np.asarray(clip_maxima, dtype=np.float32),
        tracker_protocol_version=np.asarray(TRACKER_PROTOCOL_VERSION, dtype=np.int16),
    )
    atomic_npz(output_path, result)
    if render_video:
        base_video_path = (
            output_path.with_suffix(".mp4")
            if video_output is None
            else Path(video_output).expanduser().resolve()
        )
        for sample_index in range(samples):
            destination = _sample_video_path(
                base_video_path, sample_index, samples
            )
            render_rollout_video(
                {key: result[key][sample_index] for key in REQUIRED_INPUT_FIELDS},
                xml_path=xml_path,
                output_path=destination,
                fps=fps,
                width=video_width,
                height=video_height,
            )
    return result
