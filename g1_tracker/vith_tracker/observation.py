"""Observation construction for the 1570-D deploy policy."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .runtime import (
    DEFAULT_ANGLES,
    HW_DOF,
    MUJOCO_TO_ISAACLAB,
    heading_quat,
    normalize_quat,
    quat_conj,
    quat_mul,
    quat_rotate,
    quat_to_6d,
)

TOKENIZER_OBS_DIM = 640
NORMAL_ACTOR_OBS_DIM = 930
GLOBAL_ROOT_TAIL_DIM = 9
GLOBAL_PROPRIOCEPTION_OBS_DIM = NORMAL_ACTOR_OBS_DIM + GLOBAL_ROOT_TAIL_DIM
GLOBAL_POLICY_OBS_DIM = TOKENIZER_OBS_DIM + GLOBAL_PROPRIOCEPTION_OBS_DIM

class GlobalObservationMixin:
    current_frame: int
    motion: Any
    base_pos: np.ndarray
    base_quat: np.ndarray
    base_ang_vel: np.ndarray
    joint_pos_mj: np.ndarray
    joint_vel_mj: np.ndarray
    last_action: np.ndarray
    history: deque
    init_base_pos: np.ndarray
    init_base_quat: np.ndarray
    init_ref_root_pos: np.ndarray
    init_ref_root_quat: np.ndarray

    def reset_reference_alignment(self) -> None:
        self.init_base_pos = self.base_pos.copy()
        self.init_base_quat = normalize_quat(self.base_quat.copy())
        self.init_ref_root_pos = self.motion.root_pos[self.current_frame].astype(np.float64).copy()
        self.init_ref_root_quat = normalize_quat(self.motion.root_quat[self.current_frame].astype(np.float64))

    def _target_frame(self, frame_idx: int, step_size: int) -> int:
        return min(self.current_frame + frame_idx * step_size, self.motion.timesteps - 1)

    def _aligned_ref_root_pos(self, frame: int) -> np.ndarray:
        apply_delta_heading = quat_mul(heading_quat(self.init_base_quat), quat_conj(heading_quat(self.init_ref_root_quat)))
        delta = self.motion.root_pos[frame].astype(np.float64) - self.init_ref_root_pos
        return self.init_base_pos + quat_rotate(apply_delta_heading, delta)

    def _aligned_ref_root_quat(self, frame: int) -> np.ndarray:
        apply_delta_heading = quat_mul(heading_quat(self.init_base_quat), quat_conj(heading_quat(self.init_ref_root_quat)))
        ref_root = normalize_quat(self.motion.root_quat[frame].astype(np.float64))
        return normalize_quat(quat_mul(apply_delta_heading, ref_root))

    def _policy_order_q(self) -> np.ndarray:
        return self.joint_pos_mj[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]

    def _policy_order_dq(self) -> np.ndarray:
        return self.joint_vel_mj[MUJOCO_TO_ISAACLAB]

    def _append_history(self) -> None:
        self.history.append(
            {
                "base_quat": self.base_quat.copy(),
                "base_ang_vel": self.base_ang_vel.copy(),
                "body_q": self._policy_order_q(),
                "body_dq": self._policy_order_dq(),
                "last_action": self.last_action.copy(),
            }
        )

    def _history_array(self, key: str) -> np.ndarray:
        rows = list(self.history)
        if len(rows) < 10:
            rows = [None] * (10 - len(rows)) + rows
        values = []
        for row in rows[-10:]:
            if row is None:
                if key in ("base_ang_vel", "gravity_dir"):
                    values.append(np.zeros(3, dtype=np.float64) if key == "base_ang_vel" else np.array([0.0, 0.0, -1.0]))
                else:
                    values.append(np.zeros(HW_DOF, dtype=np.float64))
            elif key == "gravity_dir":
                values.append(quat_rotate(quat_conj(row["base_quat"]), np.array([0.0, 0.0, -1.0], dtype=np.float64)))
            else:
                values.append(row[key])
        return np.asarray(values, dtype=np.float32).reshape(-1)

    def _command_multi_future(self) -> np.ndarray:
        frames = [self._target_frame(i, 5) for i in range(10)]
        return np.concatenate(
            [
                self.motion.joint_pos[frames].reshape(-1),
                self.motion.joint_vel[frames].reshape(-1),
            ]
        ).astype(np.float32)

    def _motion_anchor_orientation_10frame_step5(self) -> np.ndarray:
        out = []
        for frame_idx in range(10):
            frame = self._target_frame(frame_idx, 5)
            rel = quat_mul(quat_conj(self.base_quat), self._aligned_ref_root_quat(frame))
            out.append(quat_to_6d(rel))
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def _global_root_tail(self) -> np.ndarray:
        frame = self._target_frame(0, 5)
        rel_pos = quat_rotate(quat_conj(self.base_quat), self._aligned_ref_root_pos(frame) - self.base_pos)
        rel_ori = quat_mul(quat_conj(self.base_quat), self._aligned_ref_root_quat(frame))
        return np.concatenate([rel_pos, quat_to_6d(rel_ori)]).astype(np.float32)

    def _normal_actor_observation(self) -> np.ndarray:
        """Actor proprioception in the same order as the normal SONIC deploy path."""
        return np.concatenate(
            [
                self._history_array("base_ang_vel"),
                self._history_array("body_q"),
                self._history_array("body_dq"),
                self._history_array("last_action"),
                self._history_array("gravity_dir"),
            ]
        ).astype(np.float32)

    def compute_global_policy_observation(self) -> np.ndarray:
        tokenizer_obs = np.concatenate(
            [
                self._command_multi_future(),
                self._motion_anchor_orientation_10frame_step5(),
            ]
        ).astype(np.float32)
        normal_actor_obs = self._normal_actor_observation()
        global_tail = self._global_root_tail()
        global_proprioception = np.concatenate([normal_actor_obs, global_tail]).astype(np.float32)
        obs = np.concatenate([tokenizer_obs, global_proprioception]).astype(np.float32)
        if tokenizer_obs.shape[0] != TOKENIZER_OBS_DIM:
            raise RuntimeError(f"Tokenizer observation must be {TOKENIZER_OBS_DIM}D, got {tokenizer_obs.shape[0]}")
        if normal_actor_obs.shape[0] != NORMAL_ACTOR_OBS_DIM:
            raise RuntimeError(f"Normal actor observation must be {NORMAL_ACTOR_OBS_DIM}D, got {normal_actor_obs.shape[0]}")
        if global_tail.shape[0] != GLOBAL_ROOT_TAIL_DIM:
            raise RuntimeError(f"Global root tail must be {GLOBAL_ROOT_TAIL_DIM}D, got {global_tail.shape[0]}")
        if global_proprioception.shape[0] != GLOBAL_PROPRIOCEPTION_OBS_DIM:
            raise RuntimeError(
                f"Global proprioception must be {GLOBAL_PROPRIOCEPTION_OBS_DIM}D, got {global_proprioception.shape[0]}"
            )
        if obs.shape[0] != GLOBAL_POLICY_OBS_DIM:
            raise RuntimeError(f"Global SONIC observation must be {GLOBAL_POLICY_OBS_DIM}D, got {obs.shape[0]}")
        return obs.reshape(1, -1)
