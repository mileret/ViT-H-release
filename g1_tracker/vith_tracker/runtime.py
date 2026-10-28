"""Robot constants and MuJoCo helpers used by the G1 tracker."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

try:
    import mujoco.viewer
except Exception:
    pass

HW_DOF = 29


# Arrays copied from gear_sonic_deploy/.../policy_parameters.hpp.
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

DEFAULT_ANGLES = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10.0 * 2.0 * np.pi
DAMPING_RATIO = 2.0
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ
EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0

KP = np.array(
    [
        STIFFNESS_7520_22,
        STIFFNESS_7520_22,
        STIFFNESS_7520_14,
        STIFFNESS_7520_22,
        2.0 * STIFFNESS_5020,
        2.0 * STIFFNESS_5020,
        STIFFNESS_7520_22,
        STIFFNESS_7520_22,
        STIFFNESS_7520_14,
        STIFFNESS_7520_22,
        2.0 * STIFFNESS_5020,
        2.0 * STIFFNESS_5020,
        STIFFNESS_7520_14,
        2.0 * STIFFNESS_5020,
        2.0 * STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_4010,
        STIFFNESS_4010,
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_5020,
        STIFFNESS_4010,
        STIFFNESS_4010,
    ],
    dtype=np.float64,
)
KD = np.array(
    [
        DAMPING_7520_22,
        DAMPING_7520_22,
        DAMPING_7520_14,
        DAMPING_7520_22,
        2.0 * DAMPING_5020,
        2.0 * DAMPING_5020,
        DAMPING_7520_22,
        DAMPING_7520_22,
        DAMPING_7520_14,
        DAMPING_7520_22,
        2.0 * DAMPING_5020,
        2.0 * DAMPING_5020,
        DAMPING_7520_14,
        2.0 * DAMPING_5020,
        2.0 * DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_4010,
        DAMPING_4010,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_4010,
        DAMPING_4010,
    ],
    dtype=np.float64,
)
ACTION_SCALE = np.array(
    [
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
    ],
    dtype=np.float64,
)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_mul(quat_mul(q, np.array([0.0, *v], dtype=np.float64)), quat_conj(q))[1:]


def normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def heading_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)], dtype=np.float64)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_to_6d(q: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q)[:, :2].reshape(-1)


def pd_control(target_q: np.ndarray, q: np.ndarray, kp: np.ndarray, target_dq: np.ndarray, dq: np.ndarray, kd: np.ndarray) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd

class G1SonicEnv:
    def __init__(self, xml_path: Path, viewer: bool):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.001
        self.ctrl_start = self._resolve_policy_actuator_start()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if viewer else None

    def _resolve_policy_actuator_start(self) -> int:
        if self.model.nu == HW_DOF:
            return 0
        if self.model.nu >= HW_DOF + 6:
            actuator_names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                for i in range(6)
            ]
            if actuator_names == ["fx", "fy", "fz", "tx", "ty", "tz"]:
                return 6
        raise RuntimeError(
            f"Expected either {HW_DOF} policy actuators or 6 freebase + {HW_DOF} policy actuators, got {self.model.nu}"
        )

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()

    def sync(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()
