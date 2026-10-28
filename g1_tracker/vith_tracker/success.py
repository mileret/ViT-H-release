"""Standardized G1 tracking-success evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SONIC_ANCHOR_BODY_NAME = "pelvis"
SONIC_END_EFFECTOR_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

SONIC_STANDARDIZED_ANCHOR_HEIGHT_THRESHOLD_M = 0.25
SONIC_STANDARDIZED_ANCHOR_ORIENTATION_SQUARED_THRESHOLD_RAD2 = 1.0
SONIC_STANDARDIZED_END_EFFECTOR_HEIGHT_THRESHOLD_M = 0.25

@dataclass(frozen=True)
class SonicStandardizedFrameResult:
    """Detailed result for one synchronized reference/robot frame."""

    anchor_height_error_m: float
    anchor_orientation_error_rad: float
    end_effector_height_errors_m: np.ndarray  # (4,) float64
    failed_anchor_height: bool
    failed_anchor_orientation: bool
    failed_end_effector_height: bool

    @property
    def is_failure(self) -> bool:
        return (
            self.failed_anchor_height
            or self.failed_anchor_orientation
            or self.failed_end_effector_height
        )

def _require_array_shape(
    array: np.ndarray,
    expected_shape: tuple[int, ...],
    array_name: str,
) -> None:
    if array.shape != expected_shape:
        raise ValueError(
            f"{array_name} must have shape {expected_shape}, got {array.shape}."
        )


def _quaternion_multiply_wxyz(
    first_quaternion_wxyz: np.ndarray,
    second_quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """Multiply scalar-first quaternions with matching leading dimensions."""

    first_w, first_x, first_y, first_z = np.moveaxis(
        first_quaternion_wxyz, -1, 0
    )
    second_w, second_x, second_y, second_z = np.moveaxis(
        second_quaternion_wxyz, -1, 0
    )
    return np.stack(
        (
            first_w * second_w
            - first_x * second_x
            - first_y * second_y
            - first_z * second_z,
            first_w * second_x
            + first_x * second_w
            + first_y * second_z
            - first_z * second_y,
            first_w * second_y
            - first_x * second_z
            + first_y * second_w
            + first_z * second_x,
            first_w * second_z
            + first_x * second_y
            - first_y * second_x
            + first_z * second_w,
        ),
        axis=-1,
    )


def _quaternion_error_magnitude_wxyz(
    reference_quaternion_wxyz: np.ndarray,
    robot_quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """Match Isaac Lab ``quat_error_magnitude`` for unit quaternions."""

    if reference_quaternion_wxyz.shape != robot_quaternion_wxyz.shape:
        raise ValueError(
            "Reference and robot quaternion shapes must match, got "
            f"{reference_quaternion_wxyz.shape} and "
            f"{robot_quaternion_wxyz.shape}."
        )
    if reference_quaternion_wxyz.shape[-1] != 4:
        raise ValueError(
            "Quaternions must have final dimension 4 in (w, x, y, z) order."
        )

    robot_quaternion_conjugate_wxyz = robot_quaternion_wxyz.copy()
    robot_quaternion_conjugate_wxyz[..., 1:] *= -1.0
    quaternion_difference_wxyz = _quaternion_multiply_wxyz(
        reference_quaternion_wxyz,
        robot_quaternion_conjugate_wxyz,
    )

    # Isaac Lab flips negative-w quaternions before converting to axis-angle,
    # which selects the shortest equivalent rotation in [0, pi].
    quaternion_difference_wxyz = np.where(
        quaternion_difference_wxyz[..., 0:1] < 0.0,
        -quaternion_difference_wxyz,
        quaternion_difference_wxyz,
    )
    vector_magnitude = np.linalg.norm(
        quaternion_difference_wxyz[..., 1:], axis=-1
    )
    half_angle_rad = np.arctan2(
        vector_magnitude,
        quaternion_difference_wxyz[..., 0],
    )
    angle_rad = np.asarray(2.0 * half_angle_rad, dtype=np.float64)

    # Keep Isaac Lab's small-angle Taylor branch instead of assuming perfectly
    # normalized inputs.  This also makes the numerical behavior match its
    # axis_angle_from_quat implementation.
    sin_half_angle_over_angle = np.asarray(
        0.5 - angle_rad**2 / 48.0,
        dtype=np.float64,
    ).copy()
    np.divide(
        np.sin(half_angle_rad),
        angle_rad,
        out=sin_half_angle_over_angle,
        where=np.abs(angle_rad) > 1.0e-6,
    )
    axis_angle_error = (
        quaternion_difference_wxyz[..., 1:]
        / sin_half_angle_over_angle[..., None]
    )
    return np.linalg.norm(axis_angle_error, axis=-1)

def evaluate_sonic_standardized_frame(
    reference_anchor_position_w: np.ndarray,
    robot_anchor_position_w: np.ndarray,
    reference_anchor_quaternion_wxyz: np.ndarray,
    robot_anchor_quaternion_wxyz: np.ndarray,
    reference_end_effector_positions_w: np.ndarray,
    robot_end_effector_positions_w: np.ndarray,
    anchor_height_threshold_m: float = SONIC_STANDARDIZED_ANCHOR_HEIGHT_THRESHOLD_M,
    anchor_orientation_squared_threshold_rad2: float = (
        SONIC_STANDARDIZED_ANCHOR_ORIENTATION_SQUARED_THRESHOLD_RAD2
    ),
    end_effector_height_threshold_m: float = (
        SONIC_STANDARDIZED_END_EFFECTOR_HEIGHT_THRESHOLD_M
    ),
) -> SonicStandardizedFrameResult:
    """Evaluate one synchronized frame using SONIC standardized thresholds."""

    reference_anchor_position_w = np.asarray(
        reference_anchor_position_w, dtype=np.float64
    )
    robot_anchor_position_w = np.asarray(robot_anchor_position_w, dtype=np.float64)
    reference_anchor_quaternion_wxyz = np.asarray(
        reference_anchor_quaternion_wxyz, dtype=np.float64
    )
    robot_anchor_quaternion_wxyz = np.asarray(
        robot_anchor_quaternion_wxyz, dtype=np.float64
    )
    reference_end_effector_positions_w = np.asarray(
        reference_end_effector_positions_w, dtype=np.float64
    )
    robot_end_effector_positions_w = np.asarray(
        robot_end_effector_positions_w, dtype=np.float64
    )

    _require_array_shape(reference_anchor_position_w, (3,), "reference_anchor_position_w")
    _require_array_shape(robot_anchor_position_w, (3,), "robot_anchor_position_w")
    _require_array_shape(
        reference_anchor_quaternion_wxyz,
        (4,),
        "reference_anchor_quaternion_wxyz",
    )
    _require_array_shape(
        robot_anchor_quaternion_wxyz,
        (4,),
        "robot_anchor_quaternion_wxyz",
    )
    _require_array_shape(
        reference_end_effector_positions_w,
        (len(SONIC_END_EFFECTOR_BODY_NAMES), 3),
        "reference_end_effector_positions_w",
    )
    _require_array_shape(
        robot_end_effector_positions_w,
        (len(SONIC_END_EFFECTOR_BODY_NAMES), 3),
        "robot_end_effector_positions_w",
    )

    anchor_height_error_m = float(
        abs(reference_anchor_position_w[2] - robot_anchor_position_w[2])
    )
    anchor_orientation_error_rad = float(
        _quaternion_error_magnitude_wxyz(
            reference_anchor_quaternion_wxyz,
            robot_anchor_quaternion_wxyz,
        )
    )
    end_effector_height_errors_m = np.abs(
        reference_end_effector_positions_w[:, 2]
        - robot_end_effector_positions_w[:, 2]
    )  # (4,) float64

    failed_anchor_height = anchor_height_error_m > anchor_height_threshold_m
    failed_anchor_orientation = (
        anchor_orientation_error_rad**2
        > anchor_orientation_squared_threshold_rad2
    )
    failed_end_effector_height = bool(
        np.any(end_effector_height_errors_m > end_effector_height_threshold_m)
    )

    return SonicStandardizedFrameResult(
        anchor_height_error_m=anchor_height_error_m,
        anchor_orientation_error_rad=anchor_orientation_error_rad,
        end_effector_height_errors_m=end_effector_height_errors_m,
        failed_anchor_height=failed_anchor_height,
        failed_anchor_orientation=failed_anchor_orientation,
        failed_end_effector_height=failed_end_effector_height,
    )

class SonicStandardizedSuccessAccumulator:
    """Latch SONIC standardized failure over an online MuJoCo rollout."""

    def __init__(
        self,
        anchor_height_threshold_m: float = (
            SONIC_STANDARDIZED_ANCHOR_HEIGHT_THRESHOLD_M
        ),
        anchor_orientation_squared_threshold_rad2: float = (
            SONIC_STANDARDIZED_ANCHOR_ORIENTATION_SQUARED_THRESHOLD_RAD2
        ),
        end_effector_height_threshold_m: float = (
            SONIC_STANDARDIZED_END_EFFECTOR_HEIGHT_THRESHOLD_M
        ),
    ) -> None:
        self.anchor_height_threshold_m = anchor_height_threshold_m
        self.anchor_orientation_squared_threshold_rad2 = (
            anchor_orientation_squared_threshold_rad2
        )
        self.end_effector_height_threshold_m = end_effector_height_threshold_m
        self.reset()

    def reset(self) -> None:
        """Start accumulating a new motion."""

        self.processed_frame_count = 0
        self.active_frame_count = 0
        self.first_failure_frame_index: int | None = None
        self.failed_anchor_height = False
        self.failed_anchor_orientation = False
        self.failed_end_effector_height = False

    @property
    def is_success(self) -> bool:
        """Return true after at least one active frame and no latched failure."""

        return self.active_frame_count > 0 and self.first_failure_frame_index is None

    def update(
        self,
        reference_anchor_position_w: np.ndarray,
        robot_anchor_position_w: np.ndarray,
        reference_anchor_quaternion_wxyz: np.ndarray,
        robot_anchor_quaternion_wxyz: np.ndarray,
        reference_end_effector_positions_w: np.ndarray,
        robot_end_effector_positions_w: np.ndarray,
        frame_is_active: bool = True,
    ) -> SonicStandardizedFrameResult:
        """Evaluate one frame and latch failures only when the frame is active."""

        frame_result = evaluate_sonic_standardized_frame(
            reference_anchor_position_w=reference_anchor_position_w,
            robot_anchor_position_w=robot_anchor_position_w,
            reference_anchor_quaternion_wxyz=reference_anchor_quaternion_wxyz,
            robot_anchor_quaternion_wxyz=robot_anchor_quaternion_wxyz,
            reference_end_effector_positions_w=reference_end_effector_positions_w,
            robot_end_effector_positions_w=robot_end_effector_positions_w,
            anchor_height_threshold_m=self.anchor_height_threshold_m,
            anchor_orientation_squared_threshold_rad2=(
                self.anchor_orientation_squared_threshold_rad2
            ),
            end_effector_height_threshold_m=self.end_effector_height_threshold_m,
        )

        frame_index = self.processed_frame_count
        self.processed_frame_count += 1
        if not frame_is_active:
            return frame_result

        self.active_frame_count += 1
        self.failed_anchor_height |= frame_result.failed_anchor_height
        self.failed_anchor_orientation |= frame_result.failed_anchor_orientation
        self.failed_end_effector_height |= frame_result.failed_end_effector_height
        if frame_result.is_failure and self.first_failure_frame_index is None:
            self.first_failure_frame_index = frame_index
        return frame_result
