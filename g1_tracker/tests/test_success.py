from __future__ import annotations

import unittest

import numpy as np

from vith_tracker.success import (
    SonicStandardizedSuccessAccumulator,
    evaluate_sonic_standardized_frame,
)


class SuccessTest(unittest.TestCase):
    def test_identical_states_succeed(self) -> None:
        result = evaluate_sonic_standardized_frame(
            reference_anchor_position_w=np.zeros(3),
            robot_anchor_position_w=np.zeros(3),
            reference_anchor_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            robot_anchor_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            reference_end_effector_positions_w=np.zeros((4, 3)),
            robot_end_effector_positions_w=np.zeros((4, 3)),
        )
        self.assertFalse(result.is_failure)

    def test_large_pelvis_height_error_fails(self) -> None:
        result = evaluate_sonic_standardized_frame(
            reference_anchor_position_w=np.zeros(3),
            robot_anchor_position_w=np.array([0.0, 0.0, 0.3]),
            reference_anchor_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            robot_anchor_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            reference_end_effector_positions_w=np.zeros((4, 3)),
            robot_end_effector_positions_w=np.zeros((4, 3)),
        )
        self.assertTrue(result.is_failure)

    def test_accumulator_latches_failure(self) -> None:
        accumulator = SonicStandardizedSuccessAccumulator()
        arguments = dict(
            reference_anchor_position_w=np.zeros(3),
            reference_anchor_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            robot_anchor_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            reference_end_effector_positions_w=np.zeros((4, 3)),
            robot_end_effector_positions_w=np.zeros((4, 3)),
        )
        accumulator.update(robot_anchor_position_w=np.zeros(3), **arguments)
        self.assertTrue(accumulator.is_success)
        accumulator.update(
            robot_anchor_position_w=np.array([0.0, 0.0, 0.3]), **arguments
        )
        self.assertFalse(accumulator.is_success)
        self.assertEqual(accumulator.first_failure_frame_index, 1)


if __name__ == "__main__":
    unittest.main()
