import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hmr4d.utils.video_timeline import select_frame_indices_by_timestamp
from tools.inference_utils import (
    _extract_pipeline_state_dict,
    resolve_imgfeat_encoder,
    validate_and_load_model_state,
)
from tools.preprocess_cache import (
    CACHE_SCHEMA_VERSION,
    artifact_identities,
    atomic_write_json,
    file_identity,
    stable_fingerprint,
    validate_complete_manifest,
)


class TimestampResamplingTests(unittest.TestCase):
    def test_cfr_inputs_preserve_one_second_duration(self):
        for source_fps in (24, 25, 30, 60):
            with self.subTest(source_fps=source_fps):
                timestamps = np.arange(source_fps, dtype=np.float64) / source_fps
                selected = select_frame_indices_by_timestamp(timestamps, 1.0, 30.0)
                self.assertEqual(len(selected), 30)
                self.assertTrue(np.all(np.diff(selected) >= 0))
                if source_fps == 30:
                    np.testing.assert_array_equal(selected, np.arange(30))
                if source_fps == 60:
                    self.assertGreater(selected[-1], 30)

    def test_vfr_uses_timestamps(self):
        timestamps = np.array([0.0, 0.01, 0.08, 0.11, 0.19])
        selected = select_frame_indices_by_timestamp(timestamps, 0.22, 20.0)
        np.testing.assert_array_equal(selected, np.array([0, 2, 3, 4, 4]))

    def test_invalid_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no timestamped frames"):
            select_frame_indices_by_timestamp([], 1.0, 30.0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            select_frame_indices_by_timestamp([0.0, 0.1, 0.1], 0.2, 30.0)
        with self.assertRaisesRegex(ValueError, "Invalid target_fps"):
            select_frame_indices_by_timestamp([0.0], 1.0, 0.0)


class CacheIdentityTests(unittest.TestCase):
    def test_content_and_configuration_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "same.mp4"
            path.write_bytes(b"first")
            first = file_identity(path, required=True)
            fingerprint = stable_fingerprint({"source": first, "fps": 30})

            path.write_bytes(b"other")
            second = file_identity(path, required=True)
            self.assertNotEqual(first["sha256"], second["sha256"])
            self.assertNotEqual(
                fingerprint,
                stable_fingerprint({"source": second, "fps": 30}),
            )
            self.assertNotEqual(
                fingerprint,
                stable_fingerprint({"source": first, "fps": 25}),
            )

    def test_manifest_requires_complete_unchanged_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "bbx.pt"
            artifact.write_bytes(b"complete")
            manifest_path = root / "cache_manifest.json"
            fingerprint = stable_fingerprint({"input": "a"})
            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "complete",
                "cache_fingerprint": fingerprint,
                "artifacts": artifact_identities([artifact]),
            }
            atomic_write_json(manifest_path, manifest)
            valid, _, _ = validate_complete_manifest(manifest_path, fingerprint)
            self.assertTrue(valid)

            artifact.write_bytes(b"corrupt")
            valid, reason, _ = validate_complete_manifest(manifest_path, fingerprint)
            self.assertFalse(valid)
            self.assertIn("content changed", reason)

            manifest["status"] = "running"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            valid, reason, _ = validate_complete_manifest(manifest_path, fingerprint)
            self.assertFalse(valid)
            self.assertIn("complete", reason)


class _TinyPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser3d = torch.nn.Linear(2, 3)
        self.cond_embedder = torch.nn.Linear(3, 2)
        self.register_buffer("persistent_scale", torch.ones(1))
        self.register_buffer("scratch", torch.zeros(1), persistent=False)


class CheckpointValidationTests(unittest.TestCase):
    def test_both_containers_and_pipeline_prefix(self):
        source = _TinyPipeline()
        prefixed = {f"pipeline.{key}": value.clone() for key, value in source.state_dict().items()}
        for container in ("state_dict", "model_state_dict"):
            state = _extract_pipeline_state_dict({container: prefixed})
            target = _TinyPipeline()
            validate_and_load_model_state(target, state)
            for key, value in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[key], value)

    def test_missing_unexpected_and_shape_mismatch_are_rejected(self):
        model = _TinyPipeline()
        complete = {key: value.clone() for key, value in model.state_dict().items()}

        for missing_key in ("denoiser3d.weight", "cond_embedder.weight"):
            damaged = dict(complete)
            damaged.pop(missing_key)
            with self.assertRaisesRegex(RuntimeError, "missing"):
                validate_and_load_model_state(model, damaged)

        unexpected = dict(complete)
        unexpected["unknown.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            validate_and_load_model_state(model, unexpected)

        wrong_shape = dict(complete)
        wrong_shape["cond_embedder.weight"] = torch.ones(1, 1)
        with self.assertRaisesRegex(RuntimeError, "shape_mismatches"):
            validate_and_load_model_state(model, wrong_shape)

    def test_nonpersistent_buffer_is_not_required(self):
        model = _TinyPipeline()
        self.assertNotIn("scratch", model.state_dict())
        validate_and_load_model_state(model, dict(model.state_dict()))

    def test_mixed_prefixes_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "mixes"):
            _extract_pipeline_state_dict(
                {"state_dict": {"pipeline.a": torch.ones(1), "b": torch.ones(1)}}
            )


class EncoderConsistencyTests(unittest.TestCase):
    def test_release_encoder_is_selected_automatically(self):
        self.assertEqual(resolve_imgfeat_encoder(None, "sam3d", "release"), "sam3d")

    def test_unknown_custom_checkpoint_requires_explicit_encoder(self):
        with self.assertRaisesRegex(ValueError, "does not declare"):
            resolve_imgfeat_encoder(None, None, "no metadata")

    def test_mismatch_requires_opt_in(self):
        with self.assertRaisesRegex(ValueError, "requires sam3d"):
            resolve_imgfeat_encoder("hmr2", "sam3d", "release")
        self.assertEqual(
            resolve_imgfeat_encoder(
                "hmr2", "sam3d", "release", allow_mismatch=True
            ),
            "hmr2",
        )


class WindowAlignmentTests(unittest.TestCase):
    def test_root_and_body_receive_same_rigid_transform(self):
        from motion_lib.motion_utils.rotation_conversions import quaternion_to_matrix
        from tools.infer_video import align_window_to_previous

        root = torch.tensor([[10.0, 1.0, 0.0], [11.0, 1.0, 0.0]])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        body = torch.stack((root + torch.tensor([1.0, 0.0, 0.0]),), dim=1)
        previous_root = torch.tensor([3.0, 4.0, 0.0])
        half_sqrt = 2.0**-0.5
        previous_quat = torch.tensor([half_sqrt, 0.0, 0.0, half_sqrt])

        aligned_root, aligned_quat, aligned_body = align_window_to_previous(
            root, quat, body, previous_root, previous_quat
        )

        torch.testing.assert_close(aligned_root[0], previous_root)
        torch.testing.assert_close(aligned_root[1], torch.tensor([3.0, 5.0, 0.0]))
        torch.testing.assert_close(aligned_body[0, 0], torch.tensor([3.0, 5.0, 0.0]))
        torch.testing.assert_close(
            quaternion_to_matrix(aligned_quat[0]),
            quaternion_to_matrix(previous_quat),
        )

    @staticmethod
    def _prep(total_len, window):
        return {
            "obs_full": torch.zeros(total_len, 16, 2),
            "bbx_xys_full": torch.zeros(total_len, 3),
            "K_fullimg_full": torch.eye(3).repeat(total_len, 1, 1),
            "cam_angvel_full": torch.zeros(total_len, 6),
            "f_imgseq_full": torch.zeros(total_len, 1024),
            "chunks_meta": (
                [(0, total_len)]
                if total_len <= window
                else [(0, window), (window - 1, total_len - window + 1)]
            ),
            "window": window,
        }

    def test_single_window_positions_are_only_converted_to_z_up(self):
        from tools.infer_video import infer_video

        class Pipeline:
            @staticmethod
            def forward(batch, train=False):
                del train
                length = batch["obs"].shape[1]
                root = torch.tensor(
                    [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [3.0, 2.0, 3.0]]
                )[:length]
                return {
                    "pred_g1_joint_pos": torch.zeros(1, length, 29),
                    "pred_root_pos": root.unsqueeze(0),
                    "pred_root_quat_w": torch.tensor(
                        [[[1.0, 0.0, 0.0, 0.0]]]
                    ).expand(1, length, 4),
                    "pred_g1_body_pos_w": (root + torch.tensor([0.0, 1.0, 0.0]))[
                        None, :, None, :
                    ],
                }

        result = infer_video(Pipeline(), self._prep(3, 120), 3, device="cpu")
        np.testing.assert_allclose(
            result["root_pos_w"],
            np.array([[3.0, 1.0, 2.0], [3.0, 2.0, 2.0], [3.0, 3.0, 2.0]]),
        )
        np.testing.assert_allclose(
            result["body_pos_w"][:, 0],
            np.array([[3.0, 1.0, 3.0], [3.0, 2.0, 3.0], [3.0, 3.0, 3.0]]),
        )

    def test_two_windows_keep_body_and_root_continuous(self):
        from tools.infer_video import infer_video

        class Pipeline:
            def __init__(self):
                self.calls = 0

            def forward(self, batch, train=False):
                del train
                length = batch["obs"].shape[1]
                if self.calls == 0:
                    root = torch.stack(
                        (
                            torch.arange(length, dtype=torch.float32),
                            torch.zeros(length),
                            torch.zeros(length),
                        ),
                        dim=-1,
                    )
                    half_sqrt = 2.0**-0.5
                    quat = torch.tensor([half_sqrt, 0.0, 0.0, half_sqrt]).repeat(length, 1)
                    body_offset = torch.tensor([0.0, 1.0, 0.0])
                else:
                    root = torch.tensor([10.0, 1.0, 0.0]).repeat(length, 1)
                    root[:, 0] += torch.arange(length, dtype=torch.float32)
                    quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(length, 1)
                    body_offset = torch.tensor([1.0, 0.0, 0.0])
                self.calls += 1
                body = root[:, None, :] + body_offset
                return {
                    "pred_g1_joint_pos": torch.zeros(1, length, 29),
                    "pred_root_pos": root.unsqueeze(0),
                    "pred_root_quat_w": quat.unsqueeze(0),
                    "pred_g1_body_pos_w": body.unsqueeze(0),
                }

        result = infer_video(Pipeline(), self._prep(121, 120), 121, device="cpu")
        self.assertEqual(result["root_pos_w"].shape, (121, 3))
        self.assertEqual(result["body_pos_w"].shape, (121, 1, 3))
        np.testing.assert_allclose(
            np.linalg.norm(result["root_pos_w"][120] - result["root_pos_w"][119]),
            1.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.linalg.norm(result["body_pos_w"][120, 0] - result["body_pos_w"][119, 0]),
            1.0,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
