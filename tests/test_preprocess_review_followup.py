"""CPU regressions for SAM3D identity, camera poses, and cache ownership."""

from contextlib import contextmanager
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from hmr4d.utils.sam3d_config import (
    FAST_SAM3D_DEFAULTS, resolve_sam3d_model_config, sam3d_runtime_settings,
)
from motion_lib.motion_utils.rotation_conversions import quaternion_to_matrix
from tools import video_preprocess as preproc
from tools.infer_video import add_video_camera_data
from tools.preprocess_cache import (
    CACHE_SCHEMA_VERSION, artifact_identities, atomic_write_json,
    cache_directory_lock, require_cache_lock, stable_fingerprint,
)


def fixture_args(directory, encoder="sam3d", source="a/input.mp4"):
    root = Path(directory)
    repo = root / "repo"
    fast = repo / "third-party/Fast-SAM-3D-Body"
    for relative in (
        "inputs/checkpoints/yolo/yolov8x.pt",
        "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
        "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
        "third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model.ckpt",
        "third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"test resource")
    config = fast / "checkpoints/sam-3d-body-dinov3/model_config.yaml"
    if not config.exists():
        config.write_text("MODEL:\n  IMAGE_SIZE: [512, 512]\n", encoding="utf-8")
    video = root / source
    video.parent.mkdir(parents=True, exist_ok=True)
    if not video.exists():
        video.write_bytes(source.encode())
    return SimpleNamespace(
        video=str(video), output_root=str(root / "outputs"), imgfeat_encoder=encoder,
        sam3d_root=None, sam3d_ckpt_path=None, sam3d_mhr_path=None,
        sam3d_img_ds=1., sam3d_batch_size=16, static_cam=True, use_dpvo=False,
        f_mm=None, verbose=False, force_recompute=False, force_recompute_imgfeat=False,
    )


def fake_resample(source, destination, target_fps, crf):
    Path(destination).write_bytes(Path(source).read_bytes())
    return {
        "target_frames": 3, "source_decoded_frames": 3,
        "source_duration_seconds": .1, "strategy": "test-timeline",
    }


@contextmanager
def fake_video_io(directory):
    # Only model resources and codec I/O are fixtures; config, hashing,
    # invalidation, session locking and manifest verification are production code.
    with patch.object(preproc, "__file__", str(Path(directory) / "repo/tools/video_preprocess.py")), \
         patch.object(preproc, "resample_video_to_fps", side_effect=fake_resample), \
         patch.object(preproc, "get_video_lwh", return_value=(3, 32, 32)):
        yield


def finish_cache(cfg, payload=b"features"):
    require_cache_lock(cfg.output_dir)
    for name in ("bbx", "vitpose", "vit_features"):
        Path(cfg.paths[name]).write_bytes(payload)
    meta = Path(cfg.paths.vit_features).with_suffix(".meta.json")
    meta.write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION, "status": "complete",
        "cache_fingerprint": str(cfg.cache_fingerprint),
        "timeline": OmegaConf.to_container(cfg.timeline),
        "identity": OmegaConf.to_container(cfg.cache_identity),
        "artifacts": artifact_identities([cfg.video_path, cfg.paths.bbx, cfg.paths.vitpose, cfg.paths.vit_features, meta]),
    }
    atomic_write_json(cfg.cache_manifest_path, manifest)


class SAM3DCacheTests(unittest.TestCase):
    def test_default_custom_sibling_parent_and_missing_config(self):
        with tempfile.TemporaryDirectory() as tmp, fake_video_io(tmp):
            args = fixture_args(tmp)
            resources = preproc._resource_identities(args, Path(tmp))
            default = Path(args.sam3d_ckpt_path).with_name("model_config.yaml")
            self.assertEqual(resources["sam3d_model_config"]["path"], str(default))
            checkpoint = Path(tmp) / "custom/weights/model.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"custom")
            sibling, parent = checkpoint.with_name("model_config.yaml"), checkpoint.parent.parent / "model_config.yaml"
            parent.write_text("MODEL:\n  IMAGE_SIZE: [256, 256]\n", encoding="utf-8")
            args.sam3d_ckpt_path = str(checkpoint)
            resources = preproc._resource_identities(args, Path(tmp))
            self.assertEqual(resources["sam3d_model_config"]["path"], str(parent))
            sibling.write_text("MODEL:\n  IMAGE_SIZE: [512, 512]\n", encoding="utf-8")
            before = preproc._resource_identities(args, Path(tmp))
            self.assertEqual(before["sam3d_model_config"]["path"], str(sibling))
            sibling.write_text("MODEL:\n  IMAGE_SIZE: [256, 256]\n", encoding="utf-8")
            after = preproc._resource_identities(args, Path(tmp))
            self.assertNotEqual(stable_fingerprint(before), stable_fingerprint(after))
            sibling.unlink()
            parent.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "model_config.yaml"):
                preproc._resource_identities(args, Path(tmp))

    def test_effective_defaults_aliases_and_yaml_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, fake_video_io(tmp):
            args = fixture_args(tmp)
            preproc._resource_identities(args, Path(tmp))
            runtime = lambda env: sam3d_runtime_settings(args.sam3d_root, args.sam3d_ckpt_path, env)
            default = runtime({})
            self.assertEqual(default["image_size"], [512, 512])
            self.assertEqual(default["layer_dtype"], "fp32")
            self.assertEqual(default, runtime(FAST_SAM3D_DEFAULTS))
            self.assertEqual(default, runtime({"IMG_SIZE": "0512", "LAYER_DTYPE": "FLOAT32"}))
            self.assertEqual(runtime({"LAYER_DTYPE": "fp16"}), runtime({"LAYER_DTYPE": "float16"}))
            self.assertEqual(runtime({"LAYER_DTYPE": "bf16"}), runtime({"LAYER_DTYPE": "bfloat16"}))
            config = resolve_sam3d_model_config(args.sam3d_ckpt_path)
            config.write_text("MODEL:\n  IMAGE_SIZE: [256, 384]\n", encoding="utf-8")
            self.assertEqual(runtime({"IMG_SIZE": "0"})["image_size"], [256, 384])
            self.assertEqual(runtime({"IMG_SIZE": ""})["image_size"], [256, 384])
            self.assertEqual(runtime({"IMG_SIZE": "128"})["image_size"], [128, 128])

    def test_feature_cache_invalidates_for_config_and_effective_settings(self):
        with tempfile.TemporaryDirectory() as tmp, fake_video_io(tmp), patch.dict(os.environ, {}, clear=True):
            args = fixture_args(tmp)
            with preproc.preprocessing_session(args) as cfg:
                finish_cache(cfg)
                original = str(cfg.cache_fingerprint)
            with patch.dict(os.environ, {"IMG_SIZE": "512", "LAYER_DTYPE": "float32"}):
                with preproc.preprocessing_session(args) as cfg:
                    self.assertTrue(cfg.cache_verified)
                    self.assertEqual(cfg.cache_fingerprint, original)
                    self.assertEqual(Path(cfg.paths.vit_features).read_bytes(), b"features")
            for overrides in ({"IMG_SIZE": "256"}, {"LAYER_DTYPE": "fp16"}, {"LAYER_DTYPE": "bf16"}, {"USE_COMPILE": "1"}):
                with patch.dict(os.environ, overrides):
                    with preproc.preprocessing_session(args) as cfg:
                        self.assertFalse(cfg.cache_verified)
                        self.assertFalse(Path(cfg.paths.vit_features).exists())
                        finish_cache(cfg)
                    with preproc.preprocessing_session(args) as cfg:
                        self.assertTrue(cfg.cache_verified)
            with preproc.preprocessing_session(args) as cfg:
                finish_cache(cfg)
            resolve_sam3d_model_config(args.sam3d_ckpt_path).write_text("MODEL:\n  IMAGE_SIZE: [256, 256]\n", encoding="utf-8")
            with preproc.preprocessing_session(args) as cfg:
                self.assertFalse(cfg.cache_verified)
                self.assertFalse(Path(cfg.paths.vit_features).exists())

    def test_extractor_and_identity_share_defaults_and_resolver(self):
        source = Path(__file__).resolve().parents[1] / "hmr4d/utils/preproc/sam3d_extractor.py"
        spec = importlib.util.spec_from_file_location("review_sam3d_extractor", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp, fake_video_io(tmp), patch.dict(os.environ, {}, clear=True):
            args = fixture_args(tmp)
            preproc._resource_identities(args, Path(tmp))
            expected = sam3d_runtime_settings(args.sam3d_root, args.sam3d_ckpt_path)
            fake_module = SimpleNamespace(SAM3DBodyEstimator=lambda **kwargs: SimpleNamespace(),
                                          load_sam_3d_body=lambda *a, **kw: (SimpleNamespace(), SimpleNamespace()))
            cls = module.SAM3DExtractor
            with patch.dict("sys.modules", {"sam_3d_body": fake_module}), \
                 patch.object(cls, "_setup_sam3d_import", return_value=Path(args.sam3d_root)), \
                 patch.object(cls, "_disable_extra_bbox_padding_for_preprocessed_boxes"), \
                 patch.object(cls, "_patch_model_to_expose_pose_token"):
                extractor = cls(args.sam3d_ckpt_path, args.sam3d_mhr_path, device="cpu", expected_runtime_settings=expected)
                self.assertEqual(extractor.runtime_settings, expected)
                self.assertEqual(extractor.model_config_path, resolve_sam3d_model_config(args.sam3d_ckpt_path))
                with patch.dict(os.environ, {"IMG_SIZE": "256"}):
                    with self.assertRaisesRegex(RuntimeError, "effective settings changed"):
                        cls(args.sam3d_ckpt_path, args.sam3d_mhr_path, device="cpu", expected_runtime_settings=expected)


class CameraPoseTests(unittest.TestCase):
    def test_dpvo_rigid_inverse_identity_and_rotated_poses(self):
        with tempfile.TemporaryDirectory() as tmp, cache_directory_lock(tmp):
            half = 2 ** -.5
            quats = torch.tensor([[1, 0, 0, 0], [half, 0, 0, half], [half, half, 0, 0]], dtype=torch.float32)
            centers = torch.tensor([[1, 0, 0], [1, 2, 3], [-3, 4, 5]], dtype=torch.float32)
            trajectory = torch.cat((centers, quats[:, [1, 2, 3, 0]]), dim=1)
            path = Path(tmp) / "slam.pt"
            torch.save(trajectory, path)
            cfg = SimpleNamespace(output_dir=tmp, video_path="unused", paths=SimpleNamespace(slam=path), static_cam=False, use_dpvo=True)
            data = {}
            with patch.object(preproc, "get_video_lwh", return_value=(3, 32, 32)):
                add_video_camera_data(data, cfg)
            c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
            c2w[:, :3, :3] = quaternion_to_matrix(quats).numpy()
            c2w[:, :3, 3] = centers.numpy()
            np.testing.assert_allclose(data["T_w2c"] @ c2w, np.repeat(np.eye(4)[None], 3, axis=0), atol=1e-6)
            homogeneous = np.concatenate((centers.numpy(), np.ones((3, 1))), axis=1)
            np.testing.assert_allclose(np.einsum("tij,tj->ti", data["T_w2c"], homogeneous), [[0, 0, 0, 1]] * 3, atol=1e-6)
            np.testing.assert_array_equal(data["T_w2c"][0, :3, 3], [-1, 0, 0])

    def test_static_and_simplevo_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, cache_directory_lock(tmp):
            cfg = SimpleNamespace(output_dir=tmp, video_path="unused", paths=SimpleNamespace(slam=Path(tmp) / "slam.pt"), static_cam=True, use_dpvo=False)
            expected = torch.eye(4).repeat(3, 1, 1)
            with patch.object(preproc, "get_video_lwh", return_value=(3, 32, 32)):
                data = {}
                add_video_camera_data(data, cfg)
                np.testing.assert_array_equal(data["T_w2c"], expected.numpy())
                expected[:, :3, 3] = torch.tensor([2., -3., 4.])
                torch.save(expected, cfg.paths.slam)
                cfg.static_cam = False
                add_video_camera_data(data, cfg)
                np.testing.assert_array_equal(data["T_w2c"], expected.numpy())

    def test_zero_quaternion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, cache_directory_lock(tmp):
            path = Path(tmp) / "slam.pt"
            torch.save(torch.zeros(3, 7), path)
            cfg = SimpleNamespace(output_dir=tmp, video_path="unused", paths=SimpleNamespace(slam=path), static_cam=False, use_dpvo=True)
            with patch.object(preproc, "get_video_lwh", return_value=(3, 32, 32)):
                with self.assertRaisesRegex(ValueError, "Zero-length"):
                    add_video_camera_data({}, cfg)


def cache_worker(directory, args, barrier, attempted, owner):
    result_path = Path(directory) / ("owner.json" if owner else "contender.json")
    def report(*values):
        atomic_write_json(result_path, {"result": values})

    try:
        with fake_video_io(directory):
            if owner:
                with preproc.preprocessing_session(args) as cfg:
                    finish_cache(cfg, Path(args.video).read_bytes())
                    barrier.wait(timeout=30)
                    if not attempted.wait(timeout=30):
                        raise RuntimeError("Contender did not attempt cache access")
                    # Read after manifest publication while another process tries
                    # to invalidate the same directory. The lock must still hold.
                    report("owner", Path(cfg.paths.vit_features).read_text(), Path(cfg.video_path).read_text())
            else:
                barrier.wait(timeout=30)
                try:
                    with preproc.preprocessing_session(args):
                        report("unexpected_access")
                except RuntimeError as exc:
                    report("rejected", str(exc))
                finally:
                    attempted.set()
    except BaseException as exc:
        report("error", repr(exc))


def crashed_lock_worker(directory, ready):
    with cache_directory_lock(directory):
        ready.set()
        os._exit(0)


class CacheConcurrencyTests(unittest.TestCase):
    def test_two_processes_same_stem_and_same_identity(self):
        for same_input in (False, True):
            with self.subTest(same_input=same_input), tempfile.TemporaryDirectory() as tmp:
                args_a = fixture_args(tmp, "hmr2", "a/input.mp4")
                args_b = fixture_args(tmp, "hmr2", "a/input.mp4" if same_input else "b/input.mp4")
                context = multiprocessing.get_context("spawn")
                barrier, attempted = context.Barrier(2), context.Event()
                processes = [context.Process(target=cache_worker, args=(tmp, args, barrier, attempted, owner))
                             for args, owner in ((args_a, True), (args_b, False))]
                for process in processes:
                    process.start()
                try:
                    for process in processes:
                        process.join(timeout=45)
                        self.assertEqual(process.exitcode, 0)
                    messages = [json.loads((Path(tmp) / name).read_text(encoding="utf-8"))["result"]
                                for name in ("owner.json", "contender.json")]
                    messages = {message[0]: message[1:] for message in messages}
                    self.assertEqual(set(messages), {"owner", "rejected"}, messages)
                    self.assertIn("already in use", messages["rejected"][0])
                    self.assertEqual(messages["owner"], ["a/input.mp4", "a/input.mp4"])
                    # Once the reader releases ownership, sequential reuse or
                    # invalidation works for the contender's own source.
                    with fake_video_io(tmp), preproc.preprocessing_session(args_b) as cfg:
                        self.assertEqual(bool(cfg.cache_verified), same_input)
                        self.assertEqual(Path(cfg.video_path).read_bytes(), Path(args_b.video).read_bytes())
                        if not same_input:
                            self.assertFalse(Path(cfg.paths.vit_features).exists())
                finally:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=10)

    def test_lock_release_on_exception_and_process_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "intentional"):
                with cache_directory_lock(tmp):
                    raise ValueError("intentional")
            with cache_directory_lock(tmp):
                require_cache_lock(tmp)
            context = multiprocessing.get_context("spawn")
            ready = context.Event()
            process = context.Process(target=crashed_lock_worker, args=(tmp, ready))
            process.start()
            try:
                self.assertTrue(ready.wait(timeout=30))
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
                with cache_directory_lock(tmp):
                    require_cache_lock(tmp)
            finally:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)

    def test_unlocked_entrypoints_reject_access(self):
        with tempfile.TemporaryDirectory() as tmp, fake_video_io(tmp):
            args = fixture_args(tmp, "hmr2")
            with self.assertRaisesRegex(RuntimeError, "requires cache_directory_lock"):
                preproc.build_preprocess_cfg(args)
            with preproc.preprocessing_session(args) as cfg:
                finish_cache(cfg)
            for action in (preproc.run_preprocess, preproc.load_data_dict,
                           lambda cfg: preproc.load_camera_K(cfg, 3, 32, 32),
                           lambda cfg: add_video_camera_data({}, cfg)):
                with self.assertRaisesRegex(RuntimeError, "requires cache_directory_lock"):
                    action(cfg)

    def test_force_recompute_preserves_user_camera_calibration(self):
        with tempfile.TemporaryDirectory() as tmp, fake_video_io(tmp):
            args = fixture_args(tmp, "hmr2")
            with preproc.preprocessing_session(args) as cfg:
                calibration = Path(cfg.preprocess_dir) / "camera_K.pt"
                calibration.write_bytes(b"user calibration")
                finish_cache(cfg)
            args.force_recompute = True
            with preproc.preprocessing_session(args) as cfg:
                self.assertEqual(calibration.read_bytes(), b"user calibration")
                self.assertFalse(Path(cfg.paths.vit_features).exists())


if __name__ == "__main__":
    unittest.main()
