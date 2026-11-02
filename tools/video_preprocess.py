"""Video preprocessing required by the released offline G1 inference path."""

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm

from hmr4d.utils.geo.hmr_cam import (
    convert_K_to_K4,
    create_camera_sensor,
    estimate_K,
    get_bbx_xys_from_xyxy,
)
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.preproc import (
    Extractor,
    SAM3DExtractor,
    SimpleVO,
    Tracker,
    VitPoseExtractor,
)
from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    resample_video_to_fps,
    save_video,
)
from hmr4d.utils.video_timeline import VIDEO_RESAMPLE_VERSION
from hmr4d.utils.vis.cv2_utils import (
    draw_bbx_xyxy_on_image_batch,
    draw_coco17_skeleton_batch,
)
from tools.preprocess_cache import (
    CACHE_SCHEMA_VERSION,
    artifact_identities,
    atomic_write_json,
    directory_identity,
    file_identity,
    stable_fingerprint,
    validate_complete_manifest,
)

CRF = 23
TARGET_FPS = 30.0


def _first_existing(*paths):
    for path in paths:
        if path is not None and Path(path).is_file():
            return Path(path)
    return Path(paths[0]) if paths and paths[0] is not None else None


def _resolve_sam3d_resources(args):
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "third-party" / "Fast-SAM-3D-Body"
    root = Path(args.sam3d_root).expanduser() if args.sam3d_root else default_root
    ckpt = (
        Path(args.sam3d_ckpt_path).expanduser()
        if args.sam3d_ckpt_path
        else root / "checkpoints/sam-3d-body-dinov3/model.ckpt"
    )
    mhr = (
        Path(args.sam3d_mhr_path).expanduser()
        if args.sam3d_mhr_path
        else root / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    )
    missing = [str(path) for path in (root, ckpt, mhr) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "SAM3D is required by the selected/released model, but these resources "
            f"are missing: {missing}. Install Fast-SAM-3D-Body and pass the "
            "--sam3d-* paths; inference will not fall back to HMR2."
        )
    if not root.is_dir() or not ckpt.is_file() or not mhr.is_file():
        raise FileNotFoundError(
            f"Invalid SAM3D resource types: root={root}, checkpoint={ckpt}, mhr={mhr}"
        )
    return root.resolve(), ckpt.resolve(), mhr.resolve()


def _managed_cache_paths(cfg):
    paths = cfg.paths
    imgfeat_path = Path(paths.vit_features)
    return [
        Path(cfg.video_path),
        Path(paths.bbx),
        Path(paths.bbx_xyxy_video_overlay),
        imgfeat_path,
        imgfeat_path.with_name(f"{imgfeat_path.stem}.meta.json"),
        Path(paths.vitpose),
        Path(paths.vitpose_video_overlay),
        Path(paths.slam),
        Path(cfg.cache_manifest_path),
    ]


def _invalidate_managed_cache(cfg, reason):
    removed = []
    for path in _managed_cache_paths(cfg):
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    if removed:
        Log.warning(f"[Cache] invalidated ({reason}); removed {len(removed)} managed files")
    else:
        Log.info(f"[Cache] no reusable cache ({reason})")


def _resource_identities(args, preprocess_dir):
    repo_root = Path(__file__).resolve().parents[1]
    yolo = repo_root / "inputs/checkpoints/yolo/yolov8x.pt"
    vitpose = _first_existing(
        repo_root / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
        repo_root / "inputs/checkpoints/vitpose/vitpose-h-multi-coco-001.pth",
    )
    resources = {
        "yolo": file_identity(yolo, required=True),
        "vitpose": file_identity(vitpose, required=True),
    }
    if args.imgfeat_encoder == "sam3d":
        root, ckpt, mhr = _resolve_sam3d_resources(args)
        args.sam3d_root = str(root)
        args.sam3d_ckpt_path = str(ckpt)
        args.sam3d_mhr_path = str(mhr)
        resources.update(
            {
                "sam3d_checkpoint": file_identity(ckpt, required=True),
                "sam3d_mhr": file_identity(mhr, required=True),
                "sam3d_repository": directory_identity(
                    root,
                    (
                        "checkpoints/sam-3d-body-dinov3/model_config.yaml",
                        "sam_3d_body/__init__.py",
                    ),
                    required=True,
                ),
            }
        )
    elif args.imgfeat_encoder == "hmr2":
        hmr2 = _first_existing(
            repo_root / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
            repo_root / "inputs/checkpoints/hmr2/epoch=10-step=25000-001.ckpt",
        )
        resources["hmr2"] = file_identity(hmr2, required=True)
    else:
        raise ValueError(f"Unknown imgfeat_encoder: {args.imgfeat_encoder}")

    if args.use_dpvo:
        resources["dpvo_checkpoint"] = file_identity(
            repo_root / "inputs/checkpoints/dpvo/dpvo.pth", required=True
        )
        resources["dpvo_config"] = file_identity(
            repo_root / "third-party/DPVO/config/default.yaml", required=True
        )
    resources["camera_K"] = file_identity(preprocess_dir / "camera_K.pt", required=False)
    return resources


def build_preprocess_cfg(args):
    """Build preprocessing config, validate cache identity, and stage at 30 FPS."""
    source_video = Path(args.video).expanduser().resolve()
    if not source_video.is_file():
        raise FileNotFoundError(f"Input video not found: {source_video}")

    output_dir = Path(args.output_root) / source_video.stem
    preprocess_dir = output_dir / "preprocess"
    staged_video = output_dir / "0_input_video.mp4"
    manifest_path = preprocess_dir / "cache_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    resources = _resource_identities(args, preprocess_dir)
    source_identity = file_identity(source_video, required=True)
    cache_config = {
        "target_fps": TARGET_FPS,
        "resample_version": VIDEO_RESAMPLE_VERSION,
        "imgfeat_encoder": str(args.imgfeat_encoder),
        "model_imgfeat_encoder": getattr(args, "model_imgfeat_encoder", None),
        "sam3d_img_ds": float(args.sam3d_img_ds),
        "sam3d_batch_size": int(args.sam3d_batch_size),
        "tracker": {"type": "yolov8", "base_enlarge": 1.2},
        "pose": {"type": "vitpose_huge_coco_256x192", "img_ds": 0.5},
        "camera": {
            "method": "static" if args.static_cam else ("dpvo" if args.use_dpvo else "simplevo"),
            "f_mm": args.f_mm,
            "simplevo": {"scale": 0.5, "step": 8, "method": "sift"},
            "dpvo": {"resize": 0.5},
        },
        "verbose_overlays": bool(args.verbose),
        "resources": resources,
    }
    cache_identity = {"source": source_identity, "config": cache_config}
    cache_fingerprint = stable_fingerprint(cache_identity)

    cfg = OmegaConf.create(
        {
            "video_name": source_video.stem,
            "output_root": str(args.output_root),
            "output_dir": str(output_dir),
            "preprocess_dir": str(preprocess_dir),
            "video_path": str(staged_video),
            "source_video": str(source_video),
            "target_fps": TARGET_FPS,
            "static_cam": bool(args.static_cam),
            "verbose": bool(args.verbose),
            "use_dpvo": bool(args.use_dpvo),
            "f_mm": args.f_mm,
            "imgfeat_encoder": args.imgfeat_encoder,
            "model_imgfeat_encoder": getattr(args, "model_imgfeat_encoder", None),
            "force_recompute": bool(getattr(args, "force_recompute", False)),
            "force_recompute_imgfeat": bool(args.force_recompute_imgfeat),
            "sam3d_ckpt_path": args.sam3d_ckpt_path,
            "sam3d_mhr_path": args.sam3d_mhr_path,
            "sam3d_root": args.sam3d_root,
            "sam3d_img_ds": float(args.sam3d_img_ds),
            "sam3d_batch_size": int(args.sam3d_batch_size),
            "cache_manifest_path": str(manifest_path),
            "cache_fingerprint": cache_fingerprint,
            "cache_identity": cache_identity,
            "cache_config": cache_config,
            "paths": {
                "bbx": str(preprocess_dir / "bbx.pt"),
                "bbx_xyxy_video_overlay": str(
                    preprocess_dir / "bbx_xyxy_video_overlay.mp4"
                ),
                "vit_features": str(preprocess_dir / "vit_features.pt"),
                "vitpose": str(preprocess_dir / "vitpose.pt"),
                "vitpose_video_overlay": str(
                    preprocess_dir / "vitpose_video_overlay.mp4"
                ),
                "slam": str(preprocess_dir / "slam_results.pt"),
            },
        }
    )

    cache_ok, cache_reason, cached_manifest = validate_complete_manifest(
        manifest_path, cache_fingerprint
    )
    if cfg.force_recompute:
        cache_ok = False
        cache_reason = "--force-recompute requested"
    if not cache_ok:
        _invalidate_managed_cache(cfg, cache_reason)

    Log.info(f"[Output Dir] {output_dir}")
    if cache_ok:
        timeline = cached_manifest.get("timeline")
        Log.info(f"[Cache] verified complete preprocessing cache: {manifest_path}")
    else:
        Log.info(f"[Resample] {source_video} -> {staged_video} at {TARGET_FPS:g} FPS")
        timeline = resample_video_to_fps(
            source_video,
            staged_video,
            target_fps=TARGET_FPS,
            crf=CRF,
        )
    length, width, height = get_video_lwh(staged_video)
    if int(timeline["target_frames"]) != length:
        raise RuntimeError(
            f"Staged timeline mismatch: manifest/resampler={timeline['target_frames']}, "
            f"decoded={length}"
        )
    cfg.timeline = timeline
    cfg.cache_verified = bool(cache_ok)
    Log.info(f"[Input] {source_video}")
    Log.info(
        f"[Timeline] source_frames={timeline['source_decoded_frames']}, "
        f"source_duration={timeline['source_duration_seconds']:.6f}s, "
        f"target_frames={length}, target_fps={TARGET_FPS:g}, "
        f"strategy={timeline['strategy']}"
    )
    Log.info(f"[Staged] (L, W, H) = ({length}, {width}, {height})")

    return cfg


@torch.no_grad()
def run_preprocess(cfg):
    """Run tracking, 2D pose, image features, and optional visual odometry."""
    Log.info("[Preprocess] Start")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths

    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()
        bbx_xys = get_bbx_xys_from_xyxy(
            bbx_xyxy, base_enlarge=1.2
        ).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx, map_location="cpu")["bbx_xys"]
        Log.info(f"[Preprocess] bbx from {paths.bbx}")

    if cfg.verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx, map_location="cpu")["bbx_xyxy"]
        overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(overlay, paths.bbx_xyxy_video_overlay)

    if not Path(paths.vitpose).exists():
        extractor = VitPoseExtractor()
        vitpose = extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del extractor
    else:
        vitpose = torch.load(paths.vitpose, map_location="cpu")
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")

    if cfg.verbose:
        video = read_video_np(video_path)
        overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(overlay, paths.vitpose_video_overlay)

    imgfeat_path = Path(paths.vit_features)
    meta_path = imgfeat_path.with_name(f"{imgfeat_path.stem}.meta.json")
    encoder = str(cfg.imgfeat_encoder)
    expected_meta = {
        "encoder": encoder,
        "feature_dim": 1024,
        "img_ds": 0.5 if encoder == "hmr2" else float(cfg.sam3d_img_ds),
        "cache_fingerprint": str(cfg.cache_fingerprint),
        "encoder_resources": OmegaConf.to_container(
            cfg.cache_config.resources,
            resolve=True,
        ),
    }

    if bool(cfg.force_recompute_imgfeat):
        imgfeat_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        Log.info("[Cache] --force-recompute-imgfeat removed only image-feature cache files")
    recompute = not imgfeat_path.exists()
    if imgfeat_path.exists() and not recompute:
        if not meta_path.exists():
            raise RuntimeError(
                f"Image-feature cache has no encoder metadata: {imgfeat_path}. "
                "Use a fresh --output-root or --force-recompute-imgfeat."
            )
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (cached_meta.get(key), value)
            for key, value in expected_meta.items()
            if cached_meta.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Image-feature cache mismatch at {imgfeat_path}: {mismatches}. "
                "Use a fresh --output-root or --force-recompute-imgfeat."
            )

    if recompute:
        if encoder == "sam3d":
            extractor = SAM3DExtractor(
                checkpoint_path=cfg.sam3d_ckpt_path,
                mhr_path=cfg.sam3d_mhr_path,
                sam3d_root=cfg.sam3d_root,
                device="cuda:0",
            )
            features = extractor.extract_video_features(
                video_path,
                bbx_xys,
                img_ds=float(cfg.sam3d_img_ds),
                batch_size=int(cfg.sam3d_batch_size),
            )
        elif encoder == "hmr2":
            extractor = Extractor()
            features = extractor.extract_video_features(video_path, bbx_xys)
        else:
            raise ValueError(f"Unknown imgfeat_encoder: {encoder}")

        if features.ndim != 2 or int(features.shape[-1]) != 1024:
            raise RuntimeError(
                f"Unexpected {encoder} feature shape: {tuple(features.shape)}; "
                "expected (F, 1024)"
            )
        torch.save(features, imgfeat_path)
        meta_path.write_text(
            json.dumps(
                {
                    **expected_meta,
                    "frames": int(features.shape[0]),
                    "dtype": str(features.dtype),
                    "video_path": str(video_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        del extractor
    else:
        Log.info(f"[Preprocess] verified {encoder} features at {imgfeat_path}")

    if not cfg.static_cam and not Path(paths.slam).exists():
        if not cfg.use_dpvo:
            simple_vo = SimpleVO(
                video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm
            )
            torch.save(simple_vo.compute(), paths.slam)
        else:
            from hmr4d.utils.preproc.slam import SLAMModel

            length, width, height = get_video_lwh(video_path)
            K_fullimg = load_camera_K(cfg, length, width, height)
            intrinsics = convert_K_to_K4(K_fullimg[0])
            buffer = min(max(int(length) + 32, 1024), 4000)
            slam = SLAMModel(
                video_path,
                width,
                height,
                intrinsics,
                buffer=buffer,
                resize=0.5,
            )
            bar = tqdm(total=length, desc="DPVO")
            while True:
                tracked = slam.track()
                if not tracked:
                    break
                bar.update()
            bar.close()
            torch.save(slam.process(), paths.slam)
    elif not cfg.static_cam:
        Log.info(f"[Preprocess] slam from {paths.slam}")

    length, _, _ = get_video_lwh(video_path)
    bbx_cache = torch.load(paths.bbx, map_location="cpu")
    checks = {
        "bbx": int(bbx_cache["bbx_xys"].shape[0]),
        "vitpose": int(torch.load(paths.vitpose, map_location="cpu").shape[0]),
        "features": int(torch.load(paths.vit_features, map_location="cpu").shape[0]),
    }
    if not cfg.static_cam:
        checks["camera"] = int(np.asarray(torch.load(paths.slam, map_location="cpu")).shape[0])
    mismatched_lengths = {name: value for name, value in checks.items() if value != length}
    if mismatched_lengths:
        raise RuntimeError(
            f"Preprocessing cache length mismatch for staged video length {length}: "
            f"{mismatched_lengths}"
        )

    completed_paths = [
        cfg.video_path,
        paths.bbx,
        paths.vitpose,
        paths.vit_features,
        meta_path,
    ]
    if not cfg.static_cam:
        completed_paths.append(paths.slam)
    if cfg.verbose:
        completed_paths.extend(
            [paths.bbx_xyxy_video_overlay, paths.vitpose_video_overlay]
        )
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "complete",
        "cache_fingerprint": str(cfg.cache_fingerprint),
        "identity": OmegaConf.to_container(cfg.cache_identity, resolve=True),
        "timeline": OmegaConf.to_container(cfg.timeline, resolve=True),
        "frame_counts": checks,
        "artifacts": artifact_identities(completed_paths),
    }
    atomic_write_json(cfg.cache_manifest_path, manifest)
    Log.info(f"[Cache] wrote complete manifest: {cfg.cache_manifest_path}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time() - tic:.2f}s")


def load_camera_K(cfg, length, width, height):
    """Load calibrated full-image intrinsics or construct a default matrix."""
    path = Path(cfg.preprocess_dir) / "camera_K.pt"
    if path.is_file():
        K0 = torch.as_tensor(
            torch.load(path, map_location="cpu"), dtype=torch.float32
        )
        if K0.shape == (3, 3):
            K_fullimg = K0.unsqueeze(0).expand(length, -1, -1).clone()
        elif K0.shape == (length, 3, 3):
            K_fullimg = K0
        else:
            raise ValueError(
                f"{path}: expected (3,3) or ({length},3,3), got {tuple(K0.shape)}"
            )
    elif cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(
            length, 1, 1
        )
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    if (
        not torch.isfinite(K_fullimg).all()
        or (K_fullimg[:, 0, 0] <= 0).any()
        or (K_fullimg[:, 1, 1] <= 0).any()
    ):
        raise ValueError(f"Invalid camera focal length in {path}")

    cx, cy = K_fullimg[:, 0, 2], K_fullimg[:, 1, 2]
    if (
        ((cx - width / 2).abs() > width * 0.1).any()
        or ((cy - height / 2).abs() > height * 0.1).any()
    ):
        raise ValueError(
            f"Camera K in {path} does not match video {width}x{height}: "
            f"principal point ({cx[0].item():.1f}, {cy[0].item():.1f})."
        )
    return K_fullimg


def load_data_dict(cfg):
    """Load cached preprocessing results into the offline model input dict."""
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        trajectory = np.asarray(
            torch.load(cfg.paths.slam, map_location="cpu")
        )
        if cfg.use_dpvo:
            if trajectory.shape != (length, 7) or not np.isfinite(
                trajectory
            ).all():
                raise ValueError(
                    f"Expected DPVO (T,7), got {trajectory.shape}: "
                    f"{cfg.paths.slam}"
                )
            quat = torch.from_numpy(
                trajectory[:, [6, 3, 4, 5]]
            ).float()
            if (torch.linalg.vector_norm(quat, dim=-1) < 1e-8).any():
                raise ValueError(f"Zero-length quaternion: {cfg.paths.slam}")
            R_w2c = quaternion_to_matrix(quat).mT
        else:
            if trajectory.shape != (length, 4, 4) or not np.isfinite(
                trajectory
            ).all():
                raise ValueError(
                    f"Expected SimpleVO (T,4,4), got {trajectory.shape}: "
                    f"{cfg.paths.slam}"
                )
            R_w2c = torch.from_numpy(trajectory[:, :3, :3])

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(
            cfg.paths.bbx, map_location="cpu"
        )["bbx_xys"],
        "kp2d": torch.load(cfg.paths.vitpose, map_location="cpu"),
        "K_fullimg": load_camera_K(cfg, length, width, height),
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(
            cfg.paths.vit_features, map_location="cpu"
        ),
    }
    expected_shapes = {
        "bbx_xys": (length, 3),
        "kp2d": (length, 17, 3),
        "K_fullimg": (length, 3, 3),
        "cam_angvel": (length, 6),
        "f_imgseq": (length, 1024),
    }
    invalid = {
        key: (tuple(data[key].shape), shape)
        for key, shape in expected_shapes.items()
        if tuple(data[key].shape) != shape
    }
    if invalid:
        raise RuntimeError(f"Invalid preprocessing tensor shapes: {invalid}")
    for key in expected_shapes:
        if not torch.isfinite(torch.as_tensor(data[key])).all():
            raise RuntimeError(f"Preprocessing tensor contains non-finite values: {key}")
    return data
