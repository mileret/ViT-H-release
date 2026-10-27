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
    get_video_reader,
    get_writer,
    read_video_np,
    save_video,
)
from hmr4d.utils.vis.cv2_utils import (
    draw_bbx_xyxy_on_image_batch,
    draw_coco17_skeleton_batch,
)

CRF = 23


def build_preprocess_cfg(args):
    """Build the minimal preprocessing config and stage the input video."""
    source_video = Path(args.video)
    if not source_video.is_file():
        raise FileNotFoundError(f"Input video not found: {source_video}")

    length, width, height = get_video_lwh(source_video)
    Log.info(f"[Input] {source_video}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")

    output_dir = Path(args.output_root) / source_video.stem
    preprocess_dir = output_dir / "preprocess"
    staged_video = output_dir / "0_input_video.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create(
        {
            "video_name": source_video.stem,
            "output_root": str(args.output_root),
            "output_dir": str(output_dir),
            "preprocess_dir": str(preprocess_dir),
            "video_path": str(staged_video),
            "static_cam": bool(args.static_cam),
            "verbose": bool(args.verbose),
            "use_dpvo": bool(args.use_dpvo),
            "f_mm": args.f_mm,
            "imgfeat_encoder": args.imgfeat_encoder,
            "force_recompute_imgfeat": bool(args.force_recompute_imgfeat),
            "sam3d_ckpt_path": args.sam3d_ckpt_path,
            "sam3d_mhr_path": args.sam3d_mhr_path,
            "sam3d_root": args.sam3d_root,
            "sam3d_img_ds": float(args.sam3d_img_ds),
            "sam3d_batch_size": int(args.sam3d_batch_size),
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

    Log.info(f"[Output Dir] {output_dir}")
    if (
        not staged_video.exists()
        or get_video_lwh(source_video)[0] != get_video_lwh(staged_video)[0]
    ):
        Log.info(f"[Copy Video] {source_video} -> {staged_video}")
        reader = get_video_reader(source_video)
        writer = get_writer(staged_video, fps=30, crf=CRF)
        for image in tqdm(reader, total=length, desc="Copy"):
            writer.write_frame(image)
        writer.close()
        reader.close()

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
    }

    recompute = bool(cfg.force_recompute_imgfeat) or not imgfeat_path.exists()
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

    return {
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
