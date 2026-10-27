"""Run the released offline ViT-H-to-G1 model on one video."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video to G1 motion with the released offline model."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Offline model checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="Output NPZ path.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/preprocess"),
        help="Directory for reusable video preprocessing caches.",
    )
    parser.add_argument("-s", "--static-cam", action="store_true", help="Skip visual odometry.")
    parser.add_argument("--use-dpvo", action="store_true", help="Use DPVO instead of SimpleVO.")
    parser.add_argument("--f-mm", type=int, default=None, help="Full-frame focal length in mm.")
    parser.add_argument("--verbose", action="store_true", help="Save preprocessing overlays.")
    parser.add_argument(
        "--imgfeat-encoder",
        choices=("hmr2", "sam3d"),
        default="hmr2",
        help="Image feature encoder.",
    )
    parser.add_argument("--force-recompute-imgfeat", action="store_true")
    parser.add_argument("--sam3d-ckpt-path", type=str, default=None)
    parser.add_argument("--sam3d-mhr-path", type=str, default=None)
    parser.add_argument("--sam3d-root", type=str, default=None)
    parser.add_argument("--sam3d-img-ds", type=float, default=1.0)
    parser.add_argument("--sam3d-batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--singular-rewrite",
        choices=("none", "blender"),
        default="none",
        help="Optional visualization-only joint-angle rewrite.",
    )
    parser.add_argument(
        "--exp",
        default="offline/g1_training_set_exp1",
    )
    parser.add_argument("--model-latent-dim", type=int, default=1024)
    parser.add_argument("--model-num-layers", type=int, default=16)
    parser.add_argument("--model-num-heads", type=int, default=16)
    return parser.parse_args()
def build_preprocess_cfg(args):
    """Build the focused video-preprocessing configuration."""
    from tools.video_preprocess import build_preprocess_cfg as _build

    return _build(args)


def _to_numpy_f32(x):
    import torch

    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _identity_T_w2c(length):
    T = np.repeat(np.eye(4, dtype=np.float32)[None], int(length), axis=0)
    return T


def add_video_camera_data(data, cfg):
    """Store full video-camera matrices for Blender render, without changing model inputs."""
    import torch
    from pytorch3d.transforms import quaternion_to_matrix
    from tools.video_preprocess import get_video_lwh

    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        T_w2c = _identity_T_w2c(length)
    else:
        traj = torch.load(cfg.paths.slam, map_location="cpu")
        if cfg.use_dpvo:
            traj_quat = torch.as_tensor(traj[:, [6, 3, 4, 5]], dtype=torch.float32)
            R_w2c = quaternion_to_matrix(traj_quat).mT.detach().cpu().numpy().astype(np.float32)
            T_w2c = _identity_T_w2c(length)
            T_w2c[:, :3, :3] = R_w2c
            if traj.shape[-1] >= 3:
                T_w2c[:, :3, 3] = np.asarray(traj[:, :3], dtype=np.float32)
        else:
            traj_np = np.asarray(traj, dtype=np.float32)
            if traj_np.ndim == 3 and traj_np.shape[1:] == (4, 4):
                T_w2c = traj_np.copy()
            elif traj_np.ndim == 3 and traj_np.shape[1:] == (3, 4):
                T_w2c = _identity_T_w2c(length)
                T_w2c[:, :3, :] = traj_np
            else:
                T_w2c = _identity_T_w2c(length)
                T_w2c[:, :3, :3] = traj_np[:, :3, :3]

    data["T_w2c"] = T_w2c[:length].astype(np.float32)
    data["image_wh"] = np.asarray([width, height], dtype=np.int32)


def make_prep_from_video_data(data, window, device):
    """Convert original VITH demo data dict into G1 pipeline inference tensors."""
    import torch

    from hmr4d.model.offline.utils.obs_joints import select_coco17_no_nose_ears
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d

    length = int(data["length"].item() if torch.is_tensor(data["length"]) else data["length"])

    obs = select_coco17_no_nose_ears(normalize_kp2d(data["kp2d"], data["bbx_xys"]))
    prep = {
        # Keep full-video tensors on CPU. Long videos can have large 1024-D ViT
        # features, so each sliding window is moved to GPU only when needed.
        "obs_full": obs.to(dtype=torch.float32).cpu(),
        "bbx_xys_full": data["bbx_xys"].to(dtype=torch.float32).cpu(),
        "K_fullimg_full": data["K_fullimg"].to(dtype=torch.float32).cpu(),
        "cam_angvel_full": data["cam_angvel"].to(dtype=torch.float32).cpu(),
        "f_imgseq_full": data["f_imgseq"].to(dtype=torch.float32).cpu(),
        "window": int(window),
        "total_len": length,
    }
    if length <= window:
        chunks_meta = [(0, length)]
    else:
        starts = [0]
        while starts[-1] + window < length:
            starts.append(starts[-1] + window - 1)
        chunks_meta = [(s, min(window, length - s)) for s in starts]
    prep["chunks_meta"] = chunks_meta
    print(f"[Info] Video length={length}, {len(chunks_meta)} chunks (window={window}, overlap=1)")
    return prep, length
def infer_video(pipeline, prep, total_len, device="cuda", singular_rewrite="none"):
    import torch
    import torch.nn.functional as F
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
    from tools.inference_utils import reduce_singular_chains

    obs_full = prep["obs_full"]
    bbx_xys_full = prep["bbx_xys_full"]
    K_fullimg_full = prep["K_fullimg_full"]
    cam_angvel_full = prep["cam_angvel_full"]
    f_imgseq_full = prep["f_imgseq_full"]
    chunks_meta = prep["chunks_meta"]
    window = prep["window"]
    if prep.get("full_sequence", False):
        chunks_meta = [(0, total_len)]
        window = total_len
        print(
            f"[Info] full-sequence mode: one forward, length={total_len}; "
            f"network attention max_len remains {prep.get('trained_max_len', 120)}"
        )

    joint_pos_out = torch.zeros(total_len, 29, device=device)
    root_pos_out = torch.zeros(total_len, 3, device=device)
    root_quat_out = torch.zeros(total_len, 4, device=device)
    body_pos_out = None

    def pad_window(x, pad_len):
        if pad_len <= 0:
            return x
        return torch.cat([x, x[-1:].expand(pad_len, *x.shape[1:])], dim=0)

    prev_last_pos = None
    prev_last_quat = None

    for chunk_idx, (start, actual_len) in enumerate(chunks_meta):
        end = start + actual_len
        pad_len = window - actual_len
        obs = pad_window(obs_full[start:end], pad_len)
        bbx_xys = pad_window(bbx_xys_full[start:end], pad_len)
        K_fullimg = pad_window(K_fullimg_full[start:end], pad_len)
        cam_angvel = pad_window(cam_angvel_full[start:end], pad_len)
        f_imgseq = pad_window(f_imgseq_full[start:end], pad_len)

        obs = obs.to(device)
        bbx_xys = bbx_xys.to(device)
        K_fullimg = K_fullimg.to(device)
        cam_angvel = cam_angvel.to(device)
        f_imgseq = f_imgseq.to(device)

        batch = {
            "length": torch.tensor([actual_len], device=device, dtype=torch.long),
            "obs": obs.unsqueeze(0),
            "bbx_xys": bbx_xys.unsqueeze(0),
            "K_fullimg": K_fullimg.unsqueeze(0),
            "cam_angvel": cam_angvel.unsqueeze(0),
            "f_imgseq": f_imgseq.unsqueeze(0),
        }
        if prev_last_pos is not None:
            g1_body_pos_w = torch.zeros(1, 1, 1, 3, device=device, dtype=prev_last_pos.dtype)
            g1_body_pos_w[0, 0, 0] = prev_last_pos
            batch["g1_target"] = {"g1_body_pos_w": g1_body_pos_w}

        outputs = pipeline.forward(batch, train=False)
        pred_jp = outputs["pred_g1_joint_pos"][0, :actual_len]
        pred_rp = outputs["pred_root_pos"][0, :actual_len]
        pred_rq = outputs["pred_root_quat_w"][0, :actual_len]
        pred_body = outputs.get("pred_g1_body_pos_w")
        if pred_body is not None:
            pred_body = pred_body[0, :actual_len]
            if body_pos_out is None:
                body_pos_out = torch.zeros(total_len, pred_body.shape[1], 3, device=device)

        if prev_last_quat is not None:
            R_prev_last = quaternion_to_matrix(prev_last_quat)
            R_curr_first = quaternion_to_matrix(pred_rq[0])
            delta_R = R_prev_last @ R_curr_first.transpose(-1, -2)

            disp = pred_rp - pred_rp[0:1]
            pred_rp = pred_rp[0:1] + disp @ delta_R.transpose(-1, -2)

            R_curr_all = quaternion_to_matrix(pred_rq)
            pred_rq = matrix_to_quaternion(delta_R.unsqueeze(0) @ R_curr_all)

        if chunk_idx == 0:
            write_slice = slice(start, end)
            src_slice = slice(0, actual_len)
        else:
            write_slice = slice(start + 1, end)
            src_slice = slice(1, actual_len)
        joint_pos_out[write_slice] = pred_jp[src_slice]
        root_pos_out[write_slice] = pred_rp[src_slice]
        root_quat_out[write_slice] = pred_rq[src_slice]
        if body_pos_out is not None and pred_body is not None:
            body_pos_out[write_slice] = pred_body[src_slice]

        prev_last_pos = pred_rp[actual_len - 1].detach()
        prev_last_quat = pred_rq[actual_len - 1].detach()

    # Match infer_amass2g1.py output convention: convert internal y-up frame to G1 z-up URDF world.
    M_pos = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        device=device,
        dtype=root_pos_out.dtype,
    )
    root_pos_out = root_pos_out @ M_pos
    body_pos_z = body_pos_out @ M_pos if body_pos_out is not None else None
    R_yup = quaternion_to_matrix(root_quat_out)
    root_quat_out = matrix_to_quaternion(M_pos.transpose(-1, -2) @ R_yup)
    root_quat_out = F.normalize(root_quat_out, dim=-1)

    assert singular_rewrite in ("none", "blender"), singular_rewrite
    joint_pos_np = joint_pos_out.cpu().numpy()
    if singular_rewrite == "blender":
        joint_pos_np = reduce_singular_chains(joint_pos_np)

    results = {
        "joint_pos": joint_pos_np,
        "root_pos_w": root_pos_out.cpu().numpy(),
        "root_quat_w": root_quat_out.cpu().numpy(),
    }
    if body_pos_z is not None:
        results["body_pos_w"] = body_pos_z.cpu().numpy()
    return results


def main():
    import torch

    from tools.video_preprocess import load_data_dict, run_preprocess
    from tools.inference_utils import load_pipeline

    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Input video not found: {args.video}")
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if args.window != 120:
        raise ValueError("This release was trained with a 120-frame window; use --window 120.")

    _set_seed(args.seed)
    args.video = str(args.video)
    args.ckpt = str(args.ckpt)
    args.output_root = str(args.output_root)
    if args.output is None:
        output = Path("outputs/inference") / f"{Path(args.video).stem}_g1.npz"
    else:
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = build_preprocess_cfg(args)
    run_preprocess(cfg)
    data = load_data_dict(cfg)
    add_video_camera_data(data, cfg)
    data["source_name"] = Path(args.video).stem
    data["source_video"] = args.video
    data["preprocess_dir"] = str(cfg.preprocess_dir)
    data["imgfeat_encoder"] = args.imgfeat_encoder
    data["fps"] = 30.0

    prep, total_len = make_prep_from_video_data(
        data,
        window=args.window,
        device=args.device,
    )
    pipeline = load_pipeline(
        args.ckpt,
        exp=args.exp,
        device=args.device,
        overrides=[
            f"network.latent_dim={args.model_latent_dim}",
            f"network.num_layers={args.model_num_layers}",
            f"network.num_heads={args.model_num_heads}",
        ],
    )

    with torch.no_grad():
        results = infer_video(
            pipeline,
            prep,
            total_len,
            device=args.device,
            singular_rewrite=args.singular_rewrite,
        )

    np.savez(
        output,
        joint_pos=results["joint_pos"],
        root_pos_w=results["root_pos_w"],
        root_quat_w=results["root_quat_w"],
        **({"body_pos_w": results["body_pos_w"]} if "body_pos_w" in results else {}),
        K_fullimg=_to_numpy_f32(data.get("K_fullimg")),
        T_w2c=_to_numpy_f32(data.get("T_w2c")),
        image_wh=np.asarray(data["image_wh"], dtype=np.int32),
        source_video=args.video,
        preprocess_dir=str(data["preprocess_dir"]),
        ckpt=args.ckpt,
        exp=args.exp,
        imgfeat_encoder=args.imgfeat_encoder,
        model_latent_dim=np.array(args.model_latent_dim, dtype=np.int32),
        model_num_layers=np.array(args.model_num_layers, dtype=np.int32),
        model_num_heads=np.array(args.model_num_heads, dtype=np.int32),
        fps=np.array(30.0, dtype=np.float32),
        source_name=data["source_name"],
    )
    print(f"Saved: {output}")
    print(f"joint_pos: {results['joint_pos'].shape}")
    print(f"root_pos_w: {results['root_pos_w'].shape}")
    print(f"root_quat_w: {results['root_quat_w'].shape}")


if __name__ == "__main__":
    main()
