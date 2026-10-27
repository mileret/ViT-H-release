import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from pathlib import Path
from .vitpose_pytorch import build_model
from .vitfeat_extractor import get_batch
from tqdm import tqdm

from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps
from hmr4d.utils.geo_transform import cvt_p2d_from_pm1_to_i
from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17


class VitPoseExtractor:
    def __init__(self, tqdm_leave=True):
        ckpt_path = Path("inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth")
        if not ckpt_path.exists():
            fallback_ckpt_path = ckpt_path.with_name("vitpose-h-multi-coco-001.pth")
            if fallback_ckpt_path.exists():
                ckpt_path = fallback_ckpt_path
        self.pose = build_model("ViTPose_huge_coco_256x192", str(ckpt_path))
        self.pose.cuda().eval()

        self.flip_test = True
        self.tqdm_leave = tqdm_leave

    @torch.no_grad()
    def extract(self, video_path, bbx_xys, img_ds=0.5):
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        L, _, H, W = imgs.shape  # (L, 3, H, W)
        batch_size = 16
        vitpose = []
        for j in tqdm(range(0, L, batch_size), desc="ViTPose", leave=self.tqdm_leave):
            # Heat map
            imgs_batch = imgs[j : j + batch_size, :, :, 32:224].cuda()
            if self.flip_test:
                heatmap, heatmap_flipped = self.pose(torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)).chunk(2)
                heatmap_flipped = flip_heatmap_coco17(heatmap_flipped)
                heatmap = (heatmap + heatmap_flipped) * 0.5
                del heatmap_flipped
            else:
                heatmap = self.pose(imgs_batch.clone())  # (B, J, 64, 48)

            if False:
                # Get joint
                bbx_xys_batch = bbx_xys[j : j + batch_size].cuda()
                method = "hard"
                if method == "hard":
                    kp2d_pm1, conf = get_heatmap_preds(heatmap)
                elif method == "soft":
                    kp2d_pm1, conf = get_heatmap_preds(heatmap, soft=True)

                # Convert 64, 48 to 64, 64
                kp2d_pm1[:, :, 0] *= 24 / 32
                kp2d = cvt_p2d_from_pm1_to_i(kp2d_pm1, bbx_xys_batch[:, None])
                kp2d = torch.cat([kp2d, conf], dim=-1)

            else:  # postprocess from mmpose
                bbx_xys_batch = bbx_xys[j : j + batch_size]
                heatmap = heatmap.clone().cpu().numpy()
                center = bbx_xys_batch[:, :2].numpy()
                scale = (torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200).numpy()
                preds, maxvals = keypoints_from_heatmaps(heatmaps=heatmap, center=center, scale=scale, use_udp=True)
                kp2d = np.concatenate((preds, maxvals), axis=-1)
                kp2d = torch.from_numpy(kp2d)

            vitpose.append(kp2d.detach().cpu().clone())

        vitpose = torch.cat(vitpose, dim=0).clone()  # (F, 17, 3)
        return vitpose


def get_heatmap_preds(heatmap, normalize_keypoints=True, thr=0.0, soft=False):
    """
    heatmap: (B, J, H, W)
    """
    assert heatmap.ndim == 4, "batch_images should be 4-ndim"

    B, J, H, W = heatmap.shape
    heatmaps_reshaped = heatmap.reshape((B, J, -1))

    maxvals, idx = torch.max(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((B, J, 1))
    idx = idx.reshape((B, J, 1))
    preds = idx.repeat(1, 1, 2).float()
    preds[:, :, 0] = (preds[:, :, 0]) % W
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / W)

    pred_mask = torch.gt(maxvals, thr).repeat(1, 1, 2)
    pred_mask = pred_mask.float()
    preds *= pred_mask

    # soft peak
    if soft:
        patch_size = 5
        patch_half = patch_size // 2
        patches = torch.zeros((B, J, patch_size, patch_size)).to(heatmap)
        default_patch = torch.zeros(patch_size, patch_size).to(heatmap)
        default_patch[patch_half, patch_half] = 1
        for b in range(B):
            for j in range(17):
                x, y = preds[b, j].int()
                if x >= patch_half and x <= W - patch_half and y >= patch_half and y <= H - patch_half:
                    patches[b, j] = heatmap[
                        b, j, y - patch_half : y + patch_half + 1, x - patch_half : x + patch_half + 1
                    ]
                else:
                    patches[b, j] = default_patch

        dx, dy = soft_patch_dx_dy(patches)
        preds[:, :, 0] += dx
        preds[:, :, 1] += dy

    if normalize_keypoints:  # to [-1, 1]
        preds[:, :, 0] = preds[:, :, 0] / (W - 1) * 2 - 1
        preds[:, :, 1] = preds[:, :, 1] / (H - 1) * 2 - 1

    return preds, maxvals


def soft_patch_dx_dy(p):
    """p (B,J,P,P)"""
    p_batch_shape = p.shape[:-2]
    patch_size = p.size(-1)
    temperature = 1.0
    score = F.softmax(p.view(-1, patch_size**2) * temperature, dim=-1)

    # get a offset_grid (BN, P, P, 2) for dx, dy
    offset_grid = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))[::-1]
    offset_grid = torch.stack(offset_grid, dim=-1).float() - (patch_size - 1) / 2
    offset_grid = offset_grid.view(1, 1, patch_size, patch_size, 2).to(p.device)

    score = score.view(*p_batch_shape, patch_size, patch_size)
    dx = torch.sum(score * offset_grid[..., 0], dim=(-2, -1))
    dy = torch.sum(score * offset_grid[..., 1], dim=(-2, -1))

    if False:
        b, j = 0, 0
        print(torch.stack([dx[b, j], dy[b, j]]))
        print(p[b, j])

    return dx, dy


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pth")
    torch.save(data, tmp)
    os.replace(tmp, path)


def _make_shard_path(out_pth, num_shards, shard_idx):
    out_pth = Path(out_pth)
    return out_pth.with_name(f"{out_pth.stem}.shard{shard_idx:02d}-of-{num_shards:02d}{out_pth.suffix}")


def _validate_shard_args(num_shards, shard_idx):
    num_shards = int(num_shards)
    shard_idx = int(shard_idx)
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"shard_idx must be in [0, {num_shards}), got {shard_idx}")
    return num_shards, shard_idx


def _select_keys_for_shard(keys, start=0, limit=0, num_shards=1, shard_idx=0):
    if start:
        keys = keys[int(start) :]
    if limit:
        keys = keys[: int(limit)]
    num_shards, shard_idx = _validate_shard_args(num_shards, shard_idx)
    if num_shards > 1:
        keys = keys[shard_idx::num_shards]
    return keys


def merge_bedlam2_vitpose_shards(out_pth, num_shards, allow_missing_shards=False):
    out_pth = Path(out_pth)
    num_shards, _ = _validate_shard_args(num_shards, 0)
    merged = {}
    merged_errors = {}
    for shard_idx in range(num_shards):
        shard_pth = _make_shard_path(out_pth, num_shards, shard_idx)
        if not shard_pth.exists():
            if allow_missing_shards:
                print(f"[BEDLAM2 ViTPose][merge] missing shard: {shard_pth}")
                continue
            raise FileNotFoundError(shard_pth)
        shard_data = _torch_load_cpu(shard_pth)
        overlap = set(merged).intersection(shard_data)
        if overlap:
            raise RuntimeError(f"duplicate keys while merging {shard_pth}: {len(overlap)}")
        merged.update(shard_data)
        print(f"[BEDLAM2 ViTPose][merge] {shard_pth}: {len(shard_data)} clips")

        shard_err_pth = shard_pth.with_suffix(".errors.pth")
        if shard_err_pth.exists():
            merged_errors.update(_torch_load_cpu(shard_err_pth))

    _save_atomic(out_pth, merged)
    if merged_errors:
        err_pth = out_pth.with_suffix(".errors.pth")
        _save_atomic(err_pth, merged_errors)
        print(f"[BEDLAM2 ViTPose][merge] saved {len(merged_errors)} errors to {err_pth}")
    print(f"[BEDLAM2 ViTPose][merge] saved {len(merged)} clips to {out_pth}")
    return merged


def extract_bedlam2_vitpose(
    bbox_pth,
    out_pth,
    start=0,
    limit=0,
    save_every=20,
    img_ds=0.5,
    conf_key="kp2d",
    num_shards=1,
    shard_idx=0,
):
    """Run ViTPose on BEDLAM2 videos.

    Input is the bbox pth produced for BEDLAM2.  Output is a pth dict:
        key -> {"kp2d": (F, 17, 3), "start_end": (0, F), "mp4_path": str}
    Bedlam2G1Dataset reads this file through vitpose_pth_path.
    """
    bbox_pth = Path(bbox_pth)
    out_pth = Path(out_pth)
    bbox_data = _torch_load_cpu(bbox_pth)
    keys = list(bbox_data.keys())
    keys = _select_keys_for_shard(keys, start=start, limit=limit, num_shards=num_shards, shard_idx=shard_idx)
    if int(num_shards) > 1:
        print(f"[BEDLAM2 ViTPose] shard {int(shard_idx) + 1}/{int(num_shards)}: {len(keys)} clips -> {out_pth}")

    results = {}
    if out_pth.exists():
        results = _torch_load_cpu(out_pth)
        print(f"[BEDLAM2 ViTPose] resume from {out_pth}: {len(results)} done")

    extractor = VitPoseExtractor(tqdm_leave=False)
    processed = 0
    errors = {}
    for key in tqdm(keys, desc="BEDLAM2 ViTPose", dynamic_ncols=True):
        if key in results:
            continue
        rec = bbox_data[key]
        try:
            video_path = rec.get("mp4_path")
            if not video_path:
                raise KeyError(f"missing mp4_path for {key}")
            bbx_xys = rec["bbx_xys"].float()
            valid = rec.get("valid")
            if valid is not None:
                valid = valid.bool()
                bbx_xys = bbx_xys.clone()
                if (~valid).any() and valid.any():
                    first_valid = bbx_xys[valid][0]
                    bbx_xys[~valid] = first_valid
            kp2d = extractor.extract(str(video_path), bbx_xys, img_ds=img_ds)
            results[key] = {
                conf_key: kp2d.float().cpu(),
                "start_end": (0, int(kp2d.shape[0])),
                "mp4_path": str(video_path),
            }
            processed += 1
            if save_every > 0 and processed % int(save_every) == 0:
                _save_atomic(out_pth, results)
        except Exception as exc:
            errors[key] = repr(exc)
            print(f"[BEDLAM2 ViTPose][error] {key}: {exc}")

    _save_atomic(out_pth, results)
    if errors:
        err_path = out_pth.with_suffix(".errors.pth")
        _save_atomic(err_path, errors)
        print(f"[BEDLAM2 ViTPose] saved errors to {err_path}")
    print(f"[BEDLAM2 ViTPose] saved {len(results)} clips to {out_pth}")
    return results


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bedlam2_bbox_pth", type=Path, default=None)
    parser.add_argument("--out_pth", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--img_ds", type=float, default=0.5)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--merge_shards", action="store_true")
    parser.add_argument("--allow_missing_shards", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    out_pth = args.out_pth or Path(
        "/path/to/training_set/bedlam2/part_00/keypoints_00.pth"
    )
    _validate_shard_args(args.num_shards, args.shard_idx)
    if args.merge_shards:
        merge_bedlam2_vitpose_shards(
            out_pth,
            num_shards=args.num_shards,
            allow_missing_shards=args.allow_missing_shards,
        )
        return
    if args.bedlam2_bbox_pth is None:
        raise SystemExit("Pass --bedlam2_bbox_pth to run BEDLAM2 ViTPose extraction.")
    run_out_pth = _make_shard_path(out_pth, args.num_shards, args.shard_idx) if args.num_shards > 1 else out_pth
    extract_bedlam2_vitpose(
        args.bedlam2_bbox_pth,
        run_out_pth,
        start=args.start,
        limit=args.limit,
        save_every=args.save_every,
        img_ds=args.img_ds,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
    )


if __name__ == "__main__":
    main()
