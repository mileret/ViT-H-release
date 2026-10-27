from typing import Any, Dict
import numpy as np
from pathlib import Path
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from einops import rearrange, einsum
from hmr4d.configs import MainStore, builds

from hmr4d.utils.geo_transform import compute_T_ayfz2ay, apply_T_on_points
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo.augment_noisy_pose import (
    get_wham_aug_kp3d,
    get_visible_mask,
    get_invisible_legs_mask,
    randomly_occlude_lower_half,
    randomly_modify_hands_legs,
)
from hmr4d.utils.geo.hmr_cam import perspective_projection, normalize_kp2d, safely_render_x3d_K, get_bbx_xys

from hmr4d.utils.video_io_utils import save_video
from hmr4d.utils.vis.cv2_utils import draw_bbx_xys_on_image_batch
from hmr4d.utils.geo.flip_utils import flip_smplx_params, avg_smplx_aa
from hmr4d.model.offline.utils.postprocess import pp_static_joint, pp_static_joint_cam, process_ik
from hmr4d.model.offline.utils.obs_joints import select_coco17_no_nose_ears


class OfflinePL(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        optimizer=None,
        scheduler_cfg=None,
        #ignored_weights_prefix=["smplx", "pipeline.endecoder"],
        ignored_weights_prefix=["smplx"],
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg

        # Options
        self.ignored_weights_prefix = ignored_weights_prefix

        # The test step is the same as validation
        self.test_step = self.predict_step = self.validation_step

        # SMPLX
        self.smplx = make_smplx("supermotion_v437coco17")

    def _build_synthetic_obs(self, batch, do_augment=True, do_bbox_augment=True):
        """Build human 2D observations from paired AMASS SMPL parameters.

        This is used during training and also during validation for paired G1-AMASS
        datasets, where `kp2d`/`bbx_xys` are placeholders.
        """
        with torch.no_grad():
            gt_verts437, gt_j3d = self.smplx(**batch["smpl_params_c"])

        i_x2d = safely_render_x3d_K(gt_verts437, batch["K_fullimg"], thr=0.3)
        bbx_xys = get_bbx_xys(i_x2d, do_augment=do_bbox_augment)

        if do_augment:
            noisy_j3d = gt_j3d + get_wham_aug_kp3d(gt_j3d.shape[:2]).to(gt_j3d)
            noisy_j3d = randomly_modify_hands_legs(noisy_j3d)
            obs_i_j2d = perspective_projection(noisy_j3d, batch["K_fullimg"])
            j2d_visible_mask = get_visible_mask(gt_j3d.shape[:2]).to(gt_j3d.device)
            j2d_visible_mask[noisy_j3d[..., 2] < 0.3] = False
            legs_invisible_mask = get_invisible_legs_mask(gt_j3d.shape[:2]).to(gt_j3d.device)
            j2d_visible_mask[legs_invisible_mask] = False
        else:
            noisy_j3d = gt_j3d
            obs_i_j2d = perspective_projection(gt_j3d, batch["K_fullimg"])
            j2d_visible_mask = gt_j3d[..., 2] >= 0.3

        obs_kp2d = torch.cat([obs_i_j2d, j2d_visible_mask[:, :, :, None].float()], dim=-1)
        obs = normalize_kp2d(obs_kp2d, bbx_xys)
        obs[~j2d_visible_mask] = 0
        obs = self._select_obs_joints(obs)

        return {
            "gt_verts437": gt_verts437,
            "gt_j3d": gt_j3d,
            "bbx_xys": bbx_xys,
            "obs_kp2d": obs_kp2d,
            "j2d_visible_mask": j2d_visible_mask,
            "obs": obs,
        }

    def _build_soma_obs(self, batch, do_augment=True, do_bbox_augment=True):
        """Build 2D observations from Bones Seed/SOMA 3D joints.

        Bones Seed does not provide SMPL body_pose/betas/transl, so the AMASS
        SMPLX synthetic-observation path cannot be used. The dataset provides
        `soma_obs_joints_w`: the 14 joints after dropping nose/ears, in the same
        observation order expected by the 14-joint network.
        """
        joints_w = batch["soma_obs_joints_w"].float()  # (B, L, 14, 3), world y-up
        R_w2c = batch["T_w2c"][..., :3, :3].float()
        t_w2c = batch["T_w2c"][..., :3, 3].float()
        joints_c = torch.einsum("blij,blkj->blki", R_w2c, joints_w) + t_w2c[:, :, None]

        obs_i_j2d = perspective_projection(joints_c, batch["K_fullimg"])
        j2d_visible_mask = joints_c[..., 2] >= 0.3
        bbx_xys = get_bbx_xys(obs_i_j2d, do_augment=do_bbox_augment)

        if do_augment:
            noise = torch.randn_like(obs_i_j2d) * 4.0
            obs_i_j2d = obs_i_j2d + noise
            drop_prob = torch.rand(j2d_visible_mask.shape, device=j2d_visible_mask.device) < 0.03
            j2d_visible_mask = j2d_visible_mask & ~drop_prob

        obs_kp2d = torch.cat([obs_i_j2d, j2d_visible_mask[..., None].float()], dim=-1)
        obs = normalize_kp2d(obs_kp2d, bbx_xys)
        obs[~j2d_visible_mask] = 0

        return {
            "gt_j3d": joints_c,
            "bbx_xys": bbx_xys,
            "obs_kp2d": obs_kp2d,
            "j2d_visible_mask": j2d_visible_mask,
            "obs": obs,
        }

    def _build_training_obs(self, batch, do_augment=True, do_bbox_augment=True):
        source_is_soma = batch.get("source_is_soma")
        if source_is_soma is None:
            if "soma_obs_joints_w" in batch:
                out = self._build_soma_obs(
                    batch, do_augment=do_augment, do_bbox_augment=do_bbox_augment
                )
                out["obs_already_selected"] = True
                return out
            out = self._build_synthetic_obs(
                batch, do_augment=do_augment, do_bbox_augment=do_bbox_augment
            )
            out["obs_already_selected"] = False
            return out

        source_is_soma = source_is_soma.bool().reshape(-1)
        if bool(source_is_soma.all().item()):
            out = self._build_soma_obs(
                batch, do_augment=do_augment, do_bbox_augment=do_bbox_augment
            )
            out["obs_already_selected"] = True
            return out
        if not bool(source_is_soma.any().item()):
            out = self._build_synthetic_obs(
                batch, do_augment=do_augment, do_bbox_augment=do_bbox_augment
            )
            out["obs_already_selected"] = False
            return out

        # A genuinely mixed batch needs per-sample routing.  In particular,
        # the old key-presence check would send every sample through SOMA as
        # soon as one Bones sample introduced ``soma_obs_joints_w``.
        B = int(batch["length"].shape[0])

        def slice_batch(value, indices):
            if isinstance(value, dict):
                return {k: slice_batch(v, indices) for k, v in value.items()}
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == B:
                return value.index_select(0, indices)
            if isinstance(value, list) and len(value) == B:
                ids = indices.detach().cpu().tolist()
                return [value[i] for i in ids]
            return value

        mask_bbx_xys = batch["mask"]["bbx_xys"]
        if not torch.is_tensor(mask_bbx_xys):
            mask_bbx_xys = torch.full_like(source_is_soma, bool(mask_bbx_xys))
        else:
            mask_bbx_xys = mask_bbx_xys.bool().reshape(-1)

        groups = [
            # Bones/SOMA has no stored image bbox and uses its own 14-joint 3D source.
            (source_is_soma, "soma", True),
            # Real-video SMPL samples retain their precomputed bbox.
            ((~source_is_soma) & mask_bbx_xys, "smpl", False),
            # AMASS-like SMPL samples synthesize and augment a bbox online.
            ((~source_is_soma) & (~mask_bbx_xys), "smpl", True),
        ]

        merged = None
        for group_mask, source_kind, augment_bbox in groups:
            indices = torch.nonzero(group_mask, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            sub_batch = slice_batch(batch, indices)
            if source_kind == "soma":
                part = self._build_soma_obs(
                    sub_batch, do_augment=do_augment, do_bbox_augment=augment_bbox
                )
            else:
                part = self._build_synthetic_obs(
                    sub_batch, do_augment=do_augment, do_bbox_augment=augment_bbox
                )
                # Bring regular COCO17 samples into the same 14-joint order as SOMA.
                part["gt_j3d"] = select_coco17_no_nose_ears(part["gt_j3d"])
                part["obs_kp2d"] = select_coco17_no_nose_ears(part["obs_kp2d"])
                part["j2d_visible_mask"] = select_coco17_no_nose_ears(
                    part["j2d_visible_mask"][..., None]
                ).squeeze(-1)

            if merged is None:
                merged = {
                    "gt_j3d": part["gt_j3d"].new_zeros((B, *part["gt_j3d"].shape[1:])),
                    "bbx_xys": part["bbx_xys"].new_zeros((B, *part["bbx_xys"].shape[1:])),
                    "obs_kp2d": part["obs_kp2d"].new_zeros((B, *part["obs_kp2d"].shape[1:])),
                    "j2d_visible_mask": part["j2d_visible_mask"].new_zeros(
                        (B, *part["j2d_visible_mask"].shape[1:])
                    ),
                }
            for key in merged:
                merged[key].index_copy_(0, indices, part[key].to(merged[key]))

        if merged is None:
            raise RuntimeError("Mixed-source observation routing produced no samples")
        merged["obs"] = normalize_kp2d(merged["obs_kp2d"], merged["bbx_xys"])
        merged["obs"][~merged["j2d_visible_mask"]] = 0
        merged["obs_already_selected"] = True
        return merged

    def _select_obs_joints(self, obs):
        n_joints = int(getattr(self.pipeline.denoiser3d, "obs_num_joints", 14))
        if n_joints == 14:
            return select_coco17_no_nose_ears(obs)
        if n_joints == obs.shape[-2]:
            return obs
        raise ValueError(f"Model expects {n_joints} obs joints, got {obs.shape[-2]}.")

    def _needs_synthetic_eval_obs(self, batch):
        if "soma_obs_joints_w" in batch:
            return True
        if "smpl_params_c" not in batch:
            return False
        if "kp2d" not in batch or "bbx_xys" not in batch:
            return True
        return batch["kp2d"].abs().sum() == 0 or batch["bbx_xys"].abs().sum() == 0

    def training_step(self, batch, batch_idx):
        B, F = batch["length"].shape[0], int(batch["length"].max().item())

        # Create augmented noisy-obs : gt_j3d(coco17).
        
        
        
        mask_bbx_xys = batch["mask"]["bbx_xys"]
        all_have_stored_bbox = bool(mask_bbx_xys.all().item()) if torch.is_tensor(mask_bbx_xys) else bool(mask_bbx_xys)
        synth = self._build_training_obs(
            batch,
            do_augment=True,
            # Real-video sources use their precomputed bbox as-is.  Only
            # sources without stored image bboxes (AMASS/Bones) augment it.
            do_bbox_augment=not all_have_stored_bbox,
        )
        gt_verts437 = synth.get("gt_verts437")
        gt_j3d = synth["gt_j3d"]
        if gt_j3d.shape[-2] == 17:
            root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
        elif gt_j3d.shape[-2] == 14:
            # COCO17_NO_NOSE_EARS_IDS keeps left/right hip at slots 8 and 9.
            root_ = gt_j3d[:, :, [8, 9], :].mean(-2, keepdim=True)
        else:
            root_ = gt_j3d[:, :, :1, :]
        batch["gt_j3d"] = gt_j3d
        batch["gt_cr_coco17"] = gt_j3d - root_
        if gt_verts437 is not None:
            batch["gt_c_verts437"] = gt_verts437
            batch["gt_cr_verts437"] = gt_verts437 - root_

        # bbx_xys
        # Geometry helpers run under Lightning autocast during training, so
        # synthetic boxes may be bf16 while dataloader tensors are fp32.
        # Keep observation preprocessing in the stored-bbox dtype/device: in
        # particular, indexed assignment does not perform dtype promotion.
        bbx_xys = synth["bbx_xys"].to(batch["bbx_xys"])
        if False:  # trust image bbx_xys seems better
            batch["bbx_xys"] = bbx_xys
        else:
            batch["bbx_xys"][~mask_bbx_xys] = bbx_xys[~mask_bbx_xys]
        if False:  # visualize bbx_xys from an iPhone view
            render_w, render_h = 120, 160  # iphone main-lens 24mm 3:4
            ratio = render_w / 1528
            offset = torch.tensor([764 - 500, 1019 - 500]).to(i_x2d)
            i_x2d_render = (i_x2d + offset).clone()
            i_x2d_render = (i_x2d_render * ratio).long().clone()
            torch.clamp_(i_x2d_render[..., 0], 0, render_w - 1)
            torch.clamp_(i_x2d_render[..., 1], 0, render_h - 1)
            bbx_xys_render = bbx_xys.clone()
            bbx_xys_render[..., :2] += offset
            bbx_xys_render *= ratio

            output_dir = Path("outputs/simulated_bbx_xys")
            output_dir.mkdir(parents=True, exist_ok=True)
            video_list = []
            for bid in range(B):
                images = torch.zeros(F, render_h, render_w, 3, device=i_x2d.device)
                for fid in range(F):
                    images[fid, i_x2d_render[bid, fid, :, 1], i_x2d_render[bid, fid, :, 0]] = 255

                images = draw_bbx_xys_on_image_batch(bbx_xys_render[bid].cpu().numpy(), images.cpu().numpy())
                images = np.stack(images).astype("uint8")  # (L, H, W, 3)
                images[:, 0, :] = np.array([255, 255, 255])
                images[:, -1, :] = np.array([255, 255, 255])
                images[:, :, 0] = np.array([255, 255, 255])
                images[:, :, -1] = np.array([255, 255, 255])
                video_list.append(images)

            # stack videos
            video_output = []
            for i in range(0, len(video_list), 4):
                if i + 4 <= len(video_list):
                    video_output.append(np.concatenate(video_list[i : i + 4], axis=2))
            video_output = np.concatenate(video_output, axis=1)
            save_video(video_output, output_dir / f"{batch_idx}.mp4", fps=30, quality=5)

        # Normalize the synthetic 2D observation with the *final* bbox that is
        # also passed to cliff_cam.  For real-video datasets (BEDLAM,
        # BEDLAM2, H36M), mask["bbx_xys"] is True, so this is the precomputed
        # image bbox without online bbox augmentation.  AMASS/Bones have no
        # stored bbox (mask=False), so they keep the online augmented bbox.
        obs_kp2d = synth["obs_kp2d"].to(batch["bbx_xys"])
        batch["obs"] = normalize_kp2d(obs_kp2d, batch["bbx_xys"])
        batch["obs"][~synth["j2d_visible_mask"]] = 0
        if not bool(synth.get("obs_already_selected", False)):
            batch["obs"] = self._select_obs_joints(batch["obs"])

        if True:  # Use some detected vitpose (presave data) for non-SOMA samples.
            prob = 0.5
            source_is_soma = batch.get("source_is_soma")
            if source_is_soma is None:
                source_is_soma = torch.zeros(B, dtype=torch.bool, device=batch["obs"].device)
            else:
                source_is_soma = source_is_soma.to(batch["obs"].device).bool().reshape(-1)
            mask_real_vitpose = (
                (torch.rand(B, device=batch["obs"].device) < prob)
                & batch["mask"]["vitpose"].to(batch["obs"].device).bool()
                & (~source_is_soma)
            )
            real_obs = self._select_obs_joints(normalize_kp2d(batch["kp2d"], batch["bbx_xys"]))
            real_obs = real_obs.to(batch["obs"])
            batch["obs"][mask_real_vitpose] = real_obs[mask_real_vitpose]

        # Set untrusted frames to False
        batch["obs"][~batch["mask"]["valid"]] = 0

        if False:  # wis3d
            wis3d = make_wis3d(name="debug-aug-kp3d")
            add_motion_as_lines(gt_j3d[0], wis3d, name="gt_j3d", skeleton_type="coco17")
            add_motion_as_lines(noisy_j3d[0], wis3d, name="noisy_j3d", skeleton_type="coco17")

        # f_imgseq: apply random aug on offline extracted features
        # f_imgseq = batch["f_imgseq"] + torch.randn_like(batch["f_imgseq"]) * 0.1
        # f_imgseq[~batch["mask"]["f_imgseq"]] = 0
        # batch["f_imgseq"] = f_imgseq.clone()

        # Forward and get loss
        outputs = self.pipeline.forward(batch, train=True)

        # Log
        log_kwargs = {
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "batch_size": B,
            "sync_dist": True,
        }
        # log_kwargs_quiet: same as log_kwargs but without prog_bar (used for
        # nan_count metrics — they'd clutter the prog bar but should still go
        # to WandB so user can plot per-step bad-element counts).
        log_kwargs_quiet = {**log_kwargs, "prog_bar": False}
        self.log("train/loss", outputs["loss"], **log_kwargs)
        for k, v in outputs.items():
            if not isinstance(v, torch.Tensor) or v.ndim != 0:
                continue
            if k.endswith("_nan_count") or k.endswith("_count"):
                
                self.log(f"train/{k}", v, **log_kwargs_quiet)
            elif "_loss" in k:
                self.log(f"train/{k}", v, **log_kwargs)

        # NaN/Inf step detector: pipeline now nan_to_num's per-element NaN to a
        # 10 penalty, so total loss won't be NaN. We instead read the
        # `*_nan_count` keys (count of NaN/Inf elements before replacement) to
        # surface which loss had upstream NaN at this step.
        bad = []
        for k, v in outputs.items():
            if k.endswith("_nan_count") and isinstance(v, torch.Tensor) and v.ndim == 0:
                n = int(v.item())
                if n > 0:
                    bad.append(f"{k.replace('_nan_count', '')}={n}")
        loss_val = outputs["loss"].item() if torch.is_tensor(outputs["loss"]) else float(outputs["loss"])
        loss_finite = np.isfinite(loss_val)
        if bad or not loss_finite:
            print(
                f"[NaN/Inf TRAIN] step={self.global_step} epoch={self.current_epoch} "
                f"loss={loss_val:.4f} bad_elems={bad}",
                flush=True,
            )
            # One-shot deep diagnosis on first NaN: distinguish (a) param NaN
            # (weights poisoned → ckpt rollback required), (b) optimizer state
            # NaN (AdamW exp_avg_sq dead → reset optimizer), (c) only forward
            # NaN (single sample / fp16 edge — keep training, fix data).
            if not getattr(self, "_nan_diagnosed", False):
                bad_params = []
                for n_, p in self.named_parameters():
                    if not torch.isfinite(p).all():
                        bad_params.append(n_)
                        if len(bad_params) >= 5:
                            break
                bad_opt_states = []
                try:
                    opt = self.optimizers()
                    opt = opt.optimizer if hasattr(opt, "optimizer") else opt
                    for grp in opt.state.values():
                        for sk, sv in grp.items():
                            if torch.is_tensor(sv) and not torch.isfinite(sv).all():
                                bad_opt_states.append(sk)
                                break
                        if bad_opt_states:
                            break
                except Exception as e:
                    bad_opt_states = [f"<inspect-failed: {e}>"]
                print(
                    f"[NaN DIAG] params_nan={len(bad_params)} (e.g. {bad_params[:3]})  "
                    f"opt_state_nan={bad_opt_states}",
                    flush=True,
                )
                if bad_params:
                    print("[NaN DIAG] → weights poisoned, ROLLBACK from ckpt.", flush=True)
                elif bad_opt_states:
                    print("[NaN DIAG] → AdamW state dead, RESET optimizer (params still ok).", flush=True)
                else:
                    print("[NaN DIAG] → forward-only NaN, likely fp16 edge / single sample.", flush=True)
                self._nan_diagnosed = True

        return outputs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Options & Check
        do_postproc = self.trainer.state.stage == "test"  # Only apply postproc in test
        do_flip_test = "flip_test" in batch
        do_postproc_not_flip_test = do_postproc and not do_flip_test  # later pp when flip_test
        # DEBUG: print val batch keys once to see what's actually in there
        if batch_idx == 0:
            print(f"[VAL DEBUG] batch keys: {sorted(batch.keys())}")
            for k, v in batch.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape={tuple(v.shape)}")
                elif isinstance(v, dict):
                    print(f"  {k}: dict keys={list(v.keys())}")
                elif isinstance(v, list):
                    print(f"  {k}: list len={len(v)}")
                else:
                    print(f"  {k}: {type(v).__name__} = {v!r}")
        assert batch["B"] == 1, "Only support batch size 1 in evalution."

        # ROPE inference
        if self._needs_synthetic_eval_obs(batch):
            synth = self._build_training_obs(batch, do_augment=False, do_bbox_augment=False)
            obs = synth["obs"]
            bbx_xys = synth["bbx_xys"]
        else:
            obs = self._select_obs_joints(normalize_kp2d(batch["kp2d"], batch["bbx_xys"]))
            bbx_xys = batch["bbx_xys"]
        if "mask" in batch:
            mask_valid = batch["mask"]["valid"] if isinstance(batch["mask"], dict) else batch["mask"]
            obs[0, ~mask_valid[0]] = 0

        batch_ = {
            "length": batch["length"],
            "obs": obs,
            "bbx_xys": bbx_xys,
            "K_fullimg": batch["K_fullimg"],
            "cam_angvel": batch["cam_angvel"],
            "f_imgseq": batch["f_imgseq"],
        }
        if "mask" in batch:
            batch_["mask"] = batch["mask"]
        # Pipeline inference contract: needs g1_target (root_pos_0 anchor) plus
        # cam_angvel (already threaded). T_w2c is OPTIONAL — when present
        # (train/val) the pipeline uses the geometry-exact path R_w = R_w2c.T @ R_c;
        # when absent (demo) it falls back to cam_angvel-based reconstruction in
        # GV0 frame. Pass T_w2c through whenever batch has it so val gets exact
        # geometry.
        
        
        # offline_pipeline.py:156-171.
        batch_["g1_target"] = batch.get("g1_target", {})
        if "T_w2c" in batch:
            batch_["T_w2c"] = batch["T_w2c"]
        outputs = self.pipeline.forward(batch_, train=False, postproc=False)
        # B==1 here (asserted above); drop batch dim for downstream consumption
        # (metric callbacks expect single-sequence tensors).
        for k in ("pred_g1_joint_pos", "pred_g1_body_pos_w", "pred_joints_w", "pred_root_pos"):
            if k in outputs:
                outputs[k] = outputs[k][0]

        # Val loss: re-run pipeline on the same data through the loss-computation
        # path (train=True) but disable null_condition so the loss is deterministic
        # (no random 10% null-mask). Wasteful (extra network forward) but cleanest.
        # Build batch_loss = batch + synthesised obs/bbx_xys; encode_g1 inside the
        # loss path needs T_w2c, R_c2gv, g1_target — all already present in batch.
        #
        
        
        
        if not getattr(self, "skip_val_loss", False):
            batch_loss = {**batch, "obs": obs, "bbx_xys": bbx_xys}
            with torch.no_grad():
                loss_out = self.pipeline.forward(batch_loss, train=True, null_condition=False)
            log_kwargs = {"on_epoch": True, "prog_bar": False, "logger": True,
                          "batch_size": 1, "sync_dist": True}
            self.log("val/loss", loss_out["loss"], **log_kwargs)
            log_kwargs_quiet = {**log_kwargs, "prog_bar": False}
            for k, v in loss_out.items():
                if not isinstance(v, torch.Tensor) or v.ndim != 0:
                    continue
                if k.endswith("_nan_count") or k.endswith("_count"):
                    self.log(f"val/{k}", v, **log_kwargs_quiet)
                elif "_loss" in k:
                    self.log(f"val/{k}", v, **log_kwargs)

            # NaN/Inf step detector (val) — read *_nan_count keys (see training_step
            # for rationale).
            bad = []
            for k, v in loss_out.items():
                if k.endswith("_nan_count") and isinstance(v, torch.Tensor) and v.ndim == 0:
                    n = int(v.item())
                    if n > 0:
                        bad.append(f"{k.replace('_nan_count', '')}={n}")
            if bad:
                print(
                    f"[NaN/Inf VAL] step={self.global_step} epoch={self.current_epoch} "
                    f"batch_idx={batch_idx} loss={loss_out['loss'].item():.4f} bad_elems={bad}",
                    flush=True,
                )
        #if do_flip_test:
        if False:
            flip_test = batch["flip_test"]
            obs = self._select_obs_joints(normalize_kp2d(flip_test["kp2d"], flip_test["bbx_xys"]))
            if "mask" in batch:
                obs[0, ~batch["mask"][0]] = 0

            batch_ = {
                "length": batch["length"],
                "obs": obs,
                "bbx_xys": flip_test["bbx_xys"],
                "K_fullimg": batch["K_fullimg"],
                "cam_angvel": flip_test["cam_angvel"],
                    "f_imgseq": flip_test["f_imgseq"],
            }
            if "mask" in batch:
                batch_["mask"] = batch["mask"]
            flipped_outputs = self.pipeline.forward(batch_, train=False)

            # First update incam results
            flipped_outputs["pred_smpl_params_incam"] = {
                k: v[0] for k, v in flipped_outputs["pred_smpl_params_incam"].items()
            }
            smpl_params1 = outputs["pred_smpl_params_incam"]
            smpl_params2 = flip_smplx_params(flipped_outputs["pred_smpl_params_incam"])

            smpl_params_avg = smpl_params1.copy()
            smpl_params_avg["betas"] = (smpl_params1["betas"] + smpl_params2["betas"]) / 2
            smpl_params_avg["body_pose"] = avg_smplx_aa(smpl_params1["body_pose"], smpl_params2["body_pose"])
            smpl_params_avg["global_orient"] = avg_smplx_aa(
                smpl_params1["global_orient"], smpl_params2["global_orient"]
            )
            outputs["pred_smpl_params_incam"] = smpl_params_avg

            # Then update global results
            outputs["pred_smpl_params_global"]["betas"] = smpl_params_avg["betas"]
            outputs["pred_smpl_params_global"]["body_pose"] = smpl_params_avg["body_pose"]

            # Finally, apply postprocess
            if do_postproc:
                # temporarily recover the original batch-dim
                outputs["pred_smpl_params_global"] = {k: v[None] for k, v in outputs["pred_smpl_params_global"].items()}
                outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(outputs, self.pipeline.endecoder)
                body_pose = process_ik(outputs, self.pipeline.endecoder)
                outputs["pred_smpl_params_global"] = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}

                outputs["pred_smpl_params_global"]["body_pose"] = body_pose[0]
                # outputs["pred_smpl_params_incam"]["body_pose"] = body_pose[0]

        if False:  # wis3d
            wis3d = make_wis3d(name="debug-rich-cap")
            smplx_model = make_smplx("rich-smplx", gender="neutral").cuda()
            gender = batch["gender"][0]
            T_w2ay = batch["T_w2ay"][0]

            # Prediction
            # add_motion_as_lines(outputs_window["pred_ayfz_motion"][bid], wis3d, name="pred_ayfz_motion")

            smplx_out = smplx_model(**pred_smpl_params_global)
            for i in range(len(smplx_out.vertices)):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(smplx_out.vertices[i], smplx_model.bm.faces, name=f"pred-smplx-global")

            # GT (w)
            smplx_models = {
                "male": make_smplx("rich-smplx", gender="male").cuda(),
                "female": make_smplx("rich-smplx", gender="female").cuda(),
            }
            gt_smpl_params = {k: v[0, windows[0]] for k, v in batch["gt_smpl_params"].items()}
            gt_smplx_out = smplx_models[gender](**gt_smpl_params)

            # GT (ayfz)
            smplx_verts_ay = apply_T_on_points(gt_smplx_out.vertices, T_w2ay)
            smplx_joints_ay = apply_T_on_points(gt_smplx_out.joints, T_w2ay)
            T_ay2ayfz = compute_T_ayfz2ay(smplx_joints_ay[:1], inverse=True)[0]  # (4, 4)
            smplx_verts_ayfz = apply_T_on_points(smplx_verts_ay, T_ay2ayfz)  # (F, 22, 3)

            for i in range(len(smplx_verts_ayfz)):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(smplx_verts_ayfz[i], smplx_models[gender].bm.faces, name=f"gt-smplx-ayfz")

            breakpoint()

        if False:  # o3d
            prog_keys = [
                "pred_smpl_progress",
                "pred_localjoints_progress",
                "pred_incam_localjoints_progress",
            ]
            for k in prog_keys:
                if k in outputs_window:
                    seq_out = torch.cat(
                        [v[:, :l] for v, l in zip(outputs_window[k], length)], dim=1
                    )  # (B, P, L, J, 3) -> (P, L, J, 3) -> (P, CL, J, 3)
                    outputs[k] = seq_out[None]

        return outputs

    def configure_optimizers(self):
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler_cfg["scheduler"] is None:
            return optimizer

        scheduler_cfg = dict(self.scheduler_cfg)
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)
        return [optimizer], [scheduler_cfg]

    # ============== Utils ================= #
    def on_save_checkpoint(self, checkpoint) -> None:
        for ig_keys in self.ignored_weights_prefix:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    # Log.info(f"Remove key `{ig_keys}' from checkpoint.")
                    checkpoint["state_dict"].pop(k)

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, "cpu")
        # Standard Lightning checkpoints use ``state_dict``.  DPO policy
        # snapshots use the same VITH parameter names under
        # ``model_state_dict`` and omit Lightning metadata; accept both so the
        # exact DPO weights can be evaluated without rewriting a 1.6GB file.
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            raise KeyError(
                f"Unsupported checkpoint format: expected state_dict or model_state_dict, got {list(checkpoint)[:20]}"
            )
        legacy_cond_prefixes = (
            "learned_pos_params",
            "learned_pos_linear.",
            "embed_noisyobs.",
            "cliffcam_embedder.",
            "cam_angvel_embedder.",
            "imgseq_embedder.",
        )
        for k in list(state_dict.keys()):
            old_prefix = "pipeline.denoiser3d."
            if not k.startswith(old_prefix):
                continue
            suffix = k[len(old_prefix):]
            if suffix.startswith(legacy_cond_prefixes):
                state_dict.setdefault("pipeline.cond_embedder." + suffix, state_dict[k])
                state_dict.pop(k)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            ignored_when_saving = any(k.startswith(ig_keys) for ig_keys in self.ignored_weights_prefix)
            if not ignored_when_saving:
                real_missing.append(k)

        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")


offline_pl = builds(
    OfflinePL,
    pipeline="${pipeline}",
    optimizer="${optimizer}",
    scheduler_cfg="${scheduler_cfg}",
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="offline_pl", node=offline_pl, group="model/offline")
