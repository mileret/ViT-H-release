from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from hmr4d.utils.video_io_utils import read_video_np


class SAM3DExtractor:
    """Extract SAM3D-Body pose tokens for VITH's f_imgseq slot.

    The returned tensor is (F, 1024), matching the old HMR2 feature interface.
    HMR2 code stays intact; this class is only a drop-in feature source.
    """

    def __init__(
        self,
        checkpoint_path=None,
        mhr_path=None,
        device="cuda:0",
        tqdm_leave=True,
        feature_dim=1024,
        sam3d_root=None,
    ):
        self.device = device
        self.tqdm_leave = tqdm_leave
        self.feature_dim = int(feature_dim)
        self.sam3d_root = self._setup_sam3d_import(sam3d_root)
        self.is_fast_sam3d = bool(
            self.sam3d_root and "fast-sam-3d-body" in str(self.sam3d_root).lower()
        )
        if self.is_fast_sam3d:
            self._setup_fast_sam3d_env()

        try:
            from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
        except Exception as exc:
            raise ImportError(
                "Cannot import `sam_3d_body`. Install SAM3D-Body and set "
                "`SAM3D_BODY_ROOT`, or pass "
                "`SAM3D_BODY_ROOT` / pass `--sam3d_root` to its repository path."
            ) from exc

        checkpoint_path = checkpoint_path or os.environ.get("SAM3D_CKPT_PATH")
        mhr_path = mhr_path or os.environ.get("SAM3D_MHR_PATH")
        if checkpoint_path is None:
            default_ckpt = Path(__file__).resolve().parents[3] / "third-party" / "sam-3d-body-dinov3" / "model.ckpt"
            if default_ckpt.exists():
                checkpoint_path = str(default_ckpt)
        if mhr_path is None:
            default_mhr = Path(__file__).resolve().parents[3] / "third-party" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
            if default_mhr.exists():
                mhr_path = str(default_mhr)
        if checkpoint_path is None:
            raise FileNotFoundError(
                "SAM3D checkpoint not found. Pass --sam3d_ckpt_path or set SAM3D_CKPT_PATH. "
                "Expected relative path: third-party/sam-3d-body-dinov3/model.ckpt"
            )
        if mhr_path is None:
            raise FileNotFoundError(
                "SAM3D MHR model not found. Pass --sam3d_mhr_path or set SAM3D_MHR_PATH. "
                "Expected relative path: third-party/sam-3d-body-dinov3/assets/mhr_model.pt"
            )

        try:
            model, model_cfg = load_sam_3d_body(
                checkpoint_path,
                device=torch.device(device),
                mhr_path=mhr_path,
            )
        except TypeError:
            # Older local checkouts may not accept mhr_path.
            model, model_cfg = load_sam_3d_body(checkpoint_path, device=torch.device(device))

        # Keep GEM-X semantics: one stable primary body token.
        if hasattr(model, "cfg"):
            try:
                model.cfg.defrost()
                model.cfg.MODEL.DECODER.DO_HAND_DETECT_TOKENS = False
                model.cfg.freeze()
            except Exception:
                pass

        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        self._disable_extra_bbox_padding_for_preprocessed_boxes()
        self._patch_model_to_expose_pose_token()

    @staticmethod
    def _setup_sam3d_import(sam3d_root):
        candidates = []
        if sam3d_root:
            candidates.append(Path(sam3d_root))
        if os.environ.get("SAM3D_BODY_ROOT"):
            candidates.append(Path(os.environ["SAM3D_BODY_ROOT"]))

        repo_root = Path(__file__).resolve().parents[3]
        candidates.extend(
            [
                repo_root / "third-party" / "Fast-SAM-3D-Body-main",
                repo_root / "third-party" / "sam-3d-body",
                repo_root / "third-party" / "sam-3d-body-main",
                repo_root / "third_party" / "Fast-SAM-3D-Body-main",
                repo_root / "third_party" / "sam-3d-body",
                repo_root / "third_party" / "sam-3d-body-main",
                repo_root / "third_party" / "sam_3d_body",
            ]
        )

        for root in candidates:
            if root and root.exists():
                root = root.resolve()
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                return root
        return None

    @staticmethod
    def _setup_fast_sam3d_env():
        """Safe Fast-SAM3D defaults; explicit env/CLI values still win."""
        os.environ.setdefault("IMG_SIZE", "512")
        os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")
        os.environ.setdefault("KEYPOINT_PROMPT_INTERM_INTERVAL", "999")
        os.environ.setdefault("BODY_INTERM_PRED_LAYERS", "0,1,2")
        os.environ.setdefault("HAND_INTERM_PRED_LAYERS", "0,1")
        os.environ.setdefault("MHR_NO_CORRECTIVES", "1")
        os.environ.setdefault("LAYER_DTYPE", "fp32")
        os.environ.setdefault("DEBUG_NAN", "0")
        os.environ.setdefault("DEBUG_HAND_PREP", "0")
        os.environ.setdefault("DEBUG_BACKBONE_INPUT", "0")
        os.environ.setdefault("INTERM_TIMING", "0")

    def _disable_extra_bbox_padding_for_preprocessed_boxes(self):
        """Keep VITH/PHUMA preprocessed 1.2x square boxes unchanged."""
        transform = getattr(self.estimator, "transform", None)
        transforms = getattr(transform, "transforms", None)
        if transforms is None:
            return

        for item in transforms:
            name = item.__class__.__name__
            if name == "GetBBoxCenterScale" and hasattr(item, "padding"):
                item.padding = 1.0
            elif name == "TopdownAffine" and hasattr(item, "fix_square"):
                item.fix_square = True

    def _forward_log_context(self):
        if not self.is_fast_sam3d or os.environ.get("FAST_SAM3D_VERBOSE", "0") == "1":
            return contextlib.nullcontext()
        return open(os.devnull, "w")

    def _patch_model_to_expose_pose_token(self):
        model = self.estimator.model
        if getattr(model, "_offline_pose_token_patched", False):
            return

        original_forward_pose_branch = model.forward_pose_branch

        def _forward_pose_branch_with_pose_token(batch):
            out = original_forward_pose_branch(batch)
            if "pose_token" in out and out["pose_token"] is not None:
                return out

            body_batch_idx = getattr(model, "body_batch_idx", [])
            if len(body_batch_idx) > 0:
                batch_size, num_person = batch["img"].shape[:2]
                keypoints_prompt = torch.zeros((batch_size * num_person, 1, 3)).to(batch["img"])
                keypoints_prompt[:, :, -1] = -2
                tokens_output, _ = model.forward_decoder(
                    out["image_embeddings"][body_batch_idx],
                    init_estimate=None,
                    keypoints=keypoints_prompt[body_batch_idx],
                    prev_estimate=None,
                    condition_info=out["condition_info"][body_batch_idx],
                    batch=batch,
                )
                out["pose_token"] = tokens_output
            else:
                out["pose_token"] = None
            return out

        model.forward_pose_branch = _forward_pose_branch_with_pose_token
        model._offline_pose_token_patched = True

    def _normalize_feature_dim(self, pose_token):
        pose_token = pose_token.detach().float().cpu()
        if pose_token.ndim == 3:
            pose_token = pose_token[:, 0]
        if pose_token.shape[-1] == self.feature_dim:
            return pose_token
        if pose_token.shape[-1] > self.feature_dim:
            return pose_token[..., : self.feature_dim]
        pad = torch.zeros(
            pose_token.shape[0],
            self.feature_dim - pose_token.shape[-1],
            dtype=pose_token.dtype,
        )
        return torch.cat([pose_token, pad], dim=-1)

    def extract_video_features(
        self,
        video_path,
        bbx_xys,
        img_ds=1.0,
        batch_size=16,
        return_dict=False,
    ):
        from sam_3d_body.data.utils.prepare_batch import prepare_batch
        from sam_3d_body.utils import recursive_to

        imgs = read_video_np(video_path, scale=img_ds)
        bbx_xys = torch.as_tensor(bbx_xys).float().cpu().numpy()

        tokens = []
        transls = []
        for i in tqdm(
            range(0, len(imgs), batch_size),
            desc="SAM3D Body Feature",
            leave=self.tqdm_leave,
        ):
            batch_imgs = imgs[i : i + batch_size]
            batch_bbx = bbx_xys[i : i + batch_size]
            model_batch_list = []

            for j, img in enumerate(batch_imgs):
                cx, cy, s = batch_bbx[j]
                s_scaled = s * img_ds
                cx_scaled = cx * img_ds
                cy_scaled = cy * img_ds
                box = np.array(
                    [
                        cx_scaled - s_scaled / 2,
                        cy_scaled - s_scaled / 2,
                        cx_scaled + s_scaled / 2,
                        cy_scaled + s_scaled / 2,
                    ],
                    dtype=np.float32,
                ).reshape(1, 4)
                model_batch_list.append(
                    prepare_batch(img, self.estimator.transform, box, masks=None, masks_score=None)
                )

            collated_batch = {}
            keys = model_batch_list[0].keys()
            for key in keys:
                val0 = model_batch_list[0][key]
                if isinstance(val0, torch.Tensor):
                    collated_batch[key] = torch.cat([d[key] for d in model_batch_list])
                elif isinstance(val0, np.ndarray):
                    collated_batch[key] = np.concatenate([d[key] for d in model_batch_list])
                else:
                    collated_batch[key] = [d[key] for d in model_batch_list]

            collated_batch = recursive_to(collated_batch, self.device)
            self.estimator.model._initialize_batch(collated_batch)
            with torch.no_grad():
                with self._forward_log_context() as quiet_fp:
                    if quiet_fp is None:
                        output = self.estimator.model.forward_step(collated_batch, decoder_type="body")
                    else:
                        with contextlib.redirect_stdout(quiet_fp):
                            output = self.estimator.model.forward_step(collated_batch, decoder_type="body")

            pose_token = output.get("pose_token", None)
            if pose_token is None:
                fallback = output["mhr"]["mhr_model_params"].detach().float().cpu()
                pose_token = fallback
            tokens.append(self._normalize_feature_dim(pose_token))

            if "mhr" in output and "pred_cam_t" in output["mhr"]:
                transls.append(output["mhr"]["pred_cam_t"].detach().float().cpu())

        pose_tokens = torch.cat(tokens, dim=0).clone()
        if not return_dict:
            return pose_tokens

        ret = {"pose_tokens": pose_tokens}
        if transls:
            ret["transls"] = torch.cat(transls, dim=0).clone()
        return ret
