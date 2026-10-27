import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
from pathlib import Path
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from hmr4d.utils.net_utils import gaussian_smooth

from hmr4d.model.offline.utils.endecoder import EnDecoder
from hmr4d.model.offline.utils import stats_compose

from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from hmr4d.utils.geo.hmr_cam import (
    compute_bbox_info_bedlam,
    compute_transl_full_cam,
    get_a_pred_cam,
    get_bbx_xys,
    perspective_projection,
    project_to_bi01,
)
from hmr4d.utils.geo.hmr_global import (
    rollout_local_transl_vel,
    get_tgtcoord_rootparam,
)


class Pipeline(nn.Module):
    def __init__(self, args, args_denoiser3d, **kwargs):
        super().__init__()
        self.args = args
        self.weights = args.weights  # loss weights
        self.geometry_loss_type = str(getattr(args, "geometry_loss_type", "mse")).lower()
        self.geometry_huber_delta = float(getattr(args, "geometry_huber_delta", 1.0))
        if self.geometry_loss_type not in {"mse", "huber"}:
            raise ValueError(
                f"Unsupported geometry_loss_type={self.geometry_loss_type!r}; "
                "expected 'mse' or 'huber'."
            )
        if self.geometry_huber_delta <= 0:
            raise ValueError("geometry_huber_delta must be positive.")

        # Networks
        self.denoiser3d = instantiate(args_denoiser3d, _recursive_=False)
        # Log.info(self.denoiser3d)

        # Condition embedder (GEM-X style: condition embedding moved OUT of the
        # per-step denoiser so f_cond is built once & reused across DDIM steps,
        # and a separate unconditional f_uncond can be built for CFG). Sized from
        # the network's *_dim attrs. Used by both diffusion and regression paths.
        from hmr4d.network.offline.relative_transformer import ConditionEmbedder
        self.cond_embedder = ConditionEmbedder(
            latent_dim=self.denoiser3d.latent_dim,
            cliffcam_dim=self.denoiser3d.cliffcam_dim,
            cam_angvel_dim=self.denoiser3d.cam_angvel_dim,
            imgseq_dim=self.denoiser3d.imgseq_dim,
            obs_num_joints=getattr(self.denoiser3d, "obs_num_joints", 14),
            dropout=self.denoiser3d.dropout,
        )

        # Normalizer
        self.endecoder: EnDecoder = instantiate(args.endecoder_opt, _recursive_=False)

        # G1 FK engine — required for cr_j3d / fk_j3d / j2d / transl_c losses.
        # Without this call all FK-dependent losses silently fall back to GT-vs-GT.
        robot_args = getattr(self.args, "robot", None)
        mjcf_file = getattr(robot_args, "mjcf_file", None)
        sampled_vertices_path = getattr(robot_args, "sampled_vertices_path", None)
        if sampled_vertices_path is None:
            sampled_vertices_path = getattr(robot_args, "sampled_vertices_file", None)
        if sampled_vertices_path is None:
            default_vertices_path = (
                Path(__file__).resolve().parents[4]
                / "hmr4d/utils/body_model/g1_sampled_vertices_384.pt"
            )
            if default_vertices_path.exists():
                sampled_vertices_path = str(default_vertices_path)
        if mjcf_file is not None:
            self.endecoder.setup_g1_fk(
                mjcf_file,
                sampled_vertices_path=sampled_vertices_path,
            )
            Log.info(f"[Pipeline] G1 FK initialised from {mjcf_file}")
            if sampled_vertices_path is not None:
                Log.info(f"[Pipeline] G1 sampled vertices loaded from {sampled_vertices_path}")
        else:
            Log.warning("[Pipeline] args.robot.mjcf_file not set; FK losses will be skipped.")

        # Load G1 pred_x stats (189-D VITH format).
        # Falls back to no normalisation when stats_file is absent.
        stats_file = getattr(self.args, "stats_file", None)
        if stats_file is not None:
            try:
                import numpy as _np
                _d = _np.load(stats_file)
                if "pred_mean" in _d and "pred_std" in _d:
                    from hmr4d.model.offline.utils.endecoder import G1_PRED_X_DIM
                    _mean = torch.tensor(_d["pred_mean"]).float()
                    _std  = torch.tensor(_d["pred_std"]).float().clamp(min=1e-6)
                    assert _mean.shape[-1] == G1_PRED_X_DIM, (
                        f"pred_mean dim {_mean.shape[-1]} != G1_PRED_X_DIM {G1_PRED_X_DIM}. "
                        "Regenerate stats with tools/make_g1_pred_stats.py"
                    )
                    self.endecoder.set_g1_pred_stats(_mean, _std)
                    Log.info(f"[Pipeline] Loaded G1 pred_x stats (189-D) from {stats_file}")
                else:
                    # legacy 299-D stats – still register for decode_g1 compatibility
                    self.endecoder.compute_g1_stats_from_npz(stats_file)
                    Log.info(f"[Pipeline] Loaded legacy G1 stats (299-D) from {stats_file}")
            except Exception as e:
                Log.warning(f"[Pipeline] Could not load G1 stats from {stats_file}: {e}. "
                         "Running without normalisation.")
        else:
            Log.warning("[Pipeline] args.stats_file not set – G1 pred_x will NOT be normalised.")

        # Sanity-check: network output_dim must be 189
        from hmr4d.model.offline.utils.endecoder import G1_PRED_X_DIM
        if hasattr(self.denoiser3d, "output_dim"):
            net_dim = int(self.denoiser3d.output_dim)
            if net_dim != G1_PRED_X_DIM:
                raise ValueError(
                    f"[Pipeline] Denoiser output_dim ({net_dim}) != G1_PRED_X_DIM "
                    f"({G1_PRED_X_DIM}). Set network.output_dim={G1_PRED_X_DIM} in your exp yaml."
                )

        # ===== Diffusion (optional, config-gated) ===== #
        
        
        
        
        
        diff_cfg = getattr(self.args, "diffusion", None)
        self.diffusion_enabled = diff_cfg is not None
        if self.diffusion_enabled:
            from hmr4d.utils.diffusion.model_util import create_gaussian_diffusion
            from hmr4d.utils.diffusion.resample import create_named_schedule_sampler

            self.diffusion_cfg = diff_cfg
            self.train_diffusion = create_gaussian_diffusion(diff_cfg, training=True)
            self.test_diffusion = create_gaussian_diffusion(diff_cfg, training=False)
            self.schedule_sampler = create_named_schedule_sampler(
                getattr(diff_cfg, "schedule_sampler_type", "uniform"), self.train_diffusion
            )
            self.ddim_eta = float(getattr(diff_cfg, "ddim_eta", 0.0))
            # CFG: train with sample-level obs-drop (prob cfg_uncond_prob) to teach
            # the unconditional branch; at inference guide with guidance_param
            # (scale=1.0 -> CFG off). ∅ = drop obs (the 2D-pose condition).
            self.cfg_scale = float(getattr(diff_cfg, "guidance_param", 1.0))
            self.cfg_uncond_prob = float(getattr(diff_cfg, "cfg_uncond_prob", 0.1))
            real_values = getattr(diff_cfg, "autoregressive_real_values", None)
            self.autoregressive_real_values = None
            if real_values is not None:
                self.autoregressive_real_values = tuple(int(v) for v in real_values)
                if not self.autoregressive_real_values:
                    raise ValueError("diffusion.autoregressive_real_values cannot be empty")
                if not bool(getattr(self.denoiser3d, "real_embed", False)):
                    raise ValueError(
                        "diffusion.autoregressive_real_values requires network.real_embed=true"
                    )
                max_len = int(getattr(self.denoiser3d, "max_len", 120))
                if any(v < 0 or v >= max_len for v in self.autoregressive_real_values):
                    raise ValueError(
                        "diffusion.autoregressive_real_values must be in "
                        f"[0, {max_len - 1}], got {self.autoregressive_real_values}"
                    )
                Log.info(
                    "[Pipeline] Autoregressive GT-prefix training enabled: "
                    f"real_values={self.autoregressive_real_values} (uniform per sample)"
                )
            
            if hasattr(self.denoiser3d, "xt_dim") and int(self.denoiser3d.xt_dim) != G1_PRED_X_DIM:
                raise ValueError(
                    f"[Pipeline] Denoiser xt_dim ({int(self.denoiser3d.xt_dim)}) != "
                    f"G1_PRED_X_DIM ({G1_PRED_X_DIM}). Set network.xt_dim={G1_PRED_X_DIM}."
                )
            Log.info(
                f"[Pipeline] Diffusion ENABLED: train_steps={self.train_diffusion.num_timesteps}, "
                f"test_steps={self.test_diffusion.num_timesteps}, sampler="
                f"{getattr(diff_cfg, 'schedule_sampler_type', 'uniform')}, ddim_eta={self.ddim_eta}"
            )
        else:
            Log.info("[Pipeline] Diffusion disabled -> regression path.")

        pred_cam_stats_file = getattr(self.args, "pred_cam_stats_file", None)
        if pred_cam_stats_file is not None and hasattr(self.denoiser3d, "set_pred_cam_stats"):
            
            
            try:
                pred_cam_stats = np.load(pred_cam_stats_file)
                self.denoiser3d.set_pred_cam_stats(pred_cam_stats["mean"], pred_cam_stats["std"])
                Log.info(f"[Pipeline] Loaded robot pred_cam stats from {pred_cam_stats_file}")
            except Exception as e:
                raise RuntimeError(
                    f"[Pipeline] Failed to load pred_cam stats from {pred_cam_stats_file}: {e}. "
                    'Documentation removed.'
                    'Documentation removed.'
                ) from e
        elif hasattr(self.denoiser3d, "set_pred_cam_stats"):
            Log.warning(
                'Documentation removed.'
                'Documentation removed.'
            )

        if self.args.normalize_cam_angvel:
            cam_angvel_stats = stats_compose.cam_angvel["manual"]
            self.register_buffer("cam_angvel_mean", torch.tensor(cam_angvel_stats["mean"]), persistent=False)
            self.register_buffer("cam_angvel_std", torch.tensor(cam_angvel_stats["std"]), persistent=False)

    # ========== Training ========== #

    def forward(self, inputs, train=False, postproc=False, static_cam=False, null_condition=None, real=None, real_motion=None):
        """
        Args:
            train: bool — if True, run loss-computation path; if False, run inference.
            null_condition: bool or None — whether to randomly null-mask conditions
                (10% per frame, original VITH augmentation). When None (default),
                follows `train` (legacy behaviour). Set to False during val-loss
                computation so the loss is deterministic.
        """
        length = inputs["length"]  # (B,)
        if null_condition is None:
            null_condition = train

        # ------------------------------------------------------------------
        
        # ------------------------------------------------------------------
        cliff_cam = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])

        f_cam_angvel = inputs.get("cam_angvel", None)
        if f_cam_angvel is not None and getattr(self.args, "normalize_cam_angvel", False):
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std.clamp(min=1e-8)

        f_imgseq = inputs.get("f_imgseq", None)
        f_imgseq_mask = None
        if f_imgseq is not None and "mask" in inputs and isinstance(inputs["mask"], dict):
            f_imgseq_mask = inputs["mask"].get("f_imgseq", None)

        f_condition = {
            "obs": inputs["obs"],
            "f_cliffcam": cliff_cam,
            "f_cam_angvel": f_cam_angvel,
            "f_imgseq": f_imgseq,
            "f_imgseq_mask": f_imgseq_mask,
        }

        
        
        
        B0 = length.shape[0]
        drop_obs = None
        if self.diffusion_enabled and train and getattr(self, "cfg_uncond_prob", 0.0) > 0:
            drop_obs = torch.rand(B0, device=length.device) < self.cfg_uncond_prob

        if null_condition:
            f_condition = self._randomly_set_null_condition(
                f_condition, uncond_prob=0.1, skip_mask=drop_obs
            )

        # ------------------------------------------------------------------
        
        # ------------------------------------------------------------------
        
        
        
        target_x = None
        t_weights = None
        if self.diffusion_enabled and train:
            
            f_cond = self.cond_embedder(**f_condition, drop_obs=drop_obs)  # (B,L,latent)
            target_x = self.endecoder.encode_g1(inputs)          # (B, L, 189) x_start
            t, t_weights = self.schedule_sampler.sample(target_x.shape[0], target_x.device)
            noise = torch.randn_like(target_x)
            x_t = self.train_diffusion.q_sample(target_x, t, noise=noise)
            real = None
            if self.autoregressive_real_values is not None:
                real_choices = torch.as_tensor(
                    self.autoregressive_real_values, device=target_x.device, dtype=torch.long
                )
                real = real_choices[
                    torch.randint(real_choices.numel(), (target_x.shape[0],), device=target_x.device)
                ]
            model_output = self.denoiser3d(
                f_cond=f_cond,
                length=length,
                xt=x_t,
                timesteps=self.train_diffusion._scale_timesteps(t),
                real=real,
                gt_motion=target_x if real is not None else None,
            )
        elif self.diffusion_enabled and not train:
            
            model_output = self._ddim_sample(length, f_condition, real=real, real_motion=real_motion)
        else:
            
            f_cond = self.cond_embedder(**f_condition, drop_obs=drop_obs)
            model_output = self.denoiser3d(f_cond=f_cond, length=length)
        pred_x = model_output["pred_x"]   # (B, L, 189)

        # ------------------------------------------------------------------
        
        # ------------------------------------------------------------------
        if not train:
            decode_dict = self.endecoder.decode_g1_new(pred_x)
            out = {"model_output": model_output, "decode_dict": decode_dict}
            out["static_conf_logits"] = model_output.get("static_conf_logits")

            # FK to recover body positions if engine is available
            if hasattr(self.endecoder, "_g1_humanoid"):
                gt = inputs.get("g1_target", {})
                root_pos_0 = gt.get("g1_body_pos_w", None)
                if root_pos_0 is not None:
                    root_pos_0 = root_pos_0[:, [0], 0]  # (B,1,3)
                else:
                    
                    
                    
                    pelvis_h = float(getattr(
                        getattr(self.args, "robot", None), "default_pelvis_height", 0.793
                    ))
                    B_pred = decode_dict["local_transl_vel"].shape[0]
                    device = decode_dict["local_transl_vel"].device
                    root_pos_0 = torch.tensor(
                        [[0.0, pelvis_h, 0.0]], device=device
                    ).expand(B_pred, 1, 3)

                # Recover world-frame root orientation using GV + cam_angvel,
                # identical to original VITH's inference path.
                # Under precision=16-mixed, decode_dict tensors are Half;
                # cast everything to fp32 before the rotation math.
                go_c_rotmat_f32      = decode_dict["global_orient_rotmat"].float()
                local_transl_vel_f32 = decode_dict["local_transl_vel"].float()
                root_pos_0_f32       = root_pos_0.float()

                go_c_aa_f32  = matrix_to_axis_angle(go_c_rotmat_f32)
                go_gv_aa_f32 = matrix_to_axis_angle(
                    rotation_6d_to_matrix(decode_dict["global_orient_gv_r6d"].float())
                )
                params_w = get_smpl_params_w_Rt_v2(
                    global_orient_gv  = go_gv_aa_f32,
                    local_transl_vel  = local_transl_vel_f32,
                    global_orient_c   = go_c_aa_f32,
                    cam_angvel        = inputs["cam_angvel"].float(),
                    length            = length,
                )
                # get_smpl_params_w_Rt_v2 outputs y-up (ay, SMPL convention).
                
                pred_root_R_w    = axis_angle_to_matrix(params_w["global_orient"])  # (B,L,3,3) y-up
                pred_root_aa_w   = params_w["global_orient"]                        # (B,L,3)   y-up
                pred_root_quat_w = matrix_to_quaternion(pred_root_R_w)              # (B,L,4)   wxyz

                # Translation rollout in y-up using pred_root_aa_w,
                # anchored at root_pos_0 (y-up) from g1_target.
                pred_root_pos = rollout_local_transl_vel(
                    local_transl_vel_f32, pred_root_aa_w, root_pos_0_f32
                )
                # Recover scalar θ in BYD order from [0,θ,0]-encoded rotmat (mirrors dataset.py).
                
                R_bp_inf = decode_dict["body_pose_rotmat"].float()                 # (B,L,29,3,3)
                pred_dof_byd = torch.atan2(R_bp_inf[..., 0, 2], R_bp_inf[..., 0, 0])  # (B,L,29)
                pred_joints_w, _ = self.endecoder.fk_g1(
                    pred_dof_byd,
                    pred_root_pos,
                    pred_root_quat_w,
                    world_up='y',
                )
                out["pred_joints_w"] = pred_joints_w
                out["pred_root_pos"] = pred_root_pos
                # Task 3: expose G1-specific outputs so val_step / metric callbacks
                # can consume them without re-deriving from FK internals.
                # NOTE: per-body quaternions (30 bodies × 4) would need extra FK
                # work; skip until a metric actually needs them.
                out["pred_g1_joint_pos"]  = pred_dof_byd     # (B,L,29) scalar DOF (BYD order)
                out["pred_g1_body_pos_w"] = pred_joints_w    # (B,L,30,3)
                out["pred_root_quat_w"]   = pred_root_quat_w # (B,L,4) wxyz, world frame
            return out

        # ------------------------------------------------------------------
        
        # ------------------------------------------------------------------
        outputs = {"model_output": model_output}
        
        outputs["t_weights"] = t_weights
        total_loss = 0
        mask = inputs["mask"]["valid"]   # (B, L)

        gt = inputs["g1_target"]
        B, L = mask.shape

        # --- 4a. Simple loss: MSE on 189-D normalised pred_x ---
        
        
        
        if target_x is None:
            target_x = self.endecoder.encode_g1(inputs)  # (B, L, 189)
        simple_loss_el = F.mse_loss(pred_x, target_x, reduction="none")
        simple_loss, simple_loss_nan_count = _reduce_loss(
            simple_loss_el, mask[..., None], t_weights=t_weights, fk_gate=None
        )
        total_loss += simple_loss
        outputs["simple_loss"] = simple_loss
        outputs["simple_loss_nan_count"] = simple_loss_nan_count

        # --- 4b. Decode predictions & prepare FK inputs ---
        
        decode_dict = self.endecoder.decode_g1_new(pred_x)
        outputs["decode_dict"] = decode_dict

        # GT body positions and orientations (WORLD frame)
        gt_root_wxyz  = gt["g1_body_quat_w"][:, :, 0]      # (B,L,4) wxyz
        gt_root_R_w   = quaternion_to_matrix(gt_root_wxyz)  # (B,L,3,3) world
        gt_root_pos   = gt["g1_body_pos_w"][:, :, 0]        # (B,L,3)

        # GT joint rotation matrices (from [0,θ,0] encoding, same convention as encode_g1)
        gt_bp_r6d    = self.endecoder._dof_to_r6d(gt["g1_joint_pos"])         # (B,L,174)
        gt_bp_rotmat = rotation_6d_to_matrix(gt_bp_r6d.reshape(B, L, 29, 6))  # (B,L,29,3,3)

        # Recover world-frame root orientation using global_orient_gv + cam_angvel,
        # IDENTICAL to inference path (lines 167-191). Previously the training
        # path used the shortcut `R_w = R_w2c.T @ R_c`, which gives the model a
        # perfect-T_w2c crutch — `global_orient_gv_r6d` never enters the
        # supervision graph and is therefore unusable at inference time when
        # T_w2c is not available. Unifying both paths forces gv to be learned.
        # bf16-mixed: cast to fp32 first to avoid acos NaN at singular rotations.
        go_c_rotmat_f32      = decode_dict["global_orient_rotmat"].float()
        local_transl_vel_f32 = decode_dict["local_transl_vel"].float()
        go_c_aa_f32  = matrix_to_axis_angle(go_c_rotmat_f32)
        go_gv_aa_f32 = matrix_to_axis_angle(
            rotation_6d_to_matrix(decode_dict["global_orient_gv_r6d"].float())
        )
        params_w = get_smpl_params_w_Rt_v2(
            global_orient_gv = go_gv_aa_f32,
            local_transl_vel = local_transl_vel_f32,
            global_orient_c  = go_c_aa_f32,
            cam_angvel       = inputs["cam_angvel"].float(),
            length           = length,
        )
        pred_root_aa_w = params_w["global_orient"]                        # (B,L,3)   y-up
        pred_root_R_w  = axis_angle_to_matrix(pred_root_aa_w)             # (B,L,3,3) y-up

        # Predicted root position (rollout in world frame from GT start)
        pred_root_pos = rollout_local_transl_vel(
            local_transl_vel_f32, pred_root_aa_w, gt_root_pos[:, [0]].float()
        )  # (B,L,3)
        outputs["pred_root_pos"] = pred_root_pos

        # --- 4c. FK-based 3D joint loss ---
        if hasattr(self.endecoder, "_g1_humanoid"):
            return_vertices = hasattr(self.endecoder, "_g1_vertices_local")
            # GT: scalar joint_pos already BYD order; gt_root_wxyz already wxyz quat
            if return_vertices:
                gt_joints_w, _, gt_vertices_w = self.endecoder.fk_g1(
                    gt["g1_joint_pos"],
                    gt_root_pos,
                    gt_root_wxyz,
                    world_up='y',
                    return_vertices=True,
                )
            else:
                gt_joints_w, _ = self.endecoder.fk_g1(
                    gt["g1_joint_pos"], gt_root_pos, gt_root_wxyz, world_up='y'
                )

            # Pred: recover scalar θ (BYD) from [0,θ,0]-encoded rotmat; rotmat → wxyz quat.
            
            
            
            
            R_bp = decode_dict["body_pose_rotmat"].float()                # (B,L,29,3,3)
            pred_dof_byd = torch.atan2(R_bp[..., 0, 2], R_bp[..., 0, 0])  # (B,L,29)
            pred_root_quat_w = matrix_to_quaternion(pred_root_R_w)
            if return_vertices:
                pred_joints_w, _, pred_vertices_w = self.endecoder.fk_g1(
                    pred_dof_byd,
                    pred_root_pos,
                    pred_root_quat_w,
                    world_up='y',
                    return_vertices=True,
                )
            else:
                pred_joints_w, _ = self.endecoder.fk_g1(
                    pred_dof_byd, pred_root_pos, pred_root_quat_w, world_up='y'
                )

            
            outputs["gt_joints_w"]   = gt_joints_w    # (B,L,30,3)
            outputs["pred_joints_w"] = pred_joints_w
            if return_vertices:
                outputs["gt_vertices_w"] = gt_vertices_w      # (B,L,384,3)
                outputs["pred_vertices_w"] = pred_vertices_w

            # --- incam pose for j2d/transl_c (ORIGINAL VITH semantics) ---
            
            
            
            
            
            
            
            R_bp_inc        = decode_dict["body_pose_rotmat"].float()       # (B,L,29,3,3)
            incam_dof_byd   = torch.atan2(R_bp_inc[..., 0, 2], R_bp_inc[..., 0, 0])
            root_R_c_direct = decode_dict["global_orient_rotmat"].float()   
            R_w2c_g1        = inputs["g1_target"]["g1_T_w2c"][..., :3, :3].float()
            root_R_w_short  = R_w2c_g1.transpose(-1, -2) @ root_R_c_direct  # (B,L,3,3) world
            root_quat_w_sh  = matrix_to_quaternion(root_R_w_short)
            zeros_root      = torch.zeros_like(pred_root_pos)
            if return_vertices:
                incam_joints_w, _, incam_vertices_w = self.endecoder.fk_g1(
                    incam_dof_byd,
                    zeros_root,
                    root_quat_w_sh,
                    world_up='y',
                    return_vertices=True,
                )                                                           # (B,L,30,3), (B,L,384,3)
            else:
                incam_joints_w, _ = self.endecoder.fk_g1(
                    incam_dof_byd, zeros_root, root_quat_w_sh, world_up='y'
                )                                                           # (B,L,30,3)
            incam_joints_w_rr = incam_joints_w - incam_joints_w[:, :, :1]   # root-relative
            incam_pred_c_rootrel = torch.einsum(
                "blij,blnj->blni", R_w2c_g1, incam_joints_w_rr
            )                                                               
            
            outputs["incam_pred_c_rootrel"] = incam_pred_c_rootrel
            if return_vertices:
                incam_vertices_w_rr = incam_vertices_w - incam_joints_w[:, :, :1]
                outputs["incam_pred_c_vertices_rootrel"] = torch.einsum(
                    "blij,blvj->blvi", R_w2c_g1, incam_vertices_w_rr
                )                                                           

            g1_loss, g1_dict = compute_extra_g1_loss(inputs, outputs, self)
            total_loss += g1_loss
            outputs.update(g1_dict)

        # --- 4d. Incam / global losses (position-based, using FK body_pos_w) ---
        if hasattr(self.endecoder, "_g1_humanoid"):
            pred_body_pos_w = outputs["pred_joints_w"]   # (B,L,30,3)
            gt_body_pos_w   = outputs["gt_joints_w"]
        else:
            # Fallback: use GT body positions when FK is not available
            pred_body_pos_w = gt["g1_body_pos_w"]
            gt_body_pos_w   = gt["g1_body_pos_w"]

        outputs.update(self._prepare_robot_outputs_from_fk(
            inputs, pred_body_pos_w, gt_body_pos_w
        ))

        for extra_func in [compute_extra_incam_loss, compute_extra_global_loss]:
            extra_loss, extra_loss_dict = extra_func(inputs, outputs, self)
            total_loss += extra_loss
            outputs.update(extra_loss_dict)

        outputs["loss"] = total_loss
        return outputs

    def _prepare_robot_outputs_from_fk(self, inputs, pred_body_pos_w, gt_body_pos_w):
        'Documentation removed.'
        root_body_id = getattr(getattr(self.args, "robot", None), "root_body_id", 0)

        g1_T_w2c = inputs["g1_target"]["g1_T_w2c"]   # (B,L,4,4) G1 camera
        pred_body_pos_c = transform_points_w2c(pred_body_pos_w, g1_T_w2c)
        gt_body_pos_c   = transform_points_w2c(gt_body_pos_w,   g1_T_w2c)

        

        
        
        
        # Match AMASS/VITH bbox semantics and the pred_cam-stat scripts: use
        # the 30 FK body/joint positions, then let get_bbx_xys apply its default
        # 1.2 enlargement.  Sampled vertices remain available for vertex losses
        # but must not silently change the camera-target definition.
        robot_bbx_points_c = gt_body_pos_c
        robot_bbx_xys = compute_robot_bbx_xys(
            robot_bbx_points_c, inputs["K_fullimg"], do_augment=False,
        )

        return {
            "pred_body_pos_c": pred_body_pos_c,
            "gt_body_pos_c":   gt_body_pos_c,
            "pred_root_pos_w": pred_body_pos_w[:, :, root_body_id],
            "gt_root_pos_w":   gt_body_pos_w[:,  :, root_body_id],
            "pred_root_pos_c": pred_body_pos_c[:, :, root_body_id],
            "gt_root_pos_c":   gt_body_pos_c[:,  :, root_body_id],
            "robot_bbx_xys":   robot_bbx_xys,
        }

    # kept for backward compat
    def _prepare_robot_outputs(self, inputs, pred_g1):
        gt = inputs["g1_target"]
        return self._prepare_robot_outputs_from_fk(
            inputs, pred_g1["body_pos_w"], gt["g1_body_pos_w"], None
        )

    def _randomly_set_null_condition(self, f_condition, uncond_prob=0.1, skip_mask=None):
        'Documentation removed.'
        keys = list(f_condition.keys())
        for k in keys:
            if f_condition[k] is None:
                continue
            if k.endswith("_mask"):
                continue
            f_condition[k] = f_condition[k].clone()
            
            mask = torch.rand(f_condition[k].shape[:2], device=f_condition[k].device) < uncond_prob
            if skip_mask is not None:
                mask[skip_mask] = False  
            f_condition[k][mask] = 0.0
            if k == "f_imgseq" and f_condition.get("f_imgseq_mask", None) is not None:
                img_mask = f_condition["f_imgseq_mask"]
                if img_mask.ndim == 1:
                    img_mask = img_mask[:, None].expand(mask.shape[0], mask.shape[1]).clone()
                else:
                    img_mask = img_mask.clone()
                img_mask[mask] = False
                f_condition["f_imgseq_mask"] = img_mask
        return f_condition

    def _ddim_sample(self, length, f_condition, real=None, real_motion=None):
        'Documentation removed.'
        from hmr4d.model.offline.utils.endecoder import G1_PRED_X_DIM

        obs = f_condition["obs"]
        B, L = obs.shape[0], obs.shape[1]
        shape = (B, L, G1_PRED_X_DIM)

        
        f_cond = self.cond_embedder(**f_condition, drop_obs=None)
        scale = float(getattr(self, "cfg_scale", 1.0))
        f_uncond = None
        if scale != 1.0:
            drop_all = torch.ones(B, dtype=torch.bool, device=obs.device)  # ∅ = drop obs
            f_uncond = self.cond_embedder(**f_condition, drop_obs=drop_all)

        def _cfg(a_c, a_u):
            
            if a_c is None or a_u is None:
                return a_c
            return a_u + scale * (a_c - a_u)

        def model_fn(x, ts, **kwargs):
            out_c = self.denoiser3d(f_cond=f_cond, length=length, xt=x, timesteps=ts, real=real, gt_motion=real_motion)
            if f_uncond is None:
                return out_c
            out_u = self.denoiser3d(f_cond=f_uncond, length=length, xt=x, timesteps=ts, real=real, gt_motion=real_motion)
            out_c = dict(out_c)
            
            guided = _cfg(out_c["pred_x_start"], out_u["pred_x_start"])
            out_c["pred_x_start"] = guided
            out_c["pred_x"] = guided
            out_c["pred_cam"] = _cfg(out_c.get("pred_cam"), out_u.get("pred_cam"))
            out_c["static_conf_logits"] = _cfg(
                out_c.get("static_conf_logits"), out_u.get("static_conf_logits")
            )
            return out_c

        # Experiment-only one-step inference at the largest original timestep.
        # ``gaussian`` preserves diffusion stochasticity (a fresh x_T each call),
        # while ``zero`` is available only as an explicit deterministic ablation.
        # Neither mode enters the iterative DDIM denoising loop.
        single_step_init = getattr(self.diffusion_cfg, "single_step_max_t_init", None)
        if single_step_init is not None:
            single_step_init = str(single_step_init).lower()
            if single_step_init not in {"gaussian", "zero"}:
                raise ValueError(
                    "diffusion.single_step_max_t_init must be 'gaussian', 'zero', or null."
                )
            max_t = int(getattr(self.test_diffusion, "original_num_steps", 1000)) - 1
            if single_step_init == "gaussian":
                # Match the original stochastic DDIM path's RNG handling: draw a
                # seed from CPU RNG, then use a local device generator so repeated
                # calls receive different x_T even if Lightning resets CUDA RNG.
                cpu_noise_seed = torch.randint(0, 2**31, (1,)).item()
                generator = torch.Generator(device=obs.device).manual_seed(cpu_noise_seed)
                x_t = torch.randn(
                    *shape,
                    device=obs.device,
                    dtype=f_cond.dtype,
                    generator=generator,
                )
            else:
                x_t = torch.zeros(shape, device=obs.device, dtype=f_cond.dtype)
            timesteps = torch.full((B,), max_t, device=obs.device, dtype=torch.long)
            out = model_fn(x_t, timesteps)
            pred_x_start = out["pred_x_start"] if "pred_x_start" in out else out["pred_x"]
            return {
                "pred_x": pred_x_start,
                "pred_x_start": pred_x_start,
                "pred_cam": out.get("pred_cam"),
                "static_conf_logits": out.get("static_conf_logits"),
                "pred_context": out.get("pred_context"),
            }

        
        
        
        
        _cpu_noise_seed = torch.randint(0, 2**31, (1,)).item()  # advance CPU RNG
        _gen = torch.Generator(device=obs.device).manual_seed(_cpu_noise_seed)
        init_noise = torch.randn(*shape, device=obs.device, generator=_gen)

        out = self.test_diffusion.ddim_sample_loop_with_aux(
            model_fn,
            shape,
            noise=init_noise,            
            clip_denoised=False,         
            eta=self.ddim_eta,
            model_kwargs={"y": {}},      
            device=obs.device,
            progress=False,
        )
        return {
            "pred_x": out["pred_xstart"],
            "pred_x_start": out["pred_xstart"],
            "pred_cam": out.get("pred_cam"),
            "static_conf_logits": out.get("static_conf_logits"),
            "pred_context": out.get("pred_context"),
        }


def randomly_set_null_condition(f_condition, uncond_prob=0.1):
    """Conditions are in shape (B, L, *)"""
    keys = list(f_condition.keys())
    for k in keys:
        if f_condition[k] is None:
            continue
        f_condition[k] = f_condition[k].clone()
        mask = torch.rand(f_condition[k].shape[:2], device=f_condition[k].device) < uncond_prob
        f_condition[k][mask] = 0.0
    return f_condition


def _reduce_loss(loss_el, mask_bcast, t_weights=None, fk_gate=None):
    'Documentation removed.'
    nan_count = (~torch.isfinite(loss_el)).sum().float()

    if fk_gate is None and t_weights is None:
        return (loss_el * mask_bcast).mean(), nan_count

    B = loss_el.shape[0]
    loss_el2 = loss_el
    if fk_gate is not None:
        keep = fk_gate.view(B, *([1] * (loss_el.dim() - 1)))
        
        loss_el2 = torch.where(keep.expand_as(loss_el), loss_el, torch.zeros_like(loss_el))

    w = mask_bcast.expand_as(loss_el2).float()
    per_sample = (loss_el2 * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp(min=1)  # (B,)

    sample_w = t_weights if t_weights is not None else loss_el.new_ones(B)
    if fk_gate is not None:
        sample_w = sample_w * fk_gate.float()  
    return (per_sample * sample_w).mean(), nan_count


def _geometry_loss_elementwise(pred, target, ppl):
    """Elementwise loss for joint/vertex geometry and camera translation.

    Existing experiments keep MSE by default. New experiments can opt into
    Huber loss with ``pipeline.args.geometry_loss_type: huber``.
    """
    if ppl.geometry_loss_type == "huber":
        return F.huber_loss(
            pred,
            target,
            reduction="none",
            delta=ppl.geometry_huber_delta,
        )
    return F.mse_loss(pred, target, reduction="none")


def _get_per_sample_fps(inputs, reference, default_fps=30.0):
    """Return a (B,1,1,1) fps tensor for finite-difference geometry losses."""
    B = reference.shape[0]
    fps = inputs.get("fps", None)
    if fps is None:
        fps = reference.new_full((B,), float(default_fps))
    elif isinstance(fps, torch.Tensor):
        fps = fps.to(device=reference.device, dtype=reference.dtype).reshape(-1)
        if fps.numel() == 1:
            fps = fps.expand(B)
        elif fps.numel() != B:
            raise ValueError(f"fps must have 1 or B={B} values, got shape={tuple(inputs['fps'].shape)}")
    else:
        fps = reference.new_full((B,), float(fps))
    if not torch.isfinite(fps).all() or (fps <= 0).any():
        raise ValueError(f"Invalid per-sample fps values: {fps.detach().cpu().tolist()}")
    return fps.view(B, 1, 1, 1)


def compute_extra_incam_loss(inputs, outputs, ppl):
    weights = ppl.weights

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"]
    t_weights = outputs.get("t_weights")
    fk_gate = outputs.get("fk_gate")
    model_output = outputs["model_output"]
    pred_body_pos_c = outputs["pred_body_pos_c"]
    gt_body_pos_c = outputs["gt_body_pos_c"]
    pred_root_pos_c = outputs["pred_root_pos_c"]
    gt_root_pos_c = outputs["gt_root_pos_c"]
    robot_bbx_xys = outputs["robot_bbx_xys"]
    mask_reproj = ~inputs["mask"].get("spv_incam_only", torch.zeros_like(mask, dtype=torch.bool))

    
    
    
    
    incam_rootrel = outputs.get("incam_pred_c_rootrel")
    if incam_rootrel is not None:
        pred_cr_j3d = incam_rootrel - incam_rootrel[:, :, :1]
    else:
        pred_cr_j3d = pred_body_pos_c - pred_body_pos_c[:, :, :1]
    gt_cr_j3d   = gt_body_pos_c   - gt_body_pos_c[:, :, :1]

    if weights.get("cr_j3d", 0.0) > 0:
        cr_j3d_loss_el = _geometry_loss_elementwise(pred_cr_j3d, gt_cr_j3d, ppl)
        cr_j3d_loss, cr_j3d_loss_nan_count = _reduce_loss(
            cr_j3d_loss_el, mask[..., None, None], t_weights, fk_gate
        )
        extra_loss += cr_j3d_loss * weights["cr_j3d"]
        extra_loss_dict["cr_j3d_loss"] = cr_j3d_loss
        extra_loss_dict["cr_j3d_loss_nan_count"] = cr_j3d_loss_nan_count

    # End-effector dynamics start from the exact camera-frame, root-aligned
    # positions used by cr_j3d above, then rotate each frame into its own pelvis
    # frame before taking finite differences. With the predicted branch using
    # the same predicted root rotation that produced pred_cr_j3d, R_pred^T R_pred
    # cancels analytically, so this term mainly supervises articulated body pose
    # rather than absolute pelvis translation/orientation or camera rotation.
    ee_vel_weight = float(weights.get("ee_vel_huber", 0.0))
    ee_acc_weight = float(weights.get("ee_acc_huber", 0.0))
    if ee_vel_weight > 0.0 or ee_acc_weight > 0.0:
        ee_cfg = getattr(ppl.args, "end_effector_loss", None)
        body_ids = list(getattr(ee_cfg, "body_ids", [14, 15, 18, 19, 28, 29]))
        if not body_ids:
            raise ValueError("end_effector_loss.body_ids must not be empty.")
        if min(body_ids) < 0 or max(body_ids) >= pred_cr_j3d.shape[2]:
            raise ValueError(
                f"end_effector_loss.body_ids={body_ids} are invalid for "
                f"{pred_cr_j3d.shape[2]} cr_j3d bodies."
            )
        ee_delta = float(getattr(ee_cfg, "huber_delta", ppl.geometry_huber_delta))
        if ee_delta <= 0:
            raise ValueError("end_effector_loss.huber_delta must be positive.")
        default_ee_fps = float(getattr(ee_cfg, "fps", getattr(ppl.args, "fps", 30.0)))
        if default_ee_fps <= 0:
            raise ValueError("end_effector_loss.fps must be positive.")

        pred_root_R_c = outputs["decode_dict"]["global_orient_rotmat"].float()
        gt_root_R_w = quaternion_to_matrix(
            inputs["g1_target"]["g1_body_quat_w"][:, :, 0].float()
        )
        R_w2c = inputs["g1_target"]["g1_T_w2c"][..., :3, :3].float()
        gt_root_R_c = R_w2c @ gt_root_R_w

        pred_local = torch.einsum(
            "blij,blnj->blni", pred_root_R_c.transpose(-1, -2), pred_cr_j3d
        )
        gt_local = torch.einsum(
            "blij,blnj->blni", gt_root_R_c.transpose(-1, -2), gt_cr_j3d
        )

        pred_ee = pred_local[:, :, body_ids]
        gt_ee = gt_local[:, :, body_ids]
        ee_fps = _get_per_sample_fps(inputs, pred_ee, default_fps=default_ee_fps)

        if ee_vel_weight > 0.0 and pred_ee.shape[1] >= 2:
            pred_vel = (pred_ee[:, 1:] - pred_ee[:, :-1]) * ee_fps
            gt_vel = (gt_ee[:, 1:] - gt_ee[:, :-1]) * ee_fps
            vel_mask = mask[:, 1:] & mask[:, :-1]
            ee_vel_loss_el = F.huber_loss(
                pred_vel, gt_vel, reduction="none", delta=ee_delta
            )
            ee_vel_loss, ee_vel_loss_nan_count = _reduce_loss(
                ee_vel_loss_el, vel_mask[..., None, None], t_weights, fk_gate
            )
            extra_loss += ee_vel_loss * ee_vel_weight
            extra_loss_dict["ee_vel_huber_loss"] = ee_vel_loss
            extra_loss_dict["ee_vel_huber_loss_nan_count"] = ee_vel_loss_nan_count

        if ee_acc_weight > 0.0 and pred_ee.shape[1] >= 3:
            pred_vel = (pred_ee[:, 1:] - pred_ee[:, :-1]) * ee_fps
            gt_vel = (gt_ee[:, 1:] - gt_ee[:, :-1]) * ee_fps
            pred_acc = (pred_vel[:, 1:] - pred_vel[:, :-1]) * ee_fps
            gt_acc = (gt_vel[:, 1:] - gt_vel[:, :-1]) * ee_fps
            acc_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
            ee_acc_loss_el = F.huber_loss(
                pred_acc, gt_acc, reduction="none", delta=ee_delta
            )
            ee_acc_loss, ee_acc_loss_nan_count = _reduce_loss(
                ee_acc_loss_el, acc_mask[..., None, None], t_weights, fk_gate
            )
            extra_loss += ee_acc_loss * ee_acc_weight
            extra_loss_dict["ee_acc_huber_loss"] = ee_acc_loss
            extra_loss_dict["ee_acc_huber_loss_nan_count"] = ee_acc_loss_nan_count

    if weights.get("transl_c", 0.0) > 0 and model_output.get("pred_cam") is not None:
        pred_cam = model_output["pred_cam"]
        gt_pred_cam = get_a_pred_cam(gt_root_pos_c, robot_bbx_xys, inputs["K_fullimg"])
        gt_pred_cam[gt_pred_cam.isinf()] = -1

        gt_body_z_min = gt_body_pos_c[..., 2].min(dim=-1)[0]
        valid_mask = (
            (gt_body_z_min > 0.3)
            * (gt_pred_cam[..., 0] > 0.3)
            * (gt_pred_cam[..., 0] < 5.0)
            * (gt_pred_cam[..., 1] > -3.0)
            * (gt_pred_cam[..., 1] < 3.0)
            * (gt_pred_cam[..., 2] > -3.0)
            * (gt_pred_cam[..., 2] < 3.0)
            * (robot_bbx_xys[..., 2] > 0)
        )[..., None]
        transl_c_loss_el = _geometry_loss_elementwise(pred_cam, gt_pred_cam, ppl)
        transl_c_loss, transl_c_loss_nan_count = _reduce_loss(
            transl_c_loss_el, mask[..., None] * valid_mask, t_weights, fk_gate
        )
        extra_loss += transl_c_loss * weights["transl_c"]
        extra_loss_dict["transl_c_loss"] = transl_c_loss
        extra_loss_dict["transl_c_loss_nan_count"] = transl_c_loss_nan_count

    if weights.get("j2d", 0.0) > 0:
        reproj_z_thr = 0.3

        
        
        
        
        
        incam_rootrel = outputs.get("incam_pred_c_rootrel")
        pred_cam = model_output.get("pred_cam")
        if incam_rootrel is not None and pred_cam is not None:
            transl_c_cam = compute_transl_full_cam(pred_cam, robot_bbx_xys, inputs["K_fullimg"])
            pred_body_pos_c_j2d = incam_rootrel + transl_c_cam[:, :, None]   # (B,L,30,3)
        else:
            
            pred_body_pos_c_j2d = pred_body_pos_c

        
        
        pred_body_pos_c_j2d = pred_body_pos_c_j2d.clone()
        gt_body_pos_c_j2d = gt_body_pos_c.clone()
        pred_body_pos_c_j2d[pred_body_pos_c_j2d[..., 2].abs() <= reproj_z_thr] = reproj_z_thr
        gt_body_pos_c_j2d[gt_body_pos_c_j2d[..., 2].abs() <= reproj_z_thr] = reproj_z_thr

        pred_j2d_01 = project_to_bi01(pred_body_pos_c_j2d, robot_bbx_xys, inputs["K_fullimg"])
        gt_j2d_01 = project_to_bi01(gt_body_pos_c_j2d, robot_bbx_xys, inputs["K_fullimg"])
        
        
        pred_j2d_01 = pred_j2d_01.clamp(-2.0, 3.0)
        gt_j2d_01   = gt_j2d_01.clamp(-2.0, 3.0)

        valid_mask = (
            (gt_body_pos_c_j2d[..., 2] > reproj_z_thr)
            * (pred_body_pos_c_j2d[..., 2] > reproj_z_thr)  # Be safe
            * (gt_j2d_01[..., 0] > 0.0)
            * (gt_j2d_01[..., 0] < 1.0)
            * (gt_j2d_01[..., 1] > 0.0)
            * (gt_j2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False
        j2d_loss_el = _geometry_loss_elementwise(pred_j2d_01, gt_j2d_01, ppl)
        j2d_loss, j2d_loss_nan_count = _reduce_loss(
            j2d_loss_el, mask[..., None, None] * valid_mask, t_weights, fk_gate
        )
        extra_loss += j2d_loss * weights["j2d"]
        extra_loss_dict["j2d_loss"] = j2d_loss
        extra_loss_dict["j2d_loss_nan_count"] = j2d_loss_nan_count

    gt_vertices_w = outputs.get("gt_vertices_w")
    incam_vertices_rootrel = outputs.get("incam_pred_c_vertices_rootrel")
    if (
        gt_vertices_w is not None
        and incam_vertices_rootrel is not None
        and (weights.get("cr_verts", 0.0) > 0 or weights.get("verts2d", 0.0) > 0)
    ):
        gt_c_verts = transform_points_w2c(gt_vertices_w, inputs["g1_target"]["g1_T_w2c"])

        if weights.get("cr_verts", 0.0) > 0:
            pred_cr_verts = incam_vertices_rootrel
            gt_cr_verts = gt_c_verts - gt_root_pos_c[:, :, None]
            cr_vert_loss_el = _geometry_loss_elementwise(pred_cr_verts, gt_cr_verts, ppl)
            cr_vert_loss, cr_vert_loss_nan_count = _reduce_loss(
                cr_vert_loss_el, mask[..., None, None], t_weights, fk_gate
            )
            extra_loss += cr_vert_loss * weights["cr_verts"]
            extra_loss_dict["cr_vert_loss"] = cr_vert_loss
            extra_loss_dict["cr_vert_loss_nan_count"] = cr_vert_loss_nan_count

        if weights.get("verts2d", 0.0) > 0:
            pred_cam = model_output.get("pred_cam")
            if pred_cam is not None:
                reproj_z_thr = 0.3
                transl_c_cam = compute_transl_full_cam(pred_cam, robot_bbx_xys, inputs["K_fullimg"])
                pred_c_verts_2d = incam_vertices_rootrel + transl_c_cam[:, :, None]
                gt_c_verts_2d = gt_c_verts.clone()
                pred_c_verts_2d = pred_c_verts_2d.clone()
                pred_c_verts_2d[pred_c_verts_2d[..., 2].abs() <= reproj_z_thr] = reproj_z_thr
                gt_c_verts_2d[gt_c_verts_2d[..., 2].abs() <= reproj_z_thr] = reproj_z_thr

                pred_verts2d_01 = project_to_bi01(pred_c_verts_2d, robot_bbx_xys, inputs["K_fullimg"])
                gt_verts2d_01 = project_to_bi01(gt_c_verts_2d, robot_bbx_xys, inputs["K_fullimg"])
                valid_mask = (
                    (gt_c_verts_2d[..., 2] > reproj_z_thr)
                    * (pred_c_verts_2d[..., 2] > reproj_z_thr)
                    * (gt_verts2d_01[..., 0] > 0.0)
                    * (gt_verts2d_01[..., 0] < 1.0)
                    * (gt_verts2d_01[..., 1] > 0.0)
                    * (gt_verts2d_01[..., 1] < 1.0)
                )[..., None]
                valid_mask[~mask_reproj] = False
                verts2d_loss_el = _geometry_loss_elementwise(pred_verts2d_01, gt_verts2d_01, ppl)
                verts2d_loss, verts2d_loss_nan_count = _reduce_loss(
                    verts2d_loss_el, mask[..., None, None] * valid_mask, t_weights, fk_gate
                )
                extra_loss += verts2d_loss * weights["verts2d"]
                extra_loss_dict["verts2d_loss"] = verts2d_loss
                extra_loss_dict["verts2d_loss_nan_count"] = verts2d_loss_nan_count

    return extra_loss, extra_loss_dict


def compute_extra_global_loss(inputs, outputs, ppl):
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"]
    t_weights = outputs.get("t_weights")
    fk_gate = outputs.get("fk_gate")
    gt = inputs["g1_target"]
    gt_root_pos_w = outputs["gt_root_pos_w"]

    if weights.get("transl_w", 0.0) > 0:
        
        
        
        
        gt_root_wxyz = gt["g1_body_quat_w"][:, :, 0]                              # (B,L,4) wxyz
        gt_root_aa_w = matrix_to_axis_angle(quaternion_to_matrix(gt_root_wxyz).float())  # (B,L,3) world
        local_transl_vel = outputs["decode_dict"]["local_transl_vel"].float()
        pred_transl_w = rollout_local_transl_vel(
            local_transl_vel, gt_root_aa_w, gt_root_pos_w[:, [0]].float()
        )
        transl_w_loss_el = F.l1_loss(pred_transl_w, gt_root_pos_w, reduction="none")
        transl_w_loss, transl_w_loss_nan_count = _reduce_loss(
            transl_w_loss_el, mask[..., None], t_weights, fk_gate
        )
        extra_loss += transl_w_loss * weights["transl_w"]
        extra_loss_dict["transl_w_loss"] = transl_w_loss
        extra_loss_dict["transl_w_loss_nan_count"] = transl_w_loss_nan_count

    if weights.get("static_conf_bce", 0.0) > 0 and outputs["model_output"].get("static_conf_logits") is not None:
        # `vel_thr` is the original-VITH HuMoR threshold (0.15 m/s). Per-body
        # threshold = vel_thr × vel_thr_mul[i]; default mul = 1.0 for every body.
        # Both `g1_body_lin_vel_w` and the threshold are in m/s, so the rule is
        # invariant to motion_frames / speed-augmentation playback fps.
        vel_thr = args.static_conf.vel_thr
        body_ids = getattr(args.static_conf, "body_ids", None)
        if body_ids is None:
            body_ids = list(range(gt["g1_body_pos_w"].shape[2]))
        pred_static_conf_logits = outputs["model_output"]["static_conf_logits"]
        if pred_static_conf_logits.shape[-1] != len(body_ids):
            pred_static_conf_logits = pred_static_conf_logits[:, :, body_ids]

        # per-body threshold tensor, shape (J,)
        vel_thr_mul = getattr(args.static_conf, "vel_thr_mul", None)
        device = pred_static_conf_logits.device
        if vel_thr_mul is None:
            vel_thr_per_body = torch.full(
                (len(body_ids),), float(vel_thr), device=device, dtype=torch.float32,
            )
        else:
            assert len(vel_thr_mul) == len(body_ids), (
                f"vel_thr_mul ({len(vel_thr_mul)}) must match body_ids ({len(body_ids)})"
            )
            vel_thr_per_body = (
                torch.tensor(list(vel_thr_mul), device=device, dtype=torch.float32) * float(vel_thr)
            )

        if "g1_body_lin_vel_w" in gt:
            
            
            speed = gt["g1_body_lin_vel_w"][:, :, body_ids].norm(dim=-1)            # (B, L, J)
            static_gt = speed < vel_thr_per_body                                     # broadcast (J,)
        else:
            
            
            fps = float(getattr(args, "fps", 30.0))
            gt_disp = gt["g1_body_pos_w"][:, 1:, body_ids] - gt["g1_body_pos_w"][:, :-1, body_ids]
            speed_prev = gt_disp.norm(dim=-1) * fps                                  # (B, L-1, J) m/s
            static_prev = speed_prev < vel_thr_per_body
            
            static_gt = torch.cat([static_prev, static_prev[:, -1:]], dim=1)
        static_gt = static_gt.float()

        
        
        vel_mask = mask.clone()
        lengths = inputs["length"].long().clamp(min=1, max=mask.shape[1])
        batch_ids = torch.arange(mask.shape[0], device=mask.device)
        vel_mask[batch_ids, lengths - 1] = False
        static_conf_loss_el = F.binary_cross_entropy_with_logits(pred_static_conf_logits, static_gt, reduction="none")
        
        static_conf_loss, static_conf_loss_nan_count = _reduce_loss(
            static_conf_loss_el, vel_mask[..., None], t_weights, fk_gate=None
        )
        extra_loss += static_conf_loss * weights["static_conf_bce"]
        extra_loss_dict["static_conf_loss"] = static_conf_loss
        extra_loss_dict["static_conf_loss_nan_count"] = static_conf_loss_nan_count

    return extra_loss, extra_loss_dict


def compute_extra_g1_loss(inputs, outputs, ppl):
    """FK-based 3D joint/vertex position losses for G1 robot.

    Requires:
        outputs["pred_joints_w"]: (B, L, 30, 3) predicted world-frame joint positions
        outputs["gt_joints_w"]:   (B, L, 30, 3) GT world-frame joint positions
        outputs["pred_vertices_w"]: (B, L, 384, 3), optional world-frame vertices
        outputs["gt_vertices_w"]:   (B, L, 384, 3), optional world-frame vertices

    Coordinate contract:
        Both branches come from ``fk_g1(..., world_up='y')`` in ``forward``.
        The dataset targets are also canonical Y-up world coordinates. Therefore
        the global losses below compare absolute world positions directly; no
        additional Y-up/Z-up conversion or root alignment is applied here.
    """
    weights = ppl.weights
    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"]   # (B, L)
    t_weights = outputs.get("t_weights")
    fk_gate = outputs.get("fk_gate")

    pred_j = outputs["pred_joints_w"]   # (B, L, 30, 3)
    gt_j   = outputs["gt_joints_w"]

    # Absolute global/world-frame joint loss. Unlike cr_j3d/fk_j3d this keeps
    # root translation and global orientation in the supervision signal.
    if weights.get("global_j3d", 0.0) > 0:
        global_j3d_loss_el = _geometry_loss_elementwise(pred_j, gt_j, ppl)
        global_j3d_loss, global_j3d_loss_nan_count = _reduce_loss(
            global_j3d_loss_el, mask[..., None, None], t_weights, fk_gate
        )
        extra_loss += global_j3d_loss * weights["global_j3d"]
        extra_loss_dict["global_j3d_loss"] = global_j3d_loss
        extra_loss_dict["global_j3d_loss_nan_count"] = global_j3d_loss_nan_count

    # Absolute global/world-frame sampled-vertex loss. The sampled vertex local
    # coordinates are attached by FK after its explicit Z-up -> Y-up conversion,
    # so pred and GT are already in the same Y-up world frame here.
    global_vertices3d_weight = float(weights.get("global_vertices3d", 0.0))
    if global_vertices3d_weight > 0:
        pred_vertices_w = outputs.get("pred_vertices_w")
        gt_vertices_w = outputs.get("gt_vertices_w")
        if pred_vertices_w is None or gt_vertices_w is None:
            raise RuntimeError(
                "global_vertices3d requires sampled G1 vertices; set "
                "pipeline.args.robot.sampled_vertices_path."
            )
        global_vertices3d_loss_el = _geometry_loss_elementwise(
            pred_vertices_w, gt_vertices_w, ppl
        )
        global_vertices3d_loss, global_vertices3d_loss_nan_count = _reduce_loss(
            global_vertices3d_loss_el, mask[..., None, None], t_weights, fk_gate
        )
        extra_loss += global_vertices3d_loss * global_vertices3d_weight
        extra_loss_dict["global_vertices3d_loss"] = global_vertices3d_loss
        extra_loss_dict["global_vertices3d_loss_nan_count"] = (
            global_vertices3d_loss_nan_count
        )

    # Root-aligned FK joint loss.
    if weights.get("fk_j3d", 0.0) > 0:
        pred_cr = pred_j - pred_j[:, :, :1]
        gt_cr   = gt_j   - gt_j[:,   :, :1]
        fk_loss_el = F.mse_loss(pred_cr, gt_cr, reduction="none")
        fk_loss, fk_j3d_loss_nan_count = _reduce_loss(fk_loss_el, mask[..., None, None], t_weights, fk_gate)
        extra_loss += fk_loss * weights["fk_j3d"]
        extra_loss_dict["fk_j3d_loss"] = fk_loss
        extra_loss_dict["fk_j3d_loss_nan_count"] = fk_j3d_loss_nan_count

    # Absolute world-frame FK joint loss (root translation also supervised)
    if weights.get("fk_j3d_abs", 0.0) > 0:
        fk_abs_loss_el = F.mse_loss(pred_j, gt_j, reduction="none")
        fk_abs_loss, fk_j3d_abs_loss_nan_count = _reduce_loss(fk_abs_loss_el, mask[..., None, None], t_weights, fk_gate)
        extra_loss += fk_abs_loss * weights["fk_j3d_abs"]
        extra_loss_dict["fk_j3d_abs_loss"] = fk_abs_loss
        extra_loss_dict["fk_j3d_abs_loss_nan_count"] = fk_j3d_abs_loss_nan_count

    # Finite-difference motion losses. Velocity has T-1 frames; acceleration has T-2.
    if weights.get("fk_vel_mse", 0.0) > 0 and pred_j.shape[1] >= 2:
        pred_local = pred_j - pred_j[:, :, :1]
        gt_local = gt_j - gt_j[:, :, :1]
        pred_vel = pred_local[:, 1:] - pred_local[:, :-1]
        gt_vel = gt_local[:, 1:] - gt_local[:, :-1]
        vel_mask = mask[:, 1:] & mask[:, :-1]
        fk_vel_loss_el = F.mse_loss(pred_vel, gt_vel, reduction="none")
        fk_vel_loss, fk_vel_loss_nan_count = _reduce_loss(
            fk_vel_loss_el, vel_mask[..., None, None], t_weights, fk_gate
        )
        extra_loss += fk_vel_loss * weights["fk_vel_mse"]
        extra_loss_dict["fk_vel_mse_loss"] = fk_vel_loss
        extra_loss_dict["fk_vel_mse_loss_nan_count"] = fk_vel_loss_nan_count

    if weights.get("fk_acc_mse", 0.0) > 0 and pred_j.shape[1] >= 3:
        pred_local = pred_j - pred_j[:, :, :1]
        gt_local = gt_j - gt_j[:, :, :1]
        pred_vel = pred_local[:, 1:] - pred_local[:, :-1]
        gt_vel = gt_local[:, 1:] - gt_local[:, :-1]
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        acc_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        fk_acc_loss_el = F.mse_loss(pred_acc, gt_acc, reduction="none")
        fk_acc_loss, fk_acc_loss_nan_count = _reduce_loss(
            fk_acc_loss_el, acc_mask[..., None, None], t_weights, fk_gate
        )
        extra_loss += fk_acc_loss * weights["fk_acc_mse"]
        extra_loss_dict["fk_acc_mse_loss"] = fk_acc_loss
        extra_loss_dict["fk_acc_mse_loss_nan_count"] = fk_acc_loss_nan_count

    return extra_loss, extra_loss_dict


def transform_points_w2c(points_w, T_w2c):
    """Transform points from world coordinates to camera coordinates.

    Args:
        points_w: (B, L, N, 3)
        T_w2c: (B, L, 4, 4)
    Returns:
        (B, L, N, 3)
    """
    R_w2c = T_w2c[..., :3, :3]
    t_w2c = T_w2c[..., :3, 3]
    return torch.einsum("blij,blnj->blni", R_w2c, points_w) + t_w2c[:, :, None, :]


def compute_robot_bbx_xys(
    body_pos_c,
    K_fullimg,
    reproj_z_thr=0.3,
    do_augment=False,
    base_enlarge=1.2,
):
    'Documentation removed.'
    body_pos_c_safe = body_pos_c.clone()
    body_pos_c_safe[..., 2].clamp_(min=reproj_z_thr)
    robot_i2d = perspective_projection(body_pos_c_safe, K_fullimg)
    return get_bbx_xys(
        robot_i2d,
        do_augment=do_augment,
        base_enlarge=base_enlarge,
    )


@autocast("cuda", enabled=False)
def get_smpl_params_w_Rt_v2(
    global_orient_gv,
    local_transl_vel,
    global_orient_c,
    cam_angvel,
    length=None,
):
    """Get global R,t in GV0(ay)
    Args:
        cam_angvel: (B, L, 6), defined as R @ R_{w2c}^{t} = R_{w2c}^{t+1}
        length: (B,), valid sequence lengths before repeat-padding.
    """

    # Get R_ct_to_c0 from cam_angvel
    def as_identity(R):
        
        
        
        
        
        eye = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
        is_I = (R - eye).abs().amax(dim=(-1, -2)) < 1e-5              # (..., )
        return torch.where(is_I[..., None, None], eye, R)

    B = cam_angvel.shape[0]
    R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)
    R_t_to_tp1 = as_identity(R_t_to_tp1)

    # Get R_c2gv
    R_gv = axis_angle_to_matrix(global_orient_gv)  # (B, L, 3, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (B, L, 3, 3)

    # Camera view direction in GV coordinate: Rc2gv @ [0,0,1]
    R_c2gv = R_gv @ R_c.mT
    view_axis_gv = R_c2gv[:, :, :, 2]  # (B, L, 3)  Rc2gv is estimated, so the x-axis is not accurate, i.e. != 0

    # Rotate axis use camera relative rotation
    R_cnext2gv = R_c2gv @ R_t_to_tp1.mT
    view_axis_gv_next = R_cnext2gv[..., 2]

    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = vec1_xyz / vec1_xyz.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = vec2_xyz / vec2_xyz.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    
    
    
    aa_tp1_to_t = vec2_xyz.cross(vec1_xyz, dim=-1)                                # (B,L,3)
    _cross_norm = aa_tp1_to_t.norm(dim=-1, keepdim=True)                           # (B,L,1) ≥0
    _dot        = (vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True)                  # (B,L,1)
    aa_tp1_to_t_angle = torch.atan2(_cross_norm, _dot)                             # (B,L,1)
    aa_tp1_to_t = aa_tp1_to_t / _cross_norm.clamp(min=1e-6) * aa_tp1_to_t_angle

    if length is None:
        aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)
    else:
        # cam_angvel[t] describes t -> t+1, so a sequence with N valid frames
        # has only N-1 real transitions.  Smooth each valid prefix separately;
        # otherwise repeat-padding leaks the last transition into roughly the
        # final 12 valid frames (sigma=3, radius=12).
        aa_smoothed = torch.empty_like(aa_tp1_to_t)
        max_transitions = aa_tp1_to_t.shape[1]
        for b in range(B):
            n_frames = int(length[b].item())
            n_transitions = max(1, min(n_frames - 1, max_transitions))
            valid_aa = gaussian_smooth(aa_tp1_to_t[b : b + 1, :n_transitions], dim=-2)
            aa_smoothed[b, :n_transitions] = valid_aa[0]
            aa_smoothed[b, n_transitions:] = valid_aa[0, -1]
        aa_tp1_to_t = aa_smoothed
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT  # (B, L, 3)

    # Get R_t_to_0
    R_t_to_0 = [torch.eye(3)[None].expand(B, -1, -1).to(R_t_to_tp1)]
    for i in range(1, R_t_to_tp1.shape[1]):
        # Frame i is reached through transition (i-1) -> i.  The old code used
        # index i, which is one frame ahead and is hidden by constant-speed cams.
        step_R = R_tp1_to_t[:, i - 1]
        if length is not None:
            active = (i < length).view(B, 1, 1)
            eye = torch.eye(3, device=step_R.device, dtype=step_R.dtype).expand_as(step_R)
            step_R = torch.where(active, step_R, eye)
        R_t_to_0.append(R_t_to_0[-1] @ step_R)
    R_t_to_0 = torch.stack(R_t_to_0, dim=1)  # (B, L, 3, 3)
    R_t_to_0 = as_identity(R_t_to_0)

    global_orient = matrix_to_axis_angle(R_t_to_0 @ R_gv)

    # Rollout to global transl
    # Start from transl0, in gv0 -> flip y-axis of gv0
    transl = rollout_local_transl_vel(local_transl_vel, global_orient)
    global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="any->ay")

    smpl_params_w_Rt = {"global_orient": global_orient, "transl": transl}
    return smpl_params_w_Rt
