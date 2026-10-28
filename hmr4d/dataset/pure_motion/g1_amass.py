# ============================================================



#          trans/root_orient/pose_body/pose_hand/pose_jaw/pose_eye/poses/betas...

#          joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w (+fps)

# ============================================================

import gc
import json
import os

import joblib
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds

from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)

from .base_dataset import BaseDataset
from .cam_traj_utils import CameraAugmentorV11, StaticCameraV11
from .utils import interpolate_smpl_params
from hmr4d.utils.geo.hmr_global import get_tgtcoord_rootparam, get_R_c2gv
from hmr4d.utils.geo.hmr_cam import create_camera_sensor, get_bbx_xys, perspective_projection
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from canonicalize_motion import (
    canonicalize_g1_first_frame_bones_seed,
    canonicalize_smpl_first_frame_bones_seed,
)


def _torch_load_cpu_mmap(path):
    """Load zip-format torch files through mmap, with legacy fallback."""
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
    except RuntimeError as exc:
        if "mmap" in str(exc).lower():
            return torch.load(path, **kwargs)
        raise


# ------------------------------------------------------------------
# SLERP helpers
# ------------------------------------------------------------------

def _slerp_quat(q0, q1, alpha):
    """SLERP between two quaternion tensors (wxyz). alpha in [0,1].
    q0, q1: (..., 4) wxyz.  alpha: scalar or (...,1).
    Returns: (..., 4) normalised wxyz quaternion.
    """
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(-1, keepdim=True)          # (..., 1)
    q1 = torch.where(dot < 0, -q1, q1)             # shortest path
    dot = dot.abs().clamp(0.0, 1.0)

    theta = torch.acos(dot)                         # (..., 1)
    sin_t = theta.sin()

    # linear fallback when theta ≈ 0
    w0 = torch.where(sin_t > 1e-6,
                     torch.sin((1.0 - alpha) * theta) / (sin_t + 1e-10),
                     torch.ones_like(sin_t) * (1.0 - alpha))
    w1 = torch.where(sin_t > 1e-6,
                     torch.sin(alpha * theta) / (sin_t + 1e-10),
                     torch.ones_like(sin_t) * alpha)
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)


def slerp_sequence_quat(q_seq, tgt_len):
    """Resample a quaternion sequence to tgt_len using SLERP.

    Args:
        q_seq: (T, *shape, 4) wxyz quaternions
        tgt_len: int
    Returns:
        (tgt_len, *shape, 4) wxyz quaternions
    """
    T = q_seq.shape[0]
    if T == tgt_len:
        return F.normalize(q_seq, dim=-1)
    if T == 1:
        return F.normalize(q_seq, dim=-1).expand(tgt_len, *q_seq.shape[1:]).clone()

    extra = q_seq.shape[1:]                             # (*shape, 4)
    n_extra = len(extra)

    t_idx = torch.linspace(0, T - 1, tgt_len, device=q_seq.device, dtype=q_seq.dtype)
    i0 = t_idx.floor().long().clamp(0, T - 2)          # (tgt_len,)
    i1 = (i0 + 1).clamp(0, T - 1)
    alpha = (t_idx - i0.to(dtype=q_seq.dtype)).reshape(tgt_len, *([1] * n_extra))  # broadcastable

    q0 = q_seq[i0]   # (tgt_len, *shape, 4)
    q1 = q_seq[i1]
    return _slerp_quat(q0, q1, alpha)


# ==========================================================

# ==========================================================
def npz_get(npz_obj, candidates, required=True, file_for_log=""):
    """Return the first available key from an NPZ file."""
    for k in candidates:
        if k in npz_obj.files:
            return npz_obj[k]
    if required:
        raise KeyError(
            f"Missing keys {candidates} in npz: {file_for_log}\n"
            f"Available keys: {list(npz_obj.files)}"
        )
    return None


# ==========================================================

# ==========================================================
def interpolate_tensor(x, tgt_len):
    """Linearly resample a (T, C) tensor to ``tgt_len`` frames."""
    x = x.unsqueeze(0).transpose(1, 2)  # (1, C, T)
    x_interp = F.interpolate(x, size=tgt_len, mode="linear", align_corners=True)
    return x_interp.transpose(1, 2).squeeze(0)  # (tgt_len, C)


def interpolate_seq(x, tgt_len):
    """Linearly resample a sequence with arbitrary trailing dimensions."""
    T = x.shape[0]
    x_flat = x.reshape(T, -1)
    x_interp = interpolate_tensor(x_flat, tgt_len)
    return x_interp.reshape(tgt_len, *x.shape[1:])


def retime_speed_ratio(src_len, tgt_len, src_fps=None, tgt_fps=None):
    """Velocity multiplier for retiming a clip from src_len frames to tgt_len frames.

    Interpolation uses align_corners=True, so the exact duration ratio is based on
    frame intervals, not frame count. If fps is omitted, source and target are
    assumed to be sampled at the same fps. Values are multiplied by this ratio:
    - src_len > tgt_len: clip is sped up, velocities get larger
    - src_len < tgt_len: clip is slowed down, velocities get smaller
    """
    if src_len <= 1 or tgt_len <= 1:
        return 1.0
    src_fps = 1.0 if src_fps is None else float(src_fps)
    tgt_fps = src_fps if tgt_fps is None else float(tgt_fps)
    src_duration = float(src_len - 1) / src_fps
    tgt_duration = float(tgt_len - 1) / tgt_fps
    if tgt_duration <= 0:
        return 1.0
    return src_duration / tgt_duration


def quat_xyzw_to_wxyz(q):
    """Convert quaternions from xyzw to wxyz order."""
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def quat_wxyz_to_xyzw(q):
    """Convert quaternions from wxyz to xyzw order."""
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


# ==========================================================



# ==========================================================
def az_to_ay_rotmat(device, dtype):
    # Applied via `v @ T.T`: (x, y, z)_zup → (x, -z, y)_yup.
    # i.e. z-up gravity -z becomes y-up gravity -y.
    return torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, -1.0, 0.0]],
        device=device,
        dtype=dtype,
    )


def apply_az_to_ay_on_vec(v):
    """v: (..., 3)"""
    T = az_to_ay_rotmat(v.device, v.dtype)
    return v @ T.T


def apply_az_to_ay_on_quat_wxyz(q_wxyz):
    """q_wxyz: (..., 4)  wxyz format (as stored in NPZ and g1_target)"""
    R = quaternion_to_matrix(q_wxyz)              # (...,3,3)
    T = az_to_ay_rotmat(q_wxyz.device, q_wxyz.dtype)
    R_new = torch.matmul(T, R)                    # world-axis rotation: left-multiply
    return matrix_to_quaternion(R_new)            # back to wxyz


def apply_az_to_ay_on_axis_angle(global_orient_aa):
    T = az_to_ay_rotmat(global_orient_aa.device, global_orient_aa.dtype)
    return matrix_to_axis_angle(T @ axis_angle_to_matrix(global_orient_aa))


# ==========================================================

# G1     canonical (URDF root local: x=fwd, y=left, z=up)
#   x_new = z_old   (forward)
#   y_new = x_old   (left)
#   z_new = y_old   (up)




# ==========================================================
R_SMPL_LOCAL_TO_G1_LOCAL = torch.tensor(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)

# ------------------------------------------------------------------


# ------------------------------------------------------------------
_SMPLX_FOOT_JOINT_IDS = (7, 8, 10, 11)
_SMPLX_FOOT_CACHE: dict = {}


def get_smplx_foot_buffers(
    smplx_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    num_betas=10,
):
    """Load and cache SMPL-X foot-joint template data."""
    key = (str(smplx_path), int(num_betas))
    if key in _SMPLX_FOOT_CACHE:
        return _SMPLX_FOOT_CACHE[key]
    data = np.load(smplx_path, allow_pickle=True)
    v_template  = torch.tensor(np.asarray(data["v_template"]), dtype=torch.float32)
    shapedirs   = torch.tensor(np.asarray(data["shapedirs"][:, :, :num_betas]), dtype=torch.float32)
    J_regressor = torch.tensor(np.asarray(data["J_regressor"]), dtype=torch.float32)
    rows = J_regressor[list(_SMPLX_FOOT_JOINT_IDS)]               # (4, V)
    J_template_feet  = rows @ v_template                            # (4, 3)
    J_shapedirs_feet = torch.einsum("jv,vcb->jcb", rows, shapedirs) # (4, 3, B)
    _SMPLX_FOOT_CACHE[key] = (J_template_feet, J_shapedirs_feet)
    return _SMPLX_FOOT_CACHE[key]


def compute_smpl_floor_height(betas, J_template_pelvis, J_shapedirs_pelvis,
                               J_template_feet, J_shapedirs_feet):
    """Estimate floor height from the shaped pelvis and foot joints."""
    pelvis_y = float((J_template_pelvis + J_shapedirs_pelvis @ betas)[1])
    feet_pos  = J_template_feet + torch.einsum("jcb,b->jc", J_shapedirs_feet, betas)
    min_foot_y = float(feet_pos[:, 1].min())
    return pelvis_y - min_foot_y               # > 0




def apply_smpl_to_g1_axes_on_smpl(global_orient_aa, transl, pelvis_offset_smpl=None):
    """Rotate SMPL world-space root parameters into the robot frame."""
    R_post = R_SMPL_LOCAL_TO_G1_LOCAL.to(device=transl.device, dtype=transl.dtype)
    R_orient = axis_angle_to_matrix(global_orient_aa)
    R_orient_new = R_post.unsqueeze(0) @ R_orient
    global_orient_aa_new = matrix_to_axis_angle(R_orient_new)
    if pelvis_offset_smpl is not None:
        delta = (R_post @ pelvis_offset_smpl) - pelvis_offset_smpl   # (3,)
        transl_new = transl @ R_post.T + delta.unsqueeze(0)          # (F, 3)
    else:
        transl_new = transl @ R_post.T
    return global_orient_aa_new, transl_new, None


# ==========================================================


#   pelvis_world(f) = transl(f) + J_pelvis(betas)

# ==========================================================
_SMPLX_PELVIS_CACHE = {}


def get_smplx_pelvis_buffers(
    smplx_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    num_betas=10,
):
    """Load and cache the SMPL-X pelvis template and shape directions."""
    key = (str(smplx_path), int(num_betas))
    if key in _SMPLX_PELVIS_CACHE:
        return _SMPLX_PELVIS_CACHE[key]

    data = np.load(smplx_path, allow_pickle=True)
    v_template = torch.tensor(np.asarray(data["v_template"]), dtype=torch.float32)                       # (V, 3)
    shapedirs  = torch.tensor(np.asarray(data["shapedirs"][:, :, :num_betas]), dtype=torch.float32)      # (V, 3, B)
    J_regressor = torch.tensor(np.asarray(data["J_regressor"]), dtype=torch.float32)                     # (J, V)

    pelvis_row = J_regressor[0]                                                  # (V,)
    J_template_pelvis  = pelvis_row @ v_template                                 # (3,)
    J_shapedirs_pelvis = torch.einsum("v, vcb -> cb", pelvis_row, shapedirs)     # (3, B)

    _SMPLX_PELVIS_CACHE[key] = (J_template_pelvis, J_shapedirs_pelvis)
    return _SMPLX_PELVIS_CACHE[key]


def compute_smpl_pelvis_offset(betas, J_template_pelvis, J_shapedirs_pelvis):
    """Compute the shaped pelvis offset for one or more beta vectors."""
    return J_template_pelvis + torch.einsum("...b, cb -> ...c", betas, J_shapedirs_pelvis)


# ==========================================================

#


#     global_orient_new(f) = R0^T @ global_orient(f)
#     transl_new(f)        = R0^T @ (transl(f) - transl(0)) - J_pelvis(betas)

#       global_orient_new(0) = I ✓
#



# ==========================================================
def _first_frame_inv_rot(R0, yaw_only=False, up_axis=1, fwd_axis=2):
    """Return the rotation that removes the first-frame pose or heading."""
    if not yaw_only:
        return R0.transpose(-1, -2)
    fwd = R0[:, fwd_axis]                                  
    if up_axis == 1:
        heading = torch.atan2(fwd[0], fwd[2])
    elif up_axis == 2:
        heading = torch.atan2(fwd[1], fwd[0])
    else:  # up_axis == 0
        heading = torch.atan2(fwd[2], fwd[1])
    aa = torch.zeros(3, device=R0.device, dtype=R0.dtype)
    aa[up_axis] = -heading
    return axis_angle_to_matrix(aa)                       


def canonicalize_smpl_first_frame(global_orient_aa, transl, pelvis_offset=None,
                                  yaw_only=False, up_axis=1, fwd_axis=2):
    """Canonicalize an SMPL sequence relative to its first frame."""
    R = axis_angle_to_matrix(global_orient_aa)            # (F, 3, 3)
    R0_inv = _first_frame_inv_rot(R[0], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis)  # (3, 3)
    R_new = R0_inv @ R                                    # (F, 3, 3)
    transl_new = (transl - transl[0:1]) @ R0_inv.T        

    if pelvis_offset is not None:
        
        transl_new = transl_new - pelvis_offset
    return matrix_to_axis_angle(R_new), transl_new


def canonicalize_g1_first_frame(g1, root_body_id=0, yaw_only=False, up_axis=2, fwd_axis=0):
    """Canonicalize robot world-space poses relative to the first frame."""
    body_quat_w   = g1["body_quat_w"]                                  # (F, N, 4) wxyz
    body_pos_w    = g1["body_pos_w"]                                   # (F, N, 3)

    R = quaternion_to_matrix(body_quat_w)                              # (F, N, 3, 3)
    R0_inv = _first_frame_inv_rot(R[0, root_body_id], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis)  # (3, 3)
    center = body_pos_w[0, root_body_id].clone()                       # (3,)

    R_new = R0_inv @ R                                                 # (F, N, 3, 3)
    q_new = matrix_to_quaternion(R_new)                                # (F, N, 4) wxyz, ambiguous ±q
    
    sign = torch.where(q_new[..., 0:1] < 0, -torch.ones_like(q_new[..., 0:1]), torch.ones_like(q_new[..., 0:1]))
    q_new = q_new * sign
    if not yaw_only:
        
        
        q_new[0, root_body_id] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q_new.device, dtype=q_new.dtype)

    g1["body_quat_w"]    = q_new                                       # (F, N, 4) wxyz
    g1["body_pos_w"]     = (body_pos_w - center) @ R0_inv.T            # (F, N, 3)
    g1["body_lin_vel_w"] = g1["body_lin_vel_w"] @ R0_inv.T             # (F, N, 3)
    g1["body_ang_vel_w"] = g1["body_ang_vel_w"] @ R0_inv.T             # (F, N, 3)
    
    g1["body_pos_w"][0, root_body_id] = 0.0
    return g1


# ==========================================================



# ==========================================================
def interpolate_g1_params(g1_data, tgt_len, speed_ratio=1.0, fps_out=None):
    """Resample robot poses and velocities to a target frame count."""
    speed_ratio = float(speed_ratio)

    # --- joint DOFs and positions: linear ---
    joint_pos = interpolate_seq(g1_data["joint_pos"], tgt_len)

    # global translation: linear
    body_pos_w = interpolate_seq(g1_data["body_pos_w"], tgt_len)

    # global orientation: already wxyz in NPZ → SLERP → keep as wxyz
    quat_wxyz = g1_data["body_quat_w"]                       # (T, N, 4) wxyz
    body_quat_w_i = slerp_sequence_quat(quat_wxyz, tgt_len)  # (tgt_len, N, 4) wxyz

    
    
    
    if fps_out is not None and tgt_len >= 2:
        # joint_vel: rad/s
        jp_diff = joint_pos[1:] - joint_pos[:-1]                         # (tgt_len-1, 29)
        jp_diff = torch.cat([jp_diff, jp_diff[-1:].clone()], dim=0)      # (tgt_len, 29)
        joint_vel = jp_diff * float(fps_out)
        # body_lin_vel_w: m/s
        diff = body_pos_w[1:] - body_pos_w[:-1]
        diff = torch.cat([diff, diff[-1:].clone()], dim=0)
        body_lin_vel_w = diff * float(fps_out)
    else:
        joint_vel = interpolate_seq(g1_data["joint_vel"], tgt_len) * speed_ratio
        body_lin_vel_w = interpolate_seq(g1_data["body_lin_vel_w"], tgt_len) * speed_ratio
    body_ang_vel_w = interpolate_seq(g1_data["body_ang_vel_w"], tgt_len) * speed_ratio

    return {
        "fps": g1_data.get("fps", None),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w_i,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }


# ==========================================================

# ==========================================================
class G1AmassDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        amass_subdir="amass",
        g1_subdir="g1",
        motion_frames=120,
        l_factor=1.5,
        cam_augmentation="v11",
        limit_size=None,
        betas_dim=10,
        root_body_id=0,
        apply_az_to_ay_g1=True,
        canonicalize_first_frame=True,
        align_target="pelvis",
        floor_adjust=False,  
        
        
        yaw_only_canon=True,
        smplx_neutral_npz_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    ):
        self.data_dir = Path(data_dir)
        self.amass_dir = self.data_dir / amass_subdir
        self.g1_dir = self.data_dir / g1_subdir
        self.motion_frames = motion_frames
        self.l_factor = l_factor
        self.dataset_name = "G1_AMASS"
        self.betas_dim = betas_dim
        self.root_body_id = root_body_id
        self.apply_az_to_ay_g1 = apply_az_to_ay_g1
        self.canonicalize_first_frame = canonicalize_first_frame
        self.floor_adjust = floor_adjust
        self.yaw_only_canon = bool(yaw_only_canon)
        assert align_target in ("pelvis", "transl"), align_target
        self.align_target = align_target
        if canonicalize_first_frame and align_target == "pelvis":
            self._J_template_pelvis, self._J_shapedirs_pelvis = get_smplx_pelvis_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_pelvis = None
            self._J_shapedirs_pelvis = None
        if floor_adjust and canonicalize_first_frame and align_target == "pelvis":
            self._J_template_feet, self._J_shapedirs_feet = get_smplx_foot_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_feet = None
            self._J_shapedirs_feet = None

        
        
        self._skip_betas_aug = True

        super().__init__(cam_augmentation, limit_size)

    # =====================================================
    
    # =====================================================
    def _load_dataset(self):
        Log.info(f"[{self.dataset_name}] scanning AMASS files in {self.amass_dir}")
        Log.info(f"[{self.dataset_name}] scanning robot files in {self.g1_dir}")

        amass_files = {f.name for f in self.amass_dir.glob("*.npz")}
        g1_files = {f.name for f in self.g1_dir.glob("*.npz")}

        paired_names = sorted(amass_files & g1_files)

        amass_only = amass_files - g1_files
        g1_only = g1_files - amass_files
        if amass_only:
            Log.warning(
                f"[{self.dataset_name}] {len(amass_only)} AMASS files have no robot pair"
            )
        if g1_only:
            Log.warning(
                f"[{self.dataset_name}] {len(g1_only)} robot files have no AMASS pair"
            )

        self.paired_files = [
            {"amass": self.amass_dir / name, "g1": self.g1_dir / name, "name": Path(name).stem}
            for name in paired_names
        ]
        Log.info(f"[{self.dataset_name}] paired files={len(self.paired_files)}")

    # =====================================================
    
    # =====================================================
    def _get_idx2meta(self):
        
        
        self.idx2meta = []
        skipped = 0

        amass_fps_missing = 0
        for pair in self.paired_files:
            with np.load(pair["amass"], allow_pickle=False) as amass_data, np.load(pair["g1"]) as g1_data:
                amass_len = npz_get(amass_data, ["trans", "transl"], file_for_log=str(pair["amass"])).shape[0]
                amass_fps_val = npz_get(
                    amass_data,
                    ["mocap_framerate", "mocap_frame_rate", "fps"],
                    required=False,
                    file_for_log=str(pair["amass"]),
                )
                if amass_fps_val is None:
                    amass_fps = 120.0
                    amass_fps_missing += 1
                else:
                    amass_fps = float(amass_fps_val)

                g1_len = npz_get(g1_data, ["joint_pos"], file_for_log=str(pair["g1"])).shape[0]
                g1_fps = float(g1_data["fps"]) if "fps" in g1_data.files else 50.0

            fps_ratio = float(amass_fps) / float(g1_fps)
            amass_len_in_g1_frames = int(amass_len / fps_ratio)
            length = min(amass_len_in_g1_frames, g1_len)
            if length < 25:
                skipped += 1
                continue

            num_samples = max(length // self.motion_frames, 1)
            meta = {
                "amass_file": pair["amass"],
                "g1_file":    pair["g1"],
                "usable_len": length,        
                "fps_ratio":  fps_ratio,     # amass_fps / g1_fps
                "name":       pair["name"],
            }
            self.idx2meta.extend([meta] * num_samples)

        Log.info(
            f"[{self.dataset_name}] samples={len(self.idx2meta)} skipped={skipped}"
        )
        if amass_fps_missing > 0:
            Log.warning(
                f"[{self.dataset_name}] {amass_fps_missing} AMASS files do not "
                "declare an FPS value; using 120 FPS."
            )

    # =====================================================
    
    # =====================================================
    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        usable_len = meta["usable_len"]     
        fps_ratio  = meta["fps_ratio"]      # amass_fps / g1_fps
        tgt_len    = self.motion_frames

        
        
        lo = max(2,      int(float(tgt_len) / self.l_factor / fps_ratio))
        hi = max(lo + 1, int(float(tgt_len) * self.l_factor / fps_ratio))
        raw_subset_len = np.random.randint(lo, hi)
        if raw_subset_len <= usable_len:
            start_g1 = np.random.randint(0, usable_len - raw_subset_len + 1)
            end_g1   = start_g1 + raw_subset_len
        else:
            start_g1, end_g1 = 0, usable_len

        
        start_amass = int(round(start_g1 * fps_ratio))
        end_amass   = int(round(end_g1   * fps_ratio))
        if end_amass - start_amass < 2:
            end_amass = start_amass + 2

        
        start = start_g1
        end   = end_g1

        with np.load(meta["amass_file"], allow_pickle=False) as amass_raw, np.load(meta["g1_file"]) as g1_raw:
            # -------------------------
            
            # -------------------------
            
            
            trans = npz_get(amass_raw, ["trans", "transl"], file_for_log=str(meta["amass_file"]))[start_amass:end_amass]
            root_orient = npz_get(amass_raw, ["root_orient", "global_orient"], file_for_log=str(meta["amass_file"]))[start_amass:end_amass]
            pose_body = npz_get(amass_raw, ["pose_body", "body_pose"], file_for_log=str(meta["amass_file"]))[start_amass:end_amass]
            betas = npz_get(amass_raw, ["betas"], file_for_log=str(meta["amass_file"]))

            smpl_data = {
                
                "transl": torch.tensor(trans, dtype=torch.float32),
                "global_orient": torch.tensor(root_orient, dtype=torch.float32),
                "body_pose": torch.tensor(pose_body, dtype=torch.float32),
                "betas": torch.tensor(betas, dtype=torch.float32),
            }

            # betas: (16,) -> (T,16)
            actual_len = end_amass - start_amass
            if smpl_data["betas"].ndim == 1:
                smpl_data["betas"] = smpl_data["betas"][: self.betas_dim]
                smpl_data["betas"] = smpl_data["betas"].unsqueeze(0).expand(actual_len, -1)
            else:
                smpl_data["betas"] = smpl_data["betas"][:actual_len, : self.betas_dim]

            # -------------------------
            
            # -------------------------
            required_g1_keys = [
                "joint_pos", "joint_vel",
                "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w",
            ]
            missing = [k for k in required_g1_keys if k not in g1_raw.files]
            if missing:
                raise KeyError(
                    f"Missing robot arrays {missing} in {meta['g1_file']}; "
                    f"available keys: {list(g1_raw.files)}"
                )

            g1_data = {
                "fps": torch.tensor(float(g1_raw["fps"]), dtype=torch.float32) if "fps" in g1_raw.files else None,
                "joint_pos": torch.tensor(g1_raw["joint_pos"][start:end], dtype=torch.float32),
                "joint_vel": torch.tensor(g1_raw["joint_vel"][start:end], dtype=torch.float32),
                "body_pos_w": torch.tensor(g1_raw["body_pos_w"][start:end], dtype=torch.float32),
                "body_quat_w": torch.tensor(g1_raw["body_quat_w"][start:end], dtype=torch.float32),  # wxyz
                "body_lin_vel_w": torch.tensor(g1_raw["body_lin_vel_w"][start:end], dtype=torch.float32),
                "body_ang_vel_w": torch.tensor(g1_raw["body_ang_vel_w"][start:end], dtype=torch.float32),
            }

        
        smpl_interpolated = interpolate_smpl_params(smpl_data, tgt_len)
        g1_interpolated = interpolate_g1_params(g1_data, tgt_len)

        
        smpl_interpolated["global_orient"], smpl_interpolated["transl"], _ = get_tgtcoord_rootparam(
            smpl_interpolated["global_orient"],
            smpl_interpolated["transl"],
            tsf="az->ay",
        )

        
        
        
        
        
        
        
        if self.canonicalize_first_frame:
            g1_interpolated = canonicalize_g1_first_frame(
                g1_interpolated, root_body_id=self.root_body_id,
                yaw_only=self.yaw_only_canon, up_axis=2, fwd_axis=0,   # G1 z-up, URDF forward=+x
            )

        if self.apply_az_to_ay_g1:
            g1_interpolated["body_pos_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_pos_w"])
            g1_interpolated["body_lin_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_lin_vel_w"])
            g1_interpolated["body_ang_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_ang_vel_w"])
            g1_interpolated["body_quat_w"] = apply_az_to_ay_on_quat_wxyz(g1_interpolated["body_quat_w"])

            g1_interpolated["body_pos_w"] = _apply_ry_neg90_on_vec(g1_interpolated["body_pos_w"])
            g1_interpolated["body_lin_vel_w"] = _apply_ry_neg90_on_vec(g1_interpolated["body_lin_vel_w"])
            g1_interpolated["body_ang_vel_w"] = _apply_ry_neg90_on_vec(g1_interpolated["body_ang_vel_w"])
            g1_interpolated["body_quat_w"] = _apply_ry_neg90_on_quat_wxyz(g1_interpolated["body_quat_w"])

        
        
        if self.canonicalize_first_frame:
            
            
            if self.align_target == "pelvis":
                pelvis_offset = compute_smpl_pelvis_offset(
                    smpl_interpolated["betas"][0], self._J_template_pelvis, self._J_shapedirs_pelvis
                )
            else:
                pelvis_offset = None
            smpl_interpolated["global_orient"], smpl_interpolated["transl"] = canonicalize_smpl_first_frame(
                smpl_interpolated["global_orient"], smpl_interpolated["transl"], pelvis_offset=pelvis_offset,
                yaw_only=self.yaw_only_canon, up_axis=1, fwd_axis=2,   # SMPL y-up, forward=+z
            )
        else:
            
            g1_root_first = g1_interpolated["body_pos_w"][0, self.root_body_id].clone()
            smpl_transl_first = smpl_interpolated["transl"][0].clone()
            g1_world_offset = smpl_transl_first - g1_root_first
            g1_interpolated["body_pos_w"] = g1_interpolated["body_pos_w"] + g1_world_offset[None, None, :]

        
        if self.floor_adjust and self.canonicalize_first_frame:
            
            g1_floor_y = float(-g1_interpolated["body_pos_w"][0, :, 1].min())
            g1_interpolated["body_pos_w"][:, :, 1] += g1_floor_y
            
            with torch.no_grad():
                j24 = self.smplx_lite(
                    smpl_interpolated["body_pose"][:1],
                    smpl_interpolated["betas"][:1],
                    smpl_interpolated["global_orient"][:1],
                    smpl_interpolated["transl"][:1],
                )[0]
            smpl_floor_y = float(-j24[:, 1].min())
            smpl_interpolated["transl"][:, 1] += smpl_floor_y

        
        return {
            "data_name": "g1_paired",
            # human
            "body_pose": smpl_interpolated["body_pose"],
            "betas": smpl_interpolated["betas"],
            "global_orient": smpl_interpolated["global_orient"],
            "transl": smpl_interpolated["transl"],

            # g1 (all)
            "g1_fps": g1_interpolated["fps"] if g1_interpolated["fps"] is not None else torch.tensor(0.0),
            "g1_joint_pos": g1_interpolated["joint_pos"],
            "g1_joint_vel": g1_interpolated["joint_vel"],
            "g1_body_pos_w": g1_interpolated["body_pos_w"],
            "g1_body_quat_w": g1_interpolated["body_quat_w"],
            "g1_body_lin_vel_w": g1_interpolated["body_lin_vel_w"],
            "g1_body_ang_vel_w": g1_interpolated["body_ang_vel_w"],
        }

    def _process_data(self, data, idx):
        
        _ = data.pop("g1_fps")
        g1_joint_pos = data.pop("g1_joint_pos")            # (T, 29)
        g1_joint_vel = data.pop("g1_joint_vel")             # (T, 29)
        g1_body_pos_w = data.pop("g1_body_pos_w")           # (T, 30, 3)
        g1_body_quat_w = data.pop("g1_body_quat_w")         # (T, 30, 4) wxyz
        g1_body_lin_vel_w = data.pop("g1_body_lin_vel_w")   # (T, 30, 3)
        g1_body_ang_vel_w = data.pop("g1_body_ang_vel_w")   # (T, 30, 3)

        g1_target = {
            "g1_joint_pos": g1_joint_pos,       # (T, 29)
            "g1_joint_vel": g1_joint_vel,       # (T, 29)
            "g1_body_pos_w": g1_body_pos_w,     # (T, 30, 3)
            "g1_body_quat_w": g1_body_quat_w,   # (T, 30, 4) wxyz
            "g1_body_lin_vel_w": g1_body_lin_vel_w,  # (T, 30, 3)
            "g1_body_ang_vel_w": g1_body_ang_vel_w,  # (T, 30, 3)
        }

        return_data = super()._process_data(data, idx)
        
        
        
        
        g1_T_w2c = return_data["T_w2c"].clone()
        assert g1_T_w2c.shape[0] == return_data["T_w2c"].shape[0]
        g1_T_w2c[..., :3, 3] = 0.9 * return_data["T_w2c"][..., :3, 3]
        g1_target["g1_T_w2c"] = g1_T_w2c
        return_data["g1_target"] = g1_target

        
        
        max_len = return_data["T_w2c"].shape[0]
        for k in list(g1_target.keys()):
            v = g1_target[k]
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] != max_len:
                g1_target[k] = repeat_to_max_len(v, max_len)

        
        
        
        

        return return_data


# ==========================================================




# ==========================================================
import re


def _normalize_underscores(s):
    """Collapse repeated underscores for robust filename matching."""
    return re.sub(r"_+", "_", s)


def _candidates_from_pth_key(k):
    """Build normalized filename candidates from a motion dictionary key."""
    parts = k.split("/")
    cands = set()
    cands.add("_".join(parts[3:]))      
    if len(parts) > 4:
        cands.add("_".join(parts[4:]))  
    return {_normalize_underscores(c) for c in cands}


class G1AmassPthDataset(BaseDataset):
    """Load AMASS motions from a PTH mapping and paired robot NPZ files."""

    def __init__(
        self,
        data_dir,
        pth_path,
        g1_subdir="g1",
        amass_fps=30.0,
        motion_frames=120,
        l_factor=1.5,
        cam_augmentation="v11",
        limit_size=None,
        betas_dim=10,
        root_body_id=0,
        apply_az_to_ay_g1=True,
        skip_moyo=True,
        skip_substrings=("weiz",),
        canonicalize_first_frame=True,
        align_target="pelvis",
        floor_adjust=False,
        # ratio_lo/ratio_hi: per-axis range for k=G1/AMASS LSQ slope (x, y, z)
        
        ratio_lo=(0.7, 0.7, -float("inf")),
        ratio_hi=(1.05, 1.05,  float("inf")),
        ratio_thr=0.1,                
        ratio_n_retry=3,
        smplx_neutral_npz_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    ):
        self.data_dir = Path(data_dir)
        self.g1_dir = self.data_dir / g1_subdir
        self.pth_path = Path(pth_path)
        self.amass_fps = float(amass_fps)
        self.motion_frames = motion_frames
        self.l_factor = l_factor
        self.dataset_name = "G1_AMASS_PTH"
        self.betas_dim = betas_dim
        self.root_body_id = root_body_id
        self.apply_az_to_ay_g1 = apply_az_to_ay_g1
        self.skip_moyo = skip_moyo
        
        
        _sk = [str(s) for s in (skip_substrings or ())]
        if skip_moyo and not any("moyo" in s.lower() for s in _sk):
            _sk.append("moyo_smplxn")
        self.skip_substrings = tuple(_sk)
        self.canonicalize_first_frame = canonicalize_first_frame
        self.floor_adjust = floor_adjust
        # Root translation ratio filter (k = G1 / AMASS, per axis LSQ slope).
        
        
        
        
        ratio_lo = tuple(float(v) for v in ratio_lo)
        ratio_hi = tuple(float(v) for v in ratio_hi)
        assert len(ratio_lo) == 3 and len(ratio_hi) == 3, "ratio bounds must have three axes"
        self.ratio_lo = ratio_lo
        self.ratio_hi = ratio_hi
        self.ratio_thr = float(ratio_thr)
        self.ratio_n_retry = int(ratio_n_retry)
        assert self.ratio_n_retry >= 1, "ratio_n_retry must be at least one"
        self._ratio_stats = {"accept": 0, "retry": 0, "skip": 0}
        assert align_target in ("pelvis", "transl"), align_target
        self.align_target = align_target
        if canonicalize_first_frame and align_target == "pelvis":
            self._J_template_pelvis, self._J_shapedirs_pelvis = get_smplx_pelvis_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_pelvis = None
            self._J_shapedirs_pelvis = None
        if floor_adjust and canonicalize_first_frame and align_target == "pelvis":
            self._J_template_feet, self._J_shapedirs_feet = get_smplx_foot_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_feet = None
            self._J_shapedirs_feet = None

        
        
        self._skip_betas_aug = True

        super().__init__(cam_augmentation, limit_size)

    def _load_dataset(self):
        Log.info(f"[{self.dataset_name}] loading motions from {self.pth_path}")
        tic = Log.time()
        self.motion_files = torch.load(self.pth_path, weights_only=False)
        Log.info(
            f"[{self.dataset_name}] loaded {len(self.motion_files)} motions "
            f"in {Log.time() - tic:.2f}s"
        )

        
        self._g1name_to_pth_key = {}
        ambiguous = 0
        skipped = 0
        for k in self.motion_files.keys():
            if any(s.lower() in k.lower() for s in self.skip_substrings):
                skipped += 1
                continue
            for c in _candidates_from_pth_key(k):
                if c in self._g1name_to_pth_key:
                    ambiguous += 1
                else:
                    self._g1name_to_pth_key[c] = k
        Log.info(
            f"[{self.dataset_name}] indexed={len(self._g1name_to_pth_key)} "
            f"ambiguous={ambiguous} skipped={skipped} "
            f"skip_substrings={self.skip_substrings}"
        )

        
        Log.info(f"[{self.dataset_name}] scanning robot files in {self.g1_dir}")
        g1_files = sorted(self.g1_dir.glob("*.npz"))

        self.paired_files = []
        unmatched = []
        for g1_path in g1_files:
            norm = _normalize_underscores(g1_path.name)
            pth_key = self._g1name_to_pth_key.get(norm)
            if pth_key is None:
                unmatched.append(g1_path.name)
                continue
            self.paired_files.append(
                {"amass_key": pth_key, "g1": g1_path, "name": g1_path.stem}
            )

        Log.info(
            f"[{self.dataset_name}] paired files={len(self.paired_files)}"
        )
        if unmatched:
            Log.warning(
                f"[{self.dataset_name}] unmatched robot files={len(unmatched)}; "
                f"examples={unmatched[:5]}"
            )

    def _get_idx2meta(self):
        
        
        
        
        self.idx2meta = []
        skipped = 0

        
        
        import json, hashlib
        cache_dir = Path(".cache_g1_meta")
        cache_dir.mkdir(exist_ok=True)
        cache_key = hashlib.md5(str(self.g1_dir.resolve()).encode()).hexdigest()[:12]
        cache_file = cache_dir / f"g1_meta_{cache_key}.json"
        g1_meta_cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    g1_meta_cache = json.load(f)
            except Exception:
                g1_meta_cache = {}

        cache_dirty = False
        for pair in self.paired_files:
            entry = self.motion_files[pair["amass_key"]]
            amass_len = entry["pose"].shape[0]   

            g1_path_str = str(pair["g1"])
            mtime = pair["g1"].stat().st_mtime
            ck = f"{g1_path_str}::{mtime}"
            if ck in g1_meta_cache:
                g1_len_src, g1_fps_src = g1_meta_cache[ck]
                g1_fps_src = float(g1_fps_src)
            else:
                with np.load(pair["g1"]) as g1_data:
                    g1_len_src = int(g1_data["joint_pos"].shape[0])
                    g1_fps_src = float(g1_data["fps"]) if "fps" in g1_data.files else 50.0
                g1_meta_cache[ck] = [g1_len_src, g1_fps_src]
                cache_dirty = True

            
            g1_len_30 = max(2, int(round(g1_len_src * float(self.amass_fps) / g1_fps_src)))
            length = min(amass_len, g1_len_30)   
            if length < 25:
                skipped += 1
                continue

            num_samples = max(length // self.motion_frames, 1)
            meta = {
                "amass_key":  pair["amass_key"],
                "g1_file":    pair["g1"],
                "usable_len": length,        
                "g1_fps_src": g1_fps_src,    
                "name":       pair["name"],
            }
            self.idx2meta.extend([meta] * num_samples)

        if cache_dirty:
            try:
                with open(cache_file, "w") as f:
                    json.dump(g1_meta_cache, f)
            except Exception:
                pass

        Log.info(
            f"[{self.dataset_name}] samples={len(self.idx2meta)} skipped={skipped}"
        )

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        usable_len = meta["usable_len"]      
        g1_fps_src = meta["g1_fps_src"]      
        tgt_len    = self.motion_frames

        
        
        
        lo = max(2, int(float(tgt_len) / self.l_factor))
        hi = max(lo + 1, int(float(tgt_len) * self.l_factor) + 1)
        raw_subset_len = np.random.randint(lo, hi)
        if raw_subset_len <= usable_len:
            start = np.random.randint(0, usable_len - raw_subset_len + 1)
            end   = start + raw_subset_len
        else:
            start, end = 0, usable_len

        # ========== AMASS from .pth (already at 30fps) ==========
        entry = self.motion_files[meta["amass_key"]]
        pose_full = entry["pose"][start:end]   # (F, 66) torch.float32
        trans = entry["trans"][start:end]      # (F, 3)  torch.float32
        beta = entry["beta"]                   # (10,)   torch.float32

        actual_len = pose_full.shape[0]
        smpl_data = {
            "transl": trans.float(),
            "global_orient": pose_full[:, :3].float(),
            "body_pose": pose_full[:, 3:].float(),
            "betas": beta[: self.betas_dim].unsqueeze(0).expand(actual_len, -1).float(),
        }

        
        with np.load(meta["g1_file"]) as g1_raw:
            required_g1_keys = [
                "joint_pos", "joint_vel",
                "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w",
            ]
            missing = [k for k in required_g1_keys if k not in g1_raw.files]
            if missing:
                raise KeyError(
                    f"Missing robot arrays {missing} in {meta['g1_file']}"
                )
            g1_full_src = {
                "fps": torch.tensor(g1_fps_src, dtype=torch.float32),
                "joint_pos":      torch.tensor(g1_raw["joint_pos"], dtype=torch.float32),
                "joint_vel":      torch.tensor(g1_raw["joint_vel"], dtype=torch.float32),
                "body_pos_w":     torch.tensor(g1_raw["body_pos_w"], dtype=torch.float32),
                "body_quat_w":    torch.tensor(g1_raw["body_quat_w"], dtype=torch.float32),
                "body_lin_vel_w": torch.tensor(g1_raw["body_lin_vel_w"], dtype=torch.float32),
                "body_ang_vel_w": torch.tensor(g1_raw["body_ang_vel_w"], dtype=torch.float32),
            }

        
        
        
        amass_len = entry["pose"].shape[0]
        g1_len_30 = amass_len
        g1_at_30fps = interpolate_g1_params(g1_full_src, g1_len_30, speed_ratio=1.0)

        
        
        assert g1_at_30fps["joint_pos"].shape[0] == amass_len,\
            f"G1 resample length {g1_at_30fps['joint_pos'].shape[0]} != AMASS {amass_len}"
        end_g1 = end
        g1_slice = {
            "fps":            g1_at_30fps["fps"],
            "joint_pos":      g1_at_30fps["joint_pos"][start:end_g1],
            "joint_vel":      g1_at_30fps["joint_vel"][start:end_g1],
            "body_pos_w":     g1_at_30fps["body_pos_w"][start:end_g1],
            "body_quat_w":    g1_at_30fps["body_quat_w"][start:end_g1],
            "body_lin_vel_w": g1_at_30fps["body_lin_vel_w"][start:end_g1],
            "body_ang_vel_w": g1_at_30fps["body_ang_vel_w"][start:end_g1],
        }

        
        smpl_interpolated = interpolate_smpl_params(smpl_data, tgt_len)
        
        
        
        
        src_len_30 = g1_slice["joint_pos"].shape[0]
        speed_ratio_aug = (src_len_30 - 1) / max(tgt_len - 1, 1)
        
        
        fps_out = float(self.amass_fps)
        g1_interpolated = interpolate_g1_params(
            g1_slice, tgt_len, speed_ratio=speed_ratio_aug, fps_out=fps_out
        )

        smpl_interpolated["global_orient"], smpl_interpolated["transl"], _ = get_tgtcoord_rootparam(
            smpl_interpolated["global_orient"],
            smpl_interpolated["transl"],
            tsf="az->ay",
        )

        
        
        
        
        if self.canonicalize_first_frame:
            g1_interpolated = canonicalize_g1_first_frame(g1_interpolated, root_body_id=self.root_body_id)

        if self.apply_az_to_ay_g1:
            
            
            
            
            
            
            g1_interpolated["body_pos_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_pos_w"])
            g1_interpolated["body_lin_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_lin_vel_w"])
            g1_interpolated["body_ang_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_ang_vel_w"])
            g1_interpolated["body_quat_w"] = apply_az_to_ay_on_quat_wxyz(g1_interpolated["body_quat_w"])

        if self.canonicalize_first_frame:
            
            
            if self.align_target == "pelvis":
                pelvis_offset = compute_smpl_pelvis_offset(
                    smpl_interpolated["betas"][0], self._J_template_pelvis, self._J_shapedirs_pelvis
                )
            else:
                pelvis_offset = None
            smpl_interpolated["global_orient"], smpl_interpolated["transl"] = canonicalize_smpl_first_frame(
                smpl_interpolated["global_orient"], smpl_interpolated["transl"], pelvis_offset=pelvis_offset
            )

            
            
        else:
            g1_root_first = g1_interpolated["body_pos_w"][0, self.root_body_id].clone()
            smpl_transl_first = smpl_interpolated["transl"][0].clone()
            g1_world_offset = smpl_transl_first - g1_root_first
            g1_interpolated["body_pos_w"] = g1_interpolated["body_pos_w"] + g1_world_offset[None, None, :]

        
        if self.floor_adjust and self.canonicalize_first_frame:
            g1_floor_y = float(-g1_interpolated["body_pos_w"][0, :, 1].min())
            g1_interpolated["body_pos_w"][:, :, 1] += g1_floor_y
            
            
            with torch.no_grad():
                j24 = self.smplx_lite(
                    smpl_interpolated["body_pose"][:1],
                    smpl_interpolated["betas"][:1],
                    smpl_interpolated["global_orient"][:1],
                    smpl_interpolated["transl"][:1],
                )[0]   # (24, 3) world
            smpl_floor_y = float(-j24[:, 1].min())
            smpl_interpolated["transl"][:, 1] += smpl_floor_y

        return {
            "data_name": "g1_paired",
            "body_pose": smpl_interpolated["body_pose"],
            "betas": smpl_interpolated["betas"],
            "global_orient": smpl_interpolated["global_orient"],
            "transl": smpl_interpolated["transl"],
            "g1_fps": g1_interpolated["fps"] if g1_interpolated["fps"] is not None else torch.tensor(0.0),
            "g1_joint_pos": g1_interpolated["joint_pos"],
            "g1_joint_vel": g1_interpolated["joint_vel"],
            "g1_body_pos_w": g1_interpolated["body_pos_w"],
            "g1_body_quat_w": g1_interpolated["body_quat_w"],
            "g1_body_lin_vel_w": g1_interpolated["body_lin_vel_w"],
            "g1_body_ang_vel_w": g1_interpolated["body_ang_vel_w"],
        }

    def _process_data(self, data, idx):
        
        _ = data.pop("g1_fps")
        g1_target = {
            "g1_joint_pos":      data.pop("g1_joint_pos"),
            "g1_joint_vel":      data.pop("g1_joint_vel"),
            "g1_body_pos_w":     data.pop("g1_body_pos_w"),
            "g1_body_quat_w":    data.pop("g1_body_quat_w"),
            "g1_body_lin_vel_w": data.pop("g1_body_lin_vel_w"),
            "g1_body_ang_vel_w": data.pop("g1_body_ang_vel_w"),
        }
        return_data = super()._process_data(data, idx)
        
        
        
        
        g1_T_w2c = return_data["T_w2c"].clone()
        assert g1_T_w2c.shape[0] == return_data["T_w2c"].shape[0]
        g1_T_w2c[..., :3, 3] = 0.9 * return_data["T_w2c"][..., :3, 3]
        g1_target["g1_T_w2c"] = g1_T_w2c
        return_data["g1_target"] = g1_target

        
        
        max_len = return_data["T_w2c"].shape[0]
        for k in list(g1_target.keys()):
            v = g1_target[k]
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] != max_len:
                g1_target[k] = repeat_to_max_len(v, max_len)

        
        
        
        

        return return_data

    # ----------------------------------------------------------
    # Root translation ratio filter
    # ----------------------------------------------------------
    def _root_axis_ratios(self, data):
        """Compute per-axis least-squares robot-to-human translation ratios."""
        transl = data["transl"]                                # (F, 3) AMASS pelvis-mode transl
        g1_pos = data["g1_body_pos_w"][:, self.root_body_id]   # (F, 3) G1 root link
        a_delta = (transl     - transl[0:1]).detach().cpu().numpy()
        g_delta = (g1_pos     - g1_pos[0:1]).detach().cpu().numpy()
        out = []
        for ax in range(3):
            a = a_delta[:, ax]
            g = g_delta[:, ax]
            
            if float(np.abs(a).max()) <= self.ratio_thr:
                out.append(None)
                continue
            
            mask = np.abs(a) > self.ratio_thr
            if int(mask.sum()) < 2:
                out.append(None)
                continue
            am, gm = a[mask], g[mask]
            denom = float((am * am).sum())
            if denom < 1e-12:
                out.append(None)
                continue
            out.append(float((gm * am).sum() / denom))
        return out

    def _ratio_pass(self, data):
        ks = self._root_axis_ratios(data)
        for ax, k in enumerate(ks):
            if k is None:
                continue
            if not (self.ratio_lo[ax] <= k <= self.ratio_hi[ax]):
                return False, ks
        return True, ks

    def __getitem__(self, idx):
        
        
        
        start_idx = idx
        max_skip  = len(self.idx2meta)
        skip_n    = 0
        last_data = None
        last_idx  = idx
        while skip_n < max_skip:
            data = self._load_data(idx)
            last_data = data
            last_idx  = idx
            ok, _ks = self._ratio_pass(data)
            if ok:
                self._ratio_stats["accept"] += 1
                return self._process_data(data, idx)
            
            self._ratio_stats["skip"] += 1
            skip_n += 1
            idx = (idx + 1) % len(self.idx2meta)
            if idx == start_idx:
                break
        
        Log.warning(
            f"[{self.dataset_name}] ratio filter rejected every candidate from "
            f"start_idx={start_idx}; using idx={last_idx}"
        )
        return self._process_data(last_data, last_idx)


# ==========================================================

# ----------------------------------------------------------

#   AMASS: motion.pth   {pose(F,66) AA / trans(F,3) / beta(10)}  y-up, 30fps
#   G1:    g1.pth       {root_pos(F,3) / root_rot(F,4) xyzw / dof_pos(F,29) MJC}












# ==========================================================

import sys as _sys



_BYD_JOINT_NAMES = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
    'left_knee_joint', 'right_knee_joint',
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
    'left_ankle_roll_joint', 'right_ankle_roll_joint',
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_joint', 'right_elbow_joint',
    'left_wrist_roll_joint', 'right_wrist_roll_joint',
    'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_wrist_yaw_joint',
]
_MJC_JOINT_NAMES = [
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
    'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw',
    'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
    'left_elbow', 'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
    'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
]



_MJC_TO_BYD = [_MJC_JOINT_NAMES.index(b[:-6]) for b in _BYD_JOINT_NAMES]




_BYD_TO_MJC = [_BYD_JOINT_NAMES.index(n + '_joint') for n in _MJC_JOINT_NAMES]

_G5_AXIS_MJC = torch.tensor([
    [0, 1, 0], [1, 0, 0], [0, 0, 1],
    [0, 1, 0], [0, 1, 0], [1, 0, 0],
    [0, 1, 0], [1, 0, 0], [0, 0, 1],
    [0, 1, 0], [0, 1, 0], [1, 0, 0],
    [0, 0, 1], [1, 0, 0], [0, 1, 0],
    [0, 1, 0], [1, 0, 0], [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [0, 1, 0], [1, 0, 0], [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
], dtype=torch.float32)   # (29, 3)


def _quat_xyzw_to_wxyz(q):
    """(...,4) xyzw → (...,4) wxyz."""
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


# ----------------------------------------------------------------------

#   ay_to_az = R_x(-90°) = [[1, 0,  0],
#                            [0, 0, -1],
#                            [0, 1,  0]]



# ----------------------------------------------------------------------
def ay_to_az_rotmat(device, dtype):
    return torch.tensor(
        [[1.0, 0.0,  0.0],
         [0.0, 0.0, -1.0],
         [0.0, 1.0,  0.0]],
        device=device, dtype=dtype,
    )


def apply_ay_to_az_on_vec(v):
    """v: (..., 3) y-up → z-up.  Applied via `v @ T.T`: (x, y, z)_yup → (x, z, -y)_zup."""
    T = ay_to_az_rotmat(v.device, v.dtype)
    return v @ T.T


def apply_ay_to_az_on_quat_wxyz(q_wxyz):
    """q_wxyz: (..., 4) y-up world. Left-multiply by ay_to_az rotmat."""
    R = quaternion_to_matrix(q_wxyz)
    T = ay_to_az_rotmat(q_wxyz.device, q_wxyz.dtype)
    R_new = T @ R
    return matrix_to_quaternion(R_new)


# ----------------------------------------------------------------------

# SMPL canonical forward = SMPL local +z (y-up world).


#   R_y(-90°) = [[ 0, 0, -1],
#                [ 0, 1,  0],
#                [ 1, 0,  0]]

# ----------------------------------------------------------------------
def _ry_neg90_rotmat(device, dtype):
    return torch.tensor(
        [[0.0, 0.0, -1.0],
         [0.0, 1.0,  0.0],
         [1.0, 0.0,  0.0]],
        device=device, dtype=dtype,
    )


def _apply_ry_neg90_on_vec(v):
    """v: (..., 3) y-up world → R_y(-90°) @ v.  (x, y, z) → (-z, y, x)."""
    R = _ry_neg90_rotmat(v.device, v.dtype)
    return v @ R.T


def _apply_ry_neg90_on_quat_wxyz(q_wxyz):
    """q_wxyz: (..., 4). Left-multiply world quaternion by R_y(-90°)."""
    R = quaternion_to_matrix(q_wxyz)
    Ry = _ry_neg90_rotmat(q_wxyz.device, q_wxyz.dtype)
    R_new = Ry @ R
    return matrix_to_quaternion(R_new)


def _apply_ry_neg90_on_axis_angle(axis_angle):
    """axis_angle: (..., 3). Left-multiply world rotation by R_y(-90°)."""
    R = axis_angle_to_matrix(axis_angle)
    Ry = _ry_neg90_rotmat(axis_angle.device, axis_angle.dtype)
    return matrix_to_axis_angle(Ry @ R)


def _byd_to_mjc(dof_byd):
    """Reorder 29 joint channels from BYD to MuJoCo order."""
    return dof_byd[..., _BYD_TO_MJC]


def _dof_to_pose_aa(dof_mjc, root_rot_wxyz, g5_axis_mjc):
    """Convert robot joint angles and root rotation to axis-angle poses."""
    leading = dof_mjc.shape[:-1]                                  # e.g. (B, F)
    dof_aa = g5_axis_mjc.to(dof_mjc.device, dof_mjc.dtype) * dof_mjc[..., None]  # (..., 29, 3)
    root_aa = matrix_to_axis_angle(quaternion_to_matrix(root_rot_wxyz))           # (..., 3)
    pose_aa = torch.cat([root_aa[..., None, :], dof_aa], dim=-2)  # (..., 30, 3)
    return pose_aa



_URDF_DFS_TO_BYD_BODY_CACHE = None
def _get_urdf_dfs_to_byd_body():
    global _URDF_DFS_TO_BYD_BODY_CACHE
    if _URDF_DFS_TO_BYD_BODY_CACHE is None:
        from hmr4d.model.offline.utils.endecoder import _G1_URDF_DFS_TO_BYD_BODY
        _URDF_DFS_TO_BYD_BODY_CACHE = list(_G1_URDF_DFS_TO_BYD_BODY)
    return _URDF_DFS_TO_BYD_BODY_CACHE


_HUMANOID_BATCH_CACHE = {}
def _get_humanoid_batch(device):
    """Return a cached forward-kinematics model for the requested device."""
    key = str(device)
    if key not in _HUMANOID_BATCH_CACHE:
        from motion_lib.torch_h1_humanoid_batch import Humanoid_Batch
        _HUMANOID_BATCH_CACHE[key] = Humanoid_Batch(
            extend_hand=False, extend_head=False,
            mjcf_file='unitree_description/mjcf/g1.xml',
            device=device,
        )
    return _HUMANOID_BATCH_CACHE[key]


def fk_g1_pth_slice(dof_byd, root_pos_z, root_rot_wxyz_z, device=None):
    """Run robot forward kinematics and return BYD-ordered body poses."""
    if device is None:
        device = dof_byd.device
    dof_byd        = dof_byd.to(device).float()
    root_pos_z     = root_pos_z.to(device).float()
    root_rot_wxyz_z = root_rot_wxyz_z.to(device).float()

    dof_mjc  = _byd_to_mjc(dof_byd)                              # (F, 29)
    pose_aa  = _dof_to_pose_aa(dof_mjc[None], root_rot_wxyz_z[None], _G5_AXIS_MJC)  # (1,F,30,3)
    from pytorch3d.transforms import axis_angle_to_quaternion as _aa2q
    pose_quat = _aa2q(pose_aa)                                    # (1,F,30,4) wxyz
    pose_mat  = quaternion_to_matrix(pose_quat)                   # (1,F,30,3,3)

    humanoid = _get_humanoid_batch(device)
    pos_dfs, rot_dfs = humanoid.forward_kinematics_batch(
        pose_mat[:, :, 1:], pose_mat[:, :, 0:1], root_pos_z[None]
    )
    
    perm = torch.tensor(_get_urdf_dfs_to_byd_body(), dtype=torch.long, device=device)
    body_pos_w_z  = pos_dfs[0][:, perm]                           # (F, 30, 3)
    body_quat_w_z = matrix_to_quaternion(rot_dfs[0])[:, perm]     # (F, 30, 4) wxyz
    return body_pos_w_z, body_quat_w_z


def _finite_diff_lin_vel(x, fps):
    """Compute finite-difference linear velocity and repeat the last frame."""
    if x.shape[0] < 2:
        return torch.zeros_like(x)
    diff = (x[1:] - x[:-1]) * float(fps)
    return torch.cat([diff, diff[-1:].clone()], dim=0)


def _finite_diff_ang_vel(q_wxyz, fps):
    """Compute world-frame angular velocity from wxyz quaternions."""
    if q_wxyz.shape[0] < 2:
        return torch.zeros(q_wxyz.shape[0], q_wxyz.shape[1], 3,
                           device=q_wxyz.device, dtype=q_wxyz.dtype)
    R = quaternion_to_matrix(q_wxyz)                              # (T, N, 3, 3)
    dR = R[1:] @ R[:-1].transpose(-1, -2)                         # (T-1, N, 3, 3)
    aa = matrix_to_axis_angle(dR)                                  # (T-1, N, 3)
    ang = aa * float(fps)
    return torch.cat([ang, ang[-1:].clone()], dim=0)


# ==========================================================
class G1AmassDualPthDataset(BaseDataset):
    """Load aligned human and robot motions from paired PTH mappings."""

    def __init__(
        self,
        amass_pth_path="motion.pth",
        g1_pth_path="g1.pth",
        amass_fps=30.0,
        motion_frames=120,
        l_factor=1.5,
        cam_augmentation="v11",
        limit_size=None,
        betas_dim=10,
        root_body_id=0,
        apply_az_to_ay_g1=True,
        
        
        g1_input_is_yup=True,
        
        skip_substrings=(),
        canonicalize_first_frame=True,
        align_target="pelvis",
        floor_adjust=True,
        
        
        yaw_only_canon=True,
        
        split="train",
        split_seed=42,
        split_ratios=(0.9, 0.05, 0.05),
        
        ratio_lo=(0.7, 0.7, -float("inf")),
        ratio_hi=(1.05, 1.05,  float("inf")),
        ratio_thr=0.1,
        smplx_neutral_npz_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
        
        
        
        
        full_sequence=False,
        
        
        
        
        
        amass_only=False,
        
        
        
        
        
        #       + subset_split="train" + subset_n=20.
        subset_split=None,
        subset_n=None,
        subset_seed=42,
    ):
        self.amass_pth_path = Path(amass_pth_path)
        self.g1_pth_path    = Path(g1_pth_path)
        self.amass_fps      = float(amass_fps)
        self.motion_frames  = motion_frames
        self.l_factor       = l_factor
        self.dataset_name   = "G1_AMASS_DUAL_PTH"
        self.betas_dim      = betas_dim
        self.root_body_id   = root_body_id
        self.apply_az_to_ay_g1 = apply_az_to_ay_g1
        self.g1_input_is_yup   = bool(g1_input_is_yup)
        self.skip_substrings = tuple(str(s) for s in (skip_substrings or ()))
        self.canonicalize_first_frame = canonicalize_first_frame
        self.floor_adjust   = floor_adjust
        self.yaw_only_canon = bool(yaw_only_canon)
        assert split in ("train", "val", "test"), split
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to one"
        self.split          = split
        self.split_seed     = int(split_seed)
        self.split_ratios   = tuple(float(r) for r in split_ratios)

        ratio_lo = tuple(float(v) for v in ratio_lo)
        ratio_hi = tuple(float(v) for v in ratio_hi)
        assert len(ratio_lo) == 3 and len(ratio_hi) == 3
        self.ratio_lo = ratio_lo
        self.ratio_hi = ratio_hi
        self.ratio_thr = float(ratio_thr)
        self._ratio_stats = {"accept": 0, "skip": 0}

        assert align_target in ("pelvis", "transl"), align_target
        self.align_target = align_target
        if canonicalize_first_frame and align_target == "pelvis":
            self._J_template_pelvis, self._J_shapedirs_pelvis = get_smplx_pelvis_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_pelvis = None
            self._J_shapedirs_pelvis = None
        if floor_adjust and canonicalize_first_frame and align_target == "pelvis":
            self._J_template_feet, self._J_shapedirs_feet = get_smplx_foot_buffers(
                smplx_neutral_npz_path, num_betas=betas_dim
            )
        else:
            self._J_template_feet = None
            self._J_shapedirs_feet = None

        
        self._skip_betas_aug = True

        
        self.full_sequence = bool(full_sequence) and split != "train"
        if bool(full_sequence) and split == "train":
            Log.warning(
                f"[{self.dataset_name}] full_sequence is ignored for the training split"
            )
        
        self.amass_only = bool(amass_only) and split != "train"
        if bool(amass_only) and split == "train":
            Log.warning(
                f"[{self.dataset_name}] amass_only is ignored for the training split"
            )
        
        if subset_split is not None or subset_n is not None:
            assert split != "train", "subset evaluation is only available for val/test"
            if subset_split is not None:
                assert subset_split in ("train", "val", "test"), f"bad subset_split={subset_split!r}"
        self.subset_split = subset_split
        self.subset_n     = int(subset_n) if subset_n is not None else None
        self.subset_seed  = int(subset_seed)

        super().__init__(cam_augmentation, limit_size)

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def _load_dataset(self):
        Log.info(f"[{self.dataset_name}] loading human motions from {self.amass_pth_path}")
        self.amass_files = _torch_load_cpu_mmap(self.amass_pth_path)
        if self.amass_only:
            Log.info(f"[{self.dataset_name}] robot ground truth disabled")
            self.g1_files = None
            common = sorted(self.amass_files.keys())
        else:
            Log.info(f"[{self.dataset_name}] loading robot motions from {self.g1_pth_path}")
            self.g1_files = _torch_load_cpu_mmap(self.g1_pth_path)
            common = sorted(set(self.amass_files.keys()) & set(self.g1_files.keys()))
        before = len(common)
        if self.skip_substrings:
            common = [k for k in common
                      if not any(s.lower() in k.lower() for s in self.skip_substrings)]
        Log.info(
            f"[{self.dataset_name}] common={len(common)} filtered={before - len(common)} "
            f"skip_substrings={self.skip_substrings}"
        )

        
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(common))
        n_total = len(common)
        n_train = int(self.split_ratios[0] * n_total)
        n_val   = int(self.split_ratios[1] * n_total)
        if self.split_ratios[2] <= 0:
            n_val = n_total - n_train
        if self.split == "train":
            idx_sel = perm[:n_train]
        elif self.split == "val":
            idx_sel = perm[n_train:n_train + n_val]
        else:
            idx_sel = perm[n_train + n_val:]
        self.split_keys = [common[i] for i in idx_sel]
        Log.info(
            f"[{self.dataset_name}] split={self.split} selected={len(self.split_keys)} "
            f"ratios={self.split_ratios} seed={self.split_seed}"
        )

        
        
        
        if self.subset_split is not None:
            src_idx = {
                "train": perm[:n_train],
                "val":   perm[n_train:n_train + n_val],
                "test":  perm[n_train + n_val:],
            }[self.subset_split]
            pool = [common[i] for i in src_idx]
            Log.info(f"[{self.dataset_name}] subset_split={self.subset_split}, pool size={len(pool)}")
            if self.subset_n is not None and self.subset_n < len(pool):
                rng2 = np.random.default_rng(self.subset_seed)
                pick = rng2.choice(len(pool), size=self.subset_n, replace=False)
                pick.sort()   
                self.split_keys = [pool[i] for i in pick]
            else:
                self.split_keys = pool
            Log.info(f"[{self.dataset_name}] subset selected={len(self.split_keys)}")

    def _get_idx2meta(self):
        self.idx2meta = []
        skipped = 0
        for k in self.split_keys:
            length = int(self.amass_files[k]["pose"].shape[0])
            if not self.amass_only:
                
                g1_len = int(self.g1_files[k]["root_pos"].shape[0])
                if g1_len != length:
                    Log.warning(
                        f"[{self.dataset_name}] length mismatch for {k}: "
                        f"human={length}, robot={g1_len}"
                    )
                    skipped += 1
                    continue
            if length < 25:
                skipped += 1
                continue
            if self.full_sequence:
                
                self.idx2meta.append({
                    "key": k, "usable_len": length, "name": Path(k).stem,
                    "seg_id": 0, "num_seg": 1,
                })
            else:
                num_samples = max(length // self.motion_frames, 1)
                
                
                for seg_id in range(num_samples):
                    self.idx2meta.append({
                        "key": k, "usable_len": length, "name": Path(k).stem,
                        "seg_id": seg_id, "num_seg": num_samples,
                    })
        Log.info(
            f"[{self.dataset_name}] samples={len(self.idx2meta)} skipped={skipped} "
            f"full_sequence={self.full_sequence}"
        )

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        usable_len = meta["usable_len"]
        tgt_len    = self.motion_frames

        if self.split == "train":
            
            lo = max(2, int(float(tgt_len) / self.l_factor))
            hi = max(lo + 1, int(float(tgt_len) * self.l_factor) + 1)
            raw_subset_len = np.random.randint(lo, hi)
            if raw_subset_len <= usable_len:
                start = np.random.randint(0, usable_len - raw_subset_len + 1)
                end   = start + raw_subset_len
            else:
                start, end = 0, usable_len
        else:
            
            
            if self.full_sequence:
                
                
                start = 0
                end = usable_len
                tgt_len = usable_len
            else:
                seg_id = int(meta.get("seg_id", 0))
                raw_subset_len = min(tgt_len, usable_len)
                
                
                if meta.get("start", None) is not None:
                    start = int(meta["start"])
                else:
                    start = seg_id * tgt_len
                if start + raw_subset_len > usable_len:
                    start = max(0, usable_len - raw_subset_len)
                end = start + raw_subset_len

        
        ea = self.amass_files[meta["key"]]
        pose_full = ea["pose"][start:end].float()                  # (F, 66)
        trans     = ea["trans"][start:end].float()                 # (F, 3)
        beta      = ea["beta"].float()                             # (10,)
        actual_len = pose_full.shape[0]
        smpl_data = {
            "transl":        trans,
            "global_orient": pose_full[:, :3],
            "body_pose":     pose_full[:, 3:],
            "betas":         beta[: self.betas_dim].unsqueeze(0).expand(actual_len, -1).float(),
        }

        
        if self.amass_only:
            smpl_interpolated = interpolate_smpl_params(smpl_data, tgt_len)
            if self.canonicalize_first_frame:
                if self.align_target == "pelvis":
                    pelvis_offset = compute_smpl_pelvis_offset(
                        smpl_interpolated["betas"][0], self._J_template_pelvis, self._J_shapedirs_pelvis
                    )
                else:
                    pelvis_offset = None
                smpl_interpolated["global_orient"], smpl_interpolated["transl"] = canonicalize_smpl_first_frame(
                    smpl_interpolated["global_orient"], smpl_interpolated["transl"], pelvis_offset=pelvis_offset,
                    yaw_only=self.yaw_only_canon, up_axis=1, fwd_axis=2,   # SMPL y-up, forward=+z
                )
            if self.floor_adjust and self.canonicalize_first_frame:
                with torch.no_grad():
                    j24 = self.smplx_lite(
                        smpl_interpolated["body_pose"][:1],
                        smpl_interpolated["betas"][:1],
                        smpl_interpolated["global_orient"][:1],
                        smpl_interpolated["transl"][:1],
                    )[0]   # (24, 3) world y-up
                smpl_floor_y = float(-j24[:, 1].min())
                smpl_interpolated["transl"][:, 1] += smpl_floor_y
            
            return {
                "data_name":     "g1_paired_dualpth",
                "body_pose":     smpl_interpolated["body_pose"],
                "betas":         smpl_interpolated["betas"],
                "global_orient": smpl_interpolated["global_orient"],
                "transl":        smpl_interpolated["transl"],
            }

        
        eg = self.g1_files[meta["key"]]
        # dof_pos: MJC → BYD; root_rot: xyzw → wxyz
        dof_pos_mjc = torch.as_tensor(eg["dof_pos"][start:end], dtype=torch.float32)  # (F, 29)
        dof_pos_byd = dof_pos_mjc[..., _MJC_TO_BYD]                                   # (F, 29) BYD
        root_pos_raw     = torch.as_tensor(eg["root_pos"][start:end], dtype=torch.float32) # (F, 3)
        root_rot_xyzw_raw = torch.as_tensor(eg["root_rot"][start:end], dtype=torch.float32)# (F, 4) xyzw
        root_rot_wxyz_raw = _quat_xyzw_to_wxyz(root_rot_xyzw_raw)                          # (F, 4) wxyz

        if self.g1_input_is_yup:
            
            
            root_pos_z      = apply_ay_to_az_on_vec(root_pos_raw)
            root_rot_wxyz_z = apply_ay_to_az_on_quat_wxyz(root_rot_wxyz_raw)
        else:
            
            root_pos_z      = root_pos_raw
            root_rot_wxyz_z = root_rot_wxyz_raw

        
        smpl_interpolated = interpolate_smpl_params(smpl_data, tgt_len)

        
        dof_pos_byd_i = interpolate_seq(dof_pos_byd, tgt_len)            # (tgt_len, 29)
        root_pos_z_i  = interpolate_seq(root_pos_z,  tgt_len)            # (tgt_len, 3)
        root_rot_wxyz_z_i = slerp_sequence_quat(root_rot_wxyz_z, tgt_len)# (tgt_len, 4) wxyz, z-up

        
        with torch.no_grad():
            body_pos_w_z, body_quat_w_z = fk_g1_pth_slice(
                dof_pos_byd_i, root_pos_z_i, root_rot_wxyz_z_i, device=dof_pos_byd_i.device
            )

        
        fps_out = float(self.amass_fps)
        joint_vel_byd_i   = _finite_diff_lin_vel(dof_pos_byd_i, fps_out)             # (tgt_len, 29)
        body_lin_vel_w_z  = _finite_diff_lin_vel(body_pos_w_z, fps_out)              # (tgt_len, 30, 3)
        body_ang_vel_w_z  = _finite_diff_ang_vel(body_quat_w_z, fps_out)             # (tgt_len, 30, 3)

        g1_interpolated = {
            "fps":            torch.tensor(fps_out, dtype=torch.float32),
            "joint_pos":      dof_pos_byd_i,
            "joint_vel":      joint_vel_byd_i,
            "body_pos_w":     body_pos_w_z,
            "body_quat_w":    body_quat_w_z,
            "body_lin_vel_w": body_lin_vel_w_z,
            "body_ang_vel_w": body_ang_vel_w_z,
        }

        

        
        if self.canonicalize_first_frame:
            g1_interpolated = canonicalize_g1_first_frame(
                g1_interpolated, root_body_id=self.root_body_id,
                yaw_only=self.yaw_only_canon, up_axis=2, fwd_axis=0,   # G1 z-up, URDF forward=+x
            )

        
        if self.apply_az_to_ay_g1:
            g1_interpolated["body_pos_w"]     = apply_az_to_ay_on_vec(g1_interpolated["body_pos_w"])
            g1_interpolated["body_lin_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_lin_vel_w"])
            g1_interpolated["body_ang_vel_w"] = apply_az_to_ay_on_vec(g1_interpolated["body_ang_vel_w"])
            g1_interpolated["body_quat_w"]    = apply_az_to_ay_on_quat_wxyz(g1_interpolated["body_quat_w"])

            
            
            
            g1_interpolated["body_pos_w"]     = _apply_ry_neg90_on_vec(g1_interpolated["body_pos_w"])
            g1_interpolated["body_lin_vel_w"] = _apply_ry_neg90_on_vec(g1_interpolated["body_lin_vel_w"])
            g1_interpolated["body_ang_vel_w"] = _apply_ry_neg90_on_vec(g1_interpolated["body_ang_vel_w"])
            g1_interpolated["body_quat_w"]    = _apply_ry_neg90_on_quat_wxyz(g1_interpolated["body_quat_w"])

        
        if self.canonicalize_first_frame:
            if self.align_target == "pelvis":
                pelvis_offset = compute_smpl_pelvis_offset(
                    smpl_interpolated["betas"][0], self._J_template_pelvis, self._J_shapedirs_pelvis
                )
            else:
                pelvis_offset = None
            smpl_interpolated["global_orient"], smpl_interpolated["transl"] = canonicalize_smpl_first_frame(
                smpl_interpolated["global_orient"], smpl_interpolated["transl"], pelvis_offset=pelvis_offset,
                yaw_only=self.yaw_only_canon, up_axis=1, fwd_axis=2,   # SMPL y-up, forward=+z
            )
        else:
            g1_root_first = g1_interpolated["body_pos_w"][0, self.root_body_id].clone()
            smpl_transl_first = smpl_interpolated["transl"][0].clone()
            g1_world_offset = smpl_transl_first - g1_root_first
            g1_interpolated["body_pos_w"] = g1_interpolated["body_pos_w"] + g1_world_offset[None, None, :]

        
        if self.floor_adjust and self.canonicalize_first_frame:
            g1_floor_y = float(-g1_interpolated["body_pos_w"][0, :, 1].min())
            g1_interpolated["body_pos_w"][:, :, 1] += g1_floor_y
            with torch.no_grad():
                j24 = self.smplx_lite(
                    smpl_interpolated["body_pose"][:1],
                    smpl_interpolated["betas"][:1],
                    smpl_interpolated["global_orient"][:1],
                    smpl_interpolated["transl"][:1],
                )[0]   # (24, 3) world y-up
            smpl_floor_y = float(-j24[:, 1].min())
            smpl_interpolated["transl"][:, 1] += smpl_floor_y

        return {
            "data_name":         "g1_paired_dualpth",
            "body_pose":         smpl_interpolated["body_pose"],
            "betas":             smpl_interpolated["betas"],
            "global_orient":     smpl_interpolated["global_orient"],
            "transl":            smpl_interpolated["transl"],
            "g1_fps":            g1_interpolated["fps"],
            "g1_joint_pos":      g1_interpolated["joint_pos"],
            "g1_joint_vel":      g1_interpolated["joint_vel"],
            "g1_body_pos_w":     g1_interpolated["body_pos_w"],
            "g1_body_quat_w":    g1_interpolated["body_quat_w"],
            "g1_body_lin_vel_w": g1_interpolated["body_lin_vel_w"],
            "g1_body_ang_vel_w": g1_interpolated["body_ang_vel_w"],
        }

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def _process_data(self, data, idx):
        
        
        if self.amass_only:
            seed = (self.split_seed * 1_000_003 + int(idx)) % (2 ** 32)
            self._static_cam_seed = int(seed)
            np_state    = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            np.random.seed(seed)
            torch.manual_seed(seed)
            try:
                return_data = super()._process_data(data, idx)
            finally:
                np.random.set_state(np_state)
                torch.random.set_rng_state(torch_state)
            return return_data

        sample_fps = torch.as_tensor(data.pop("g1_fps"), dtype=torch.float32).reshape(())
        g1_target = {
            "g1_joint_pos":      data.pop("g1_joint_pos"),
            "g1_joint_vel":      data.pop("g1_joint_vel"),
            "g1_body_pos_w":     data.pop("g1_body_pos_w"),
            "g1_body_quat_w":    data.pop("g1_body_quat_w"),
            "g1_body_lin_vel_w": data.pop("g1_body_lin_vel_w"),
            "g1_body_ang_vel_w": data.pop("g1_body_ang_vel_w"),
        }
        if self.split == "train":
            return_data = super()._process_data(data, idx)
        else:
            
            
            seed = (self.split_seed * 1_000_003 + int(idx)) % (2 ** 32)
            
            
            self._static_cam_seed = int(seed)
            np_state    = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            np.random.seed(seed)
            torch.manual_seed(seed)
            try:
                return_data = super()._process_data(data, idx)
            finally:
                np.random.set_state(np_state)
                torch.random.set_rng_state(torch_state)

        
        g1_T_w2c = return_data["T_w2c"].clone()
        g1_T_w2c[..., :3, 3] = 0.9 * return_data["T_w2c"][..., :3, 3]
        g1_target["g1_T_w2c"] = g1_T_w2c
        return_data["g1_target"] = g1_target
        return_data["fps"] = sample_fps

        
        max_len = return_data["T_w2c"].shape[0]
        for k in list(g1_target.keys()):
            v = g1_target[k]
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] != max_len:
                g1_target[k] = repeat_to_max_len(v, max_len)
        return return_data

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def _root_axis_ratios(self, data):
        transl = data["transl"]
        g1_pos = data["g1_body_pos_w"][:, self.root_body_id]
        a_delta = (transl - transl[0:1]).detach().cpu().numpy()
        g_delta = (g1_pos - g1_pos[0:1]).detach().cpu().numpy()
        out = []
        for ax in range(3):
            a = a_delta[:, ax]; g = g_delta[:, ax]
            if float(np.abs(a).max()) <= self.ratio_thr:
                out.append(None); continue
            mask = np.abs(a) > self.ratio_thr
            if int(mask.sum()) < 2:
                out.append(None); continue
            am, gm = a[mask], g[mask]
            denom = float((am * am).sum())
            if denom < 1e-12:
                out.append(None); continue
            out.append(float((gm * am).sum() / denom))
        return out

    def _ratio_pass(self, data):
        ks = self._root_axis_ratios(data)
        for ax, k in enumerate(ks):
            if k is None: continue
            if not (self.ratio_lo[ax] <= k <= self.ratio_hi[ax]):
                return False, ks
        return True, ks

    def __getitem__(self, idx):
        
        
        
        if self.split != "train":
            return self._process_data(self._load_data(idx), idx)

        start_idx = idx
        max_skip  = len(self.idx2meta)
        skip_n    = 0
        last_data = None
        last_idx  = idx
        while skip_n < max_skip:
            data = self._load_data(idx)
            last_data = data
            last_idx  = idx
            ok, _ks = self._ratio_pass(data)
            if ok:
                self._ratio_stats["accept"] += 1
                return self._process_data(data, idx)
            self._ratio_stats["skip"] += 1
            skip_n += 1
            idx = (idx + 1) % len(self.idx2meta)
            if idx == start_idx:
                break
        Log.warning(
            f"[{self.dataset_name}] ratio filter rejected every candidate from "
            f"start_idx={start_idx}; using idx={last_idx}"
        )
        return self._process_data(last_data, last_idx)


BONES_SEED_SOMA_OBS14 = [6, 7, 9, 15, 10, 16, 11, 17, 20, 24, 21, 25, 22, 26]


def _load_joblib(path):
    return joblib.load(path)


def _as_float_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32).clone()


class BonesSeedG1SomaDataset(Dataset):
    """Prepared Bones Seed paired SOMA/G1 dataset.

    The exported ``dof_pos`` and ``body_*_full`` arrays use BYD joint/body
    order, while quaternions use wxyz. This class loads every top-level
    ``*.pkl`` pair as-is and does not apply another motion selection pass.
    """

    def __init__(
        self,
        data_dir="/path/to/humanoid_data/bones_seed_g1_soma_pair",
        index_path=None,
        motion_frames=120,
        l_factor=1.5,
        cam_augmentation="v11",
        limit_size=None,
        root_body_id=0,
        canonicalize_first_frame=True,
        yaw_only_canon=True,
        floor_adjust=True,
        split="train",
        split_seed=42,
        split_ratios=(0.9, 0.05, 0.05),
        dof_pos_order="byd",
        soma_obs_indices=tuple(BONES_SEED_SOMA_OBS14),
        full_sequence=False,
        subset_split=None,
        subset_n=None,
        subset_seed=42,
    ):
        assert split in ("train", "val", "test"), split
        if dof_pos_order != "byd":
            raise ValueError(
                "BonesSeedG1SomaDataset requires dof_pos_order='byd': the "
                "Bones Seed exporter already writes all 29 dof_pos channels "
                "in BYD order. Treating them as MJC silently permutes the labels."
            )
        assert abs(sum(split_ratios) - 1.0) < 1e-6
        self.data_dir = Path(data_dir)
        # Persist lightweight sequence metadata next to the staged local data.
        # A one-process prewarm creates it before torchrun, so eight DDP ranks
        # do not each decompress and scan all ~33 GB Bones Seed pkl files.
        self.index_path = Path(index_path) if index_path else self.data_dir / "bones_seed_index_v1.json"
        self.motion_frames = int(motion_frames)
        self.l_factor = float(l_factor)
        self.cam_augmentation = cam_augmentation
        self.limit_size = limit_size
        self.root_body_id = int(root_body_id)
        self.canonicalize_first_frame = bool(canonicalize_first_frame)
        self.yaw_only_canon = bool(yaw_only_canon)
        self.floor_adjust = bool(floor_adjust)
        self.split = split
        self.split_seed = int(split_seed)
        self.split_ratios = tuple(float(x) for x in split_ratios)
        self.dof_pos_order = dof_pos_order
        self.soma_obs_indices = tuple(int(i) for i in soma_obs_indices)
        self.full_sequence = bool(full_sequence) and split != "train"
        self.subset_split = subset_split
        self.subset_n = int(subset_n) if subset_n is not None else None
        self.subset_seed = int(subset_seed)
        self.dataset_name = "BONES_SEED_G1_SOMA"
        self._cached_pkl_path = None
        self._cached_pkl_data = None

        self._load_dataset()
        self._get_idx2meta()

    def __len__(self):
        if self.limit_size is not None:
            return min(int(self.limit_size), len(self.idx2meta))
        return len(self.idx2meta)

    def _load_dataset(self):
        pkl_files = sorted(self.data_dir.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found under {self.data_dir}")

        fingerprint = [
            {"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in pkl_files
        ]
        cached_records = None
        if self.index_path is not None and self.index_path.is_file():
            with self.index_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("version") == 1 and cached.get("fingerprint") == fingerprint:
                cached_records = cached.get("records")
            else:
                Log.warning(f"[{self.dataset_name}] stale index, rebuilding: {self.index_path}")

        if cached_records is None:
            cached_records = []
            for p in pkl_files:
                data = _load_joblib(p)
                if not isinstance(data, dict):
                    raise TypeError(f"Bones Seed file must contain a dict: {p}")
                for seq_name, payload in data.items():
                    key = f"{p.stem}/{seq_name}"
                    g1 = payload["g1"]
                    soma = payload["soma"]
                    required_g1 = (
                        "root_pos", "root_rot", "dof_pos",
                        "body_pos_w_full", "body_quat_w_full",
                    )
                    required_soma = ("global_orient", "joints")
                    lengths = [int(np.asarray(g1[name]).shape[0]) for name in required_g1]
                    lengths += [int(np.asarray(soma[name]).shape[0]) for name in required_soma]
                    if len(set(lengths)) != 1:
                        raise ValueError(f"Unpaired Bones Seed lengths for {key}: {lengths}")
                    if np.asarray(g1["dof_pos"]).shape[-1] != 29:
                        raise ValueError(f"Expected 29 BYD joints for {key}")
                    if np.asarray(g1["body_pos_w_full"]).shape[-2:] != (30, 3):
                        raise ValueError(f"Expected 30 G1 bodies for {key}")
                    if np.asarray(soma["joints"]).shape[-2:] != (28, 3):
                        raise ValueError(f"Expected 28 SOMA joints for {key}")
                    cached_records.append(
                        {"key": key, "file_name": p.name, "seq_name": seq_name, "length": lengths[0]}
                    )
                del data
            if self.index_path is not None:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self.index_path.with_name(f"{self.index_path.name}.tmp.{os.getpid()}")
                with tmp_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {"version": 1, "fingerprint": fingerprint, "records": cached_records},
                        f,
                        ensure_ascii=False,
                    )
                os.replace(tmp_path, self.index_path)
                Log.info(f"[{self.dataset_name}] wrote index: {self.index_path}")

        self.records = {}
        for rec in cached_records:
            key = rec["key"]
            if key in self.records:
                raise ValueError(f"Duplicate Bones Seed key: {key}")
            self.records[key] = {
                "file": str(self.data_dir / rec["file_name"]),
                "seq_name": rec["seq_name"],
                "length": int(rec["length"]),
            }

        # Keep mirrored variants (suffix _M) in the same split to prevent
        # train/validation leakage.  Membership is random; final key order is
        # sorted by pkl so sequential loading can reuse the per-worker cache.
        groups = {}
        for key in sorted(self.records):
            file_stem, seq_name = key.split("/", 1)
            group_name = seq_name[:-2] if seq_name.endswith("_M") else seq_name
            groups.setdefault(f"{file_stem}/{group_name}", []).append(key)
        group_names = sorted(groups)
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(group_names))
        n_total = len(group_names)
        n_train = int(self.split_ratios[0] * n_total)
        n_val = int(self.split_ratios[1] * n_total)
        splits = {
            "train": perm[:n_train],
            "val": perm[n_train:n_train + n_val],
            "test": perm[n_train + n_val:],
        }
        split_for_keys = self.subset_split or self.split
        assert split_for_keys in ("train", "val", "test")
        selected_groups = [group_names[i] for i in splits[split_for_keys]]
        selected = sorted(key for group_name in selected_groups for key in groups[group_name])
        if self.subset_n is not None and self.subset_n < len(selected):
            rng2 = np.random.default_rng(self.subset_seed)
            pick = rng2.choice(len(selected), size=self.subset_n, replace=False)
            pick.sort()
            selected = [selected[i] for i in pick]
        self.split_keys = selected
        Log.info(
            f"[{self.dataset_name}] files={len(pkl_files)} "
            f"groups={len(group_names)} seqs={len(self.records)} "
            f"split={self.split} selected={len(self.split_keys)}; runtime_filter=off"
        )

    def _seq_len(self, key):
        return int(self.records[key]["length"])

    def _load_record_payload(self, key):
        rec = self.records[key]
        pkl_path = rec["file"]
        if self._cached_pkl_path != pkl_path:
            # Release the old decompressed session before evaluating the next
            # joblib.load; otherwise both large dictionaries coexist briefly.
            self._cached_pkl_data = None
            self._cached_pkl_path = None
            gc.collect()
            self._cached_pkl_data = _load_joblib(pkl_path)
            self._cached_pkl_path = pkl_path
        return self._cached_pkl_data[rec["seq_name"]]

    def _get_idx2meta(self):
        self.idx2meta = []
        for key in self.split_keys:
            length = self._seq_len(key)
            if length < 25:
                continue
            if self.full_sequence:
                self.idx2meta.append({"key": key, "usable_len": length, "seg_id": 0, "num_seg": 1})
            else:
                n_seg = max(length // self.motion_frames, 1)
                for seg_id in range(n_seg):
                    self.idx2meta.append({"key": key, "usable_len": length, "seg_id": seg_id, "num_seg": n_seg})
        Log.info(f"[{self.dataset_name}] idx2meta={len(self.idx2meta)} full_sequence={self.full_sequence}")

    def _slice_bounds(self, idx):
        meta = self.idx2meta[idx]
        usable_len = meta["usable_len"]
        tgt_len = self.motion_frames
        if self.split == "train":
            lo = max(2, int(tgt_len / self.l_factor))
            hi = max(lo + 1, int(tgt_len * self.l_factor) + 1)
            raw_len = np.random.randint(lo, hi)
            if raw_len <= usable_len:
                start = np.random.randint(0, usable_len - raw_len + 1)
                end = start + raw_len
            else:
                start, end = 0, usable_len
        else:
            if self.full_sequence:
                start, end, tgt_len = 0, usable_len, usable_len
            else:
                raw_len = min(tgt_len, usable_len)
                start = int(meta.get("seg_id", 0)) * tgt_len
                if start + raw_len > usable_len:
                    start = max(0, usable_len - raw_len)
                end = start + raw_len
        return start, end, tgt_len

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        rec = self._load_record_payload(meta["key"])
        g1_raw = rec["g1"]
        soma_raw = rec["soma"]
        start, end, tgt_len = self._slice_bounds(idx)

        fps = float(g1_raw.get("fps", soma_raw.get("fps", 30.0)))
        dof_pos = _as_float_tensor(g1_raw["dof_pos"][start:end])
        # Exported Bones Seed dof_pos is already in the repository-wide BYD
        # order. Do not apply the MJC->BYD permutation used by BEDLAM/H36M.
        root_pos = _as_float_tensor(g1_raw["root_pos"][start:end])
        root_rot = _as_float_tensor(g1_raw["root_rot"][start:end])
        body_pos = _as_float_tensor(g1_raw["body_pos_w_full"][start:end])
        body_quat = _as_float_tensor(g1_raw["body_quat_w_full"][start:end])

        soma_go = _as_float_tensor(soma_raw["global_orient"][start:end])
        soma_joints = _as_float_tensor(soma_raw["joints"][start:end])

        dof_pos = interpolate_seq(dof_pos, tgt_len)
        root_pos = interpolate_seq(root_pos, tgt_len)
        root_rot = slerp_sequence_quat(root_rot, tgt_len)
        body_pos = interpolate_seq(body_pos, tgt_len)
        body_quat = slerp_sequence_quat(body_quat, tgt_len)
        soma_go_q = slerp_sequence_quat(matrix_to_quaternion(axis_angle_to_matrix(soma_go)), tgt_len)
        soma_go = matrix_to_axis_angle(quaternion_to_matrix(soma_go_q))
        soma_joints = interpolate_seq(soma_joints, tgt_len)

        g1 = {
            "fps": torch.tensor(fps, dtype=torch.float32),
            "joint_pos": dof_pos,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "body_pos_w_full": body_pos,
            "body_quat_w_full": body_quat,
        }
        if self.canonicalize_first_frame:
            g1 = canonicalize_g1_first_frame_bones_seed(
                g1,
                root_body_id=self.root_body_id,
                yaw_only=self.yaw_only_canon,
                up_axis=2,
                fwd_axis=0,
            )
            soma_go, soma_joints = canonicalize_smpl_first_frame_bones_seed(
                soma_go,
                soma_joints,
                yaw_only=self.yaw_only_canon,
                up_axis=2,
                fwd_axis=1,
            )
        else:
            g1["body_pos_w"] = body_pos
            g1["body_quat_w"] = body_quat

        soma_go = apply_az_to_ay_on_axis_angle(soma_go)
        soma_joints = apply_az_to_ay_on_vec(soma_joints)

        g1["body_pos_w"] = apply_az_to_ay_on_vec(g1["body_pos_w"])
        g1["body_quat_w"] = apply_az_to_ay_on_quat_wxyz(g1["body_quat_w"])

        # Both BonesSeed first-frame canonicalizers operate in the source
        # z-up frame and map their selected forward axis to +x.  Rotate the
        # human and robot together after z-up -> y-up conversion so both use
        # the training convention forward=+z.  Applying this only to G1
        # creates the observed exact 90-degree human/robot heading mismatch.
        soma_go = _apply_ry_neg90_on_axis_angle(soma_go)
        soma_joints = _apply_ry_neg90_on_vec(soma_joints)
        g1["body_pos_w"] = _apply_ry_neg90_on_vec(g1["body_pos_w"])
        g1["body_quat_w"] = _apply_ry_neg90_on_quat_wxyz(g1["body_quat_w"])
        g1["root_pos"] = g1["body_pos_w"][:, self.root_body_id].clone()
        g1["root_rot"] = g1["body_quat_w"][:, self.root_body_id].clone()

        if self.floor_adjust:
            floor_y = float(min(g1["body_pos_w"][0, :, 1].min(), soma_joints[0, :, 1].min()))
            g1["body_pos_w"][:, :, 1] -= floor_y
            soma_joints[:, :, 1] -= floor_y
            g1["root_pos"] = g1["body_pos_w"][:, self.root_body_id].clone()

        g1["joint_vel"] = _finite_diff_lin_vel(g1["joint_pos"], fps)
        g1["body_lin_vel_w"] = _finite_diff_lin_vel(g1["body_pos_w"], fps)
        g1["body_ang_vel_w"] = _finite_diff_ang_vel(g1["body_quat_w"], fps)

        return {
            "data_name": "bones_seed_g1_soma",
            "key": meta["key"],
            "fps": fps,
            "soma_global_orient": soma_go,
            "soma_joints_w": soma_joints,
            "soma_obs_joints_w": soma_joints[:, self.soma_obs_indices],
            "g1_joint_pos": g1["joint_pos"],
            "g1_joint_vel": g1["joint_vel"],
            "g1_body_pos_w": g1["body_pos_w"],
            "g1_body_quat_w": g1["body_quat_w"],
            "g1_body_lin_vel_w": g1["body_lin_vel_w"],
            "g1_body_ang_vel_w": g1["body_ang_vel_w"],
        }

    def _make_camera(self, soma_joints_w, idx):
        length = soma_joints_w.shape[0]
        _, _, K_fullimg = create_camera_sensor(1000, 1000, 43.3)
        if self.cam_augmentation == "v11":
            augmentor = CameraAugmentorV11()
            return augmentor(soma_joints_w, length), K_fullimg
        if self.cam_augmentation == "static_v11":
            seed = (self.split_seed * 1_000_003 + int(idx)) % (2 ** 32)
            augmentor = StaticCameraV11(seed=seed)
            return augmentor(soma_joints_w, length), K_fullimg
        raise NotImplementedError(f"Unknown cam_augmentation={self.cam_augmentation!r}")

    def _process_data(self, data, idx):
        length = int(data["soma_joints_w"].shape[0])
        T_w2c, K = self._make_camera(data["soma_joints_w"], idx)
        K_fullimg = K.repeat(length, 1, 1)
        R_c2gv = get_R_c2gv(T_w2c[:, :3, :3], torch.tensor([0, -1, 0], dtype=torch.float32))
        cam_angvel = compute_cam_angvel(T_w2c[:, :3, :3])

        g1_target = {
            "g1_joint_pos": data["g1_joint_pos"],
            "g1_joint_vel": data["g1_joint_vel"],
            "g1_body_pos_w": data["g1_body_pos_w"],
            "g1_body_quat_w": data["g1_body_quat_w"],
            "g1_body_lin_vel_w": data["g1_body_lin_vel_w"],
            "g1_body_ang_vel_w": data["g1_body_ang_vel_w"],
        }
        g1_T_w2c = T_w2c.clone()
        g1_T_w2c[..., :3, 3] = 0.9 * T_w2c[..., :3, 3]
        g1_target["g1_T_w2c"] = g1_T_w2c

        zeros_body = torch.zeros(length, 63, dtype=torch.float32)
        zeros_betas = torch.zeros(length, 10, dtype=torch.float32)
        smpl_params = {
            "body_pose": zeros_body,
            "betas": zeros_betas,
            "global_orient": data["soma_global_orient"],
            "transl": data["soma_joints_w"][:, 0],
        }

        out = {
            "meta": {"data_name": data["data_name"], "idx": idx, "key": data["key"], "T_w2c": T_w2c},
            "length": length,
            "fps": torch.tensor(float(data["fps"]), dtype=torch.float32),
            "smpl_params_c": smpl_params,
            "smpl_params_w": smpl_params,
            "T_w2c": T_w2c,
            "R_c2gv": R_c2gv,
            "gravity_vec": torch.tensor([0, -1, 0], dtype=torch.float32),
            "bbx_xys": torch.zeros((length, 3), dtype=torch.float32),
            "K_fullimg": K_fullimg,
            "f_imgseq": torch.zeros((length, 1024), dtype=torch.float32),
            "kp2d": torch.zeros(length, 17, 3, dtype=torch.float32),
            "cam_angvel": cam_angvel,
            "soma_joints_w": data["soma_joints_w"],
            "soma_obs_joints_w": data["soma_obs_joints_w"],
            "g1_target": g1_target,
            "mask": {
                "valid": get_valid_mask(length, length),
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
            },
        }

        max_len = length
        out["smpl_params_c"] = repeat_to_max_len_dict(out["smpl_params_c"], max_len)
        out["smpl_params_w"] = repeat_to_max_len_dict(out["smpl_params_w"], max_len)
        for k in ("T_w2c", "R_c2gv", "K_fullimg", "cam_angvel", "soma_joints_w", "soma_obs_joints_w"):
            out[k] = repeat_to_max_len(out[k], max_len)
        for k, v in list(g1_target.items()):
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                g1_target[k] = repeat_to_max_len(v, max_len)
        return out

    def __getitem__(self, idx):
        return self._process_data(self._load_data(idx), idx)


# ==========================================================

# ==========================================================




