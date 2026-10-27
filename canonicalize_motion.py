import torch
import torch.nn.functional as F
import numpy as np

from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)

def az_to_ay_rotmat(device, dtype):
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


def ay_to_az_rotmat(device, dtype):
    return torch.tensor(
        [[1.0, 0.0,  0.0],
         [0.0, 0.0, -1.0],
         [0.0, 1.0,  0.0]],
        device=device, dtype=dtype,
    )


def apply_ay_to_az_on_vec(v):
    """v: (..., 3) y-up → z-up."""
    T = ay_to_az_rotmat(v.device, v.dtype)
    return v @ T.T


def apply_ay_to_az_on_quat_wxyz(q_wxyz):
    """q_wxyz: (..., 4) y-up world. Left-multiply by ay_to_az rotmat."""
    R = quaternion_to_matrix(q_wxyz)
    T = ay_to_az_rotmat(q_wxyz.device, q_wxyz.dtype)
    R_new = T @ R
    return matrix_to_quaternion(R_new)


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


R_SMPL_LOCAL_TO_G1_LOCAL = torch.tensor(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)


def apply_smpl_to_g1_axes_on_smpl(global_orient_aa, transl, pelvis_offset_smpl=None):
    
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


_SMPLX_PELVIS_CACHE = {}


def get_smplx_pelvis_buffers(
    smplx_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    num_betas=10,
):
    'Documentation removed.'
    key = (str(smplx_path), int(num_betas))
    if key in _SMPLX_PELVIS_CACHE:
        return _SMPLX_PELVIS_CACHE[key]

    data = np.load(smplx_path, allow_pickle=True)
    v_template  = torch.tensor(np.asarray(data["v_template"]), dtype=torch.float32)                   # (V, 3)
    shapedirs   = torch.tensor(np.asarray(data["shapedirs"][:, :, :num_betas]), dtype=torch.float32)  # (V, 3, B)
    J_regressor = torch.tensor(np.asarray(data["J_regressor"]), dtype=torch.float32)                  # (J, V)

    pelvis_row = J_regressor[0]                                              # (V,)
    J_template_pelvis  = pelvis_row @ v_template                             # (3,)
    J_shapedirs_pelvis = torch.einsum("v, vcb -> cb", pelvis_row, shapedirs) # (3, B)

    _SMPLX_PELVIS_CACHE[key] = (J_template_pelvis, J_shapedirs_pelvis)
    return _SMPLX_PELVIS_CACHE[key]


def compute_smpl_pelvis_offset(betas, J_template_pelvis, J_shapedirs_pelvis):
    'Documentation removed.'
    return J_template_pelvis + torch.einsum("...b, cb -> ...c", betas, J_shapedirs_pelvis)


_SMPLX_FOOT_JOINT_IDS = (7, 8, 10, 11)
_SMPLX_FOOT_CACHE: dict = {}


def get_smplx_foot_buffers(
    smplx_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    num_betas=10,
):
    key = (str(smplx_path), int(num_betas))
    if key in _SMPLX_FOOT_CACHE:
        return _SMPLX_FOOT_CACHE[key]
    data = np.load(smplx_path, allow_pickle=True)
    v_template  = torch.tensor(np.asarray(data["v_template"]), dtype=torch.float32)
    shapedirs   = torch.tensor(np.asarray(data["shapedirs"][:, :, :num_betas]), dtype=torch.float32)
    J_regressor = torch.tensor(np.asarray(data["J_regressor"]), dtype=torch.float32)
    rows = J_regressor[list(_SMPLX_FOOT_JOINT_IDS)]                # (4, V)
    J_template_feet  = rows @ v_template                             # (4, 3)
    J_shapedirs_feet = torch.einsum("jv,vcb->jcb", rows, shapedirs) # (4, 3, B)
    _SMPLX_FOOT_CACHE[key] = (J_template_feet, J_shapedirs_feet)
    return _SMPLX_FOOT_CACHE[key]


def compute_smpl_floor_height(betas, J_template_pelvis, J_shapedirs_pelvis,
                               J_template_feet, J_shapedirs_feet):
    pelvis_y = float((J_template_pelvis + J_shapedirs_pelvis @ betas)[1])
    feet_pos  = J_template_feet + torch.einsum("jcb,b->jc", J_shapedirs_feet, betas)
    min_foot_y = float(feet_pos[:, 1].min())
    return pelvis_y - min_foot_y               # > 0

def _first_frame_inv_rot(R0, yaw_only=False, up_axis=1, fwd_axis=2):

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
    R = axis_angle_to_matrix(global_orient_aa)                              # (F, 3, 3)
    R0_inv = _first_frame_inv_rot(
        R[0], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis
    )                                                                        # (3, 3)
    R_new      = R0_inv @ R                                                  # (F, 3, 3)
    transl_new = (transl - transl[0:1]) @ R0_inv.T                          # (F, 3)
    #   transl_new[0] = 0 after centering

    if pelvis_offset is not None:
        
        transl_new = transl_new - pelvis_offset
    return matrix_to_axis_angle(R_new), transl_new


def canonicalize_smpl_first_frame_bones_seed(global_orient_aa, joints,
                                             yaw_only=False, up_axis=2, fwd_axis=1):
    """Bones Seed/SOMA variant: center the first root joint and normalize heading."""
    R = axis_angle_to_matrix(global_orient_aa)
    R0_inv = _first_frame_inv_rot(
        R[0], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis
    )
    R_new = R0_inv @ R
    joints_new = (joints - joints[[0], [0]]) @ R0_inv.T[None]
    return matrix_to_axis_angle(R_new), joints_new


# ==========================================================
# canonicalize_g1_first_frame
# ==========================================================

def canonicalize_g1_first_frame(g1, root_body_id=0, yaw_only=False, up_axis=2, fwd_axis=0):
    body_quat_w = g1["body_quat_w"]                                         # (F, N, 4) wxyz
    body_pos_w  = g1["body_pos_w"]                                          # (F, N, 3)

    R = quaternion_to_matrix(body_quat_w)                                   # (F, N, 3, 3)
    R0_inv = _first_frame_inv_rot(
        R[0, root_body_id], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis
    )                                                                        # (3, 3)
    center = body_pos_w[0, root_body_id].clone()                            # (3,)

    R_new = R0_inv @ R                                                      # (F, N, 3, 3)
    q_new = matrix_to_quaternion(R_new)                                     # (F, N, 4) wxyz, ambiguous ±q

    
    sign = torch.where(
        q_new[..., 0:1] < 0,
        -torch.ones_like(q_new[..., 0:1]),
         torch.ones_like(q_new[..., 0:1]),
    )
    q_new = q_new * sign

    if not yaw_only:
        q_new[0, root_body_id] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=q_new.device, dtype=q_new.dtype
        )

    g1["body_quat_w"]    = q_new                                            # (F, N, 4) wxyz
    g1["body_pos_w"]     = (body_pos_w - center) @ R0_inv.T                 # (F, N, 3)
    g1["body_lin_vel_w"] = g1["body_lin_vel_w"] @ R0_inv.T                  # (F, N, 3)
    g1["body_ang_vel_w"] = g1["body_ang_vel_w"] @ R0_inv.T                  # (F, N, 3)
    g1["body_pos_w"][0, root_body_id] = 0.0
    return g1


def canonicalize_g1_first_frame_bones_seed(g1, root_body_id=0, yaw_only=False, up_axis=2, fwd_axis=0):
    """Bones Seed/G1 variant using FK full-body fields from the paired pkl files."""
    body_quat_w = g1["body_quat_w_full"]
    body_pos_w = g1["body_pos_w_full"]

    R = quaternion_to_matrix(body_quat_w)
    R0_inv = _first_frame_inv_rot(
        R[0, root_body_id], yaw_only=yaw_only, up_axis=up_axis, fwd_axis=fwd_axis
    )
    center = body_pos_w[0, root_body_id].clone()

    R_new = R0_inv @ R
    q_new = matrix_to_quaternion(R_new)
    sign = torch.where(
        q_new[..., 0:1] < 0,
        -torch.ones_like(q_new[..., 0:1]),
        torch.ones_like(q_new[..., 0:1]),
    )
    q_new = q_new * sign

    if not yaw_only:
        q_new[0, root_body_id] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=q_new.device, dtype=q_new.dtype
        )

    g1["body_quat_w"] = q_new
    g1["body_pos_w"] = (body_pos_w - center) @ R0_inv.T
    g1["root_rot"] = g1["body_quat_w"][:, root_body_id].clone()
    g1["root_pos"] = g1["body_pos_w"][:, root_body_id].clone()
    return g1
