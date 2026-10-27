import torch
import torch.nn.functional as F
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from einops import rearrange


def aa_to_r6d(x):
    return matrix_to_rotation_6d(axis_angle_to_matrix(x))


def r6d_to_aa(x):
    return matrix_to_axis_angle(rotation_6d_to_matrix(x))


def interpolate_smpl_params(smpl_params, tgt_len):
    """
    smpl_params['body_pose'] (L, 63)
    tgt_len: L->L'
    """
    betas = smpl_params["betas"]
    body_pose = smpl_params["body_pose"]
    global_orient = smpl_params["global_orient"]  # (L, 3)
    transl = smpl_params["transl"]  # (L, 3)

    # Interpolate
    body_pose = rearrange(aa_to_r6d(body_pose.reshape(-1, 21, 3)), "l j c -> c j l")
    body_pose = F.interpolate(body_pose, tgt_len, mode="linear", align_corners=True)
    body_pose = r6d_to_aa(rearrange(body_pose, "c j l -> l j c")).reshape(-1, 63)

    # although this should be the same as above, we do it for consistency
    betas = rearrange(betas, "l c -> c 1 l")
    betas = F.interpolate(betas, tgt_len, mode="linear", align_corners=True)
    betas = rearrange(betas, "c 1 l -> l c")

    # global_orient: SLERP via quaternion to avoid R6D linear-interp artifacts
    # on large rotations (>π) and speed-augmented sequences with fast turns.
    from pytorch3d.transforms import (
        axis_angle_to_quaternion,
        quaternion_to_axis_angle,
    )
    # Local import to avoid forcing a project-wide pytorch3d dep at module load.
    # quat is wxyz, shape (L, 4)
    go_quat = axis_angle_to_quaternion(global_orient)              # (L, 4) wxyz
    go_quat_t = _slerp_quat_sequence(go_quat, tgt_len)             # (tgt_len, 4)
    global_orient = quaternion_to_axis_angle(go_quat_t)            # (tgt_len, 3)

    transl = rearrange(transl, "l c -> c 1 l")
    transl = F.interpolate(transl, tgt_len, mode="linear", align_corners=True)
    transl = rearrange(transl, "c 1 l -> l c")

    return {"body_pose": body_pose, "betas": betas, "global_orient": global_orient, "transl": transl}


def _slerp_quat_sequence(q_seq, tgt_len):
    """SLERP a (T, 4) wxyz quaternion sequence to (tgt_len, 4)."""
    T = q_seq.shape[0]
    if T == tgt_len:
        return F.normalize(q_seq, dim=-1)
    if T == 1:
        return F.normalize(q_seq, dim=-1).expand(tgt_len, -1).clone()

    t_idx = torch.linspace(0, T - 1, tgt_len, device=q_seq.device, dtype=q_seq.dtype)
    i0 = t_idx.floor().long().clamp(0, T - 2)
    i1 = (i0 + 1).clamp(0, T - 1)
    alpha = (t_idx - i0.to(t_idx)).unsqueeze(-1)                   # (tgt_len, 1)

    q0 = F.normalize(q_seq[i0], dim=-1)
    q1 = F.normalize(q_seq[i1], dim=-1)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(0.0, 1.0)
    theta = torch.acos(dot)
    sin_t = theta.sin()
    w0 = torch.where(sin_t > 1e-6,
                     torch.sin((1.0 - alpha) * theta) / (sin_t + 1e-10),
                     1.0 - alpha)
    w1 = torch.where(sin_t > 1e-6,
                     torch.sin(alpha * theta) / (sin_t + 1e-10),
                     alpha)
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)


def rotate_around_axis(global_orient, transl, axis="y"):
    """Global coordinate augmentation. Random rotation around y-axis"""
    angle = torch.rand(1) * 2 * torch.pi
    if axis == "y":
        aa = torch.tensor([0.0, angle, 0.0]).float().unsqueeze(0)
    rmat = axis_angle_to_matrix(aa)

    global_orient = matrix_to_axis_angle(rmat @ axis_angle_to_matrix(global_orient))
    transl = (rmat.squeeze(0) @ transl.T).T
    return global_orient, transl


def augment_betas(betas, std=0.1):
    dim = betas.shape[-1]
    noise = torch.normal(mean=torch.zeros(dim), std=torch.ones(dim) * std)
    betas_aug = betas + noise[None]
    return betas_aug


# interpolate_g1_params + _finite_diff_lin_vel/_finite_diff_ang_vel @ 30fps,
