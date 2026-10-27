"""Small local compatibility subset of the roma package used by SAM3D.

The VITH environment already provides PyTorch3D; these two functions match
the conventions needed by Fast-SAM-3D-Body without installing into a shared
conda environment.
"""
import torch
import torch.nn.functional as F


def rotvec_to_rotmat(x):
    shape = x.shape
    v = x.reshape(-1, 3)
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    axis = v / angle.clamp_min(1e-8)
    half = angle * 0.5
    q = torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)
    w, a, b, c = q.unbind(-1)
    R = torch.stack((
        1 - 2 * (b*b + c*c), 2 * (a*b - c*w), 2 * (a*c + b*w),
        2 * (a*b + c*w), 1 - 2 * (a*a + c*c), 2 * (b*c - a*w),
        2 * (a*c - b*w), 2 * (b*c + a*w), 1 - 2 * (a*a + b*b),
    ), dim=-1).reshape(-1, 3, 3)
    return R.reshape(shape[:-1] + (3, 3))


def special_gramschmidt(x):
    a1, a2 = x[..., 0], x[..., 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)
