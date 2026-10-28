import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_quaternion,
    quaternion_to_matrix,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
)
from hmr4d.configs import MainStore, builds
from hmr4d.utils.geo.augment_noisy_pose import gaussian_augment
import hmr4d.utils.matrix as matrix
from hmr4d.utils.pylogger import Log
from hmr4d.utils.geo.hmr_global import get_local_transl_vel, rollout_local_transl_vel
from hmr4d.utils.smplx_utils import make_smplx
from . import stats_compose

# ------------------------------------------------------------------
# G1 joint-ordering constants (copied from dataset.py to avoid import)
# ------------------------------------------------------------------
_G1_BYD_JOINT_NAMES = [
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
_G1_MJCF_JOINT_NAMES = [
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
# index in byd for each mujoco slot  (len=29)
_G1_BYD_TO_MJC = [
    _G1_BYD_JOINT_NAMES.index(n + '_joint') for n in _G1_MJCF_JOINT_NAMES
]

# Per-joint rotation axes in MJC order (copied verbatim from dataset.py:13-53).
# Each row is the unit axis around which that 1-DOF joint rotates in the G1 URDF.
_G1_G5_ROTATION_AXIS = [
    [0, 1, 0],  # left_hip_pitch
    [1, 0, 0],  # left_hip_roll
    [0, 0, 1],  # left_hip_yaw
    [0, 1, 0],  # left_knee
    [0, 1, 0],  # left_ankle_pitch
    [1, 0, 0],  # left_ankle_roll
    [0, 1, 0],  # right_hip_pitch
    [1, 0, 0],  # right_hip_roll
    [0, 0, 1],  # right_hip_yaw
    [0, 1, 0],  # right_knee
    [0, 1, 0],  # right_ankle_pitch
    [1, 0, 0],  # right_ankle_roll
    [0, 0, 1],  # waist_yaw
    [1, 0, 0],  # waist_roll
    [0, 1, 0],  # waist_pitch
    [0, 1, 0],  # left_shoulder_pitch
    [1, 0, 0],  # left_shoulder_roll
    [0, 0, 1],  # left_shoulder_yaw
    [0, 1, 0],  # left_elbow
    [1, 0, 0],  # left_wrist_roll
    [0, 1, 0],  # left_wrist_pitch
    [0, 0, 1],  # left_wrist_yaw
    [0, 1, 0],  # right_shoulder_pitch
    [1, 0, 0],  # right_shoulder_roll
    [0, 0, 1],  # right_shoulder_yaw
    [0, 1, 0],  # right_elbow
    [1, 0, 0],  # right_wrist_roll
    [0, 1, 0],  # right_wrist_pitch
    [0, 0, 1],  # right_wrist_yaw
]

# G1 pred_x layout
# [body_pose_r6d(174) | global_orient_r6d(6) | global_orient_gv_r6d(6) | local_transl_vel(3)]
G1_N_DOFS = 29
G1_BODY_POSE_DIM = G1_N_DOFS * 6     # 174
G1_PRED_X_DIM    = G1_BODY_POSE_DIM + 6 + 6 + 3  # 189


# 30-body permutation: URDF-DFS body index → BYD-body slot.
# npz `body_pos_w[:, i, :]` indexes into BYD-body order:
#   body[0]   = pelvis
#   body[i+1] = link of BYD joint i  (i = 0..28)
# FK runs natively in URDF-DFS order; we re-gather the output along this index
# so callers that compare against npz body_pos_w see matching slots.
_G1_URDF_DFS_TO_BYD_BODY = [
    0,   # pelvis
    1,   # L_hip_pitch
    7,   # R_hip_pitch
    13,  # waist_yaw
    2,   # L_hip_roll
    8,   # R_hip_roll
    14,  # waist_roll
    3,   # L_hip_yaw
    9,   # R_hip_yaw
    15,  # waist_pitch (torso)
    4,   # L_knee
    10,  # R_knee
    16,  # L_shoulder_pitch
    23,  # R_shoulder_pitch
    5,   # L_ankle_pitch
    11,  # R_ankle_pitch
    17,  # L_shoulder_roll
    24,  # R_shoulder_roll
    6,   # L_ankle_roll
    12,  # R_ankle_roll
    18,  # L_shoulder_yaw
    25,  # R_shoulder_yaw
    19,  # L_elbow
    26,  # R_elbow
    20,  # L_wrist_roll
    27,  # R_wrist_roll
    21,  # L_wrist_pitch
    28,  # R_wrist_pitch
    22,  # L_wrist_yaw
    29,  # R_wrist_yaw
]


def _dof_to_pose_aa_torch(dof, root_rot, g5_rotation_axis):
    '''
    dof: (..., 29) - torch tensor
    root_rot: (..., 4) - torch tensor, quaternion in wxyz format
    g5_rotation_axis: (1, 29, 3) - torch tensor, rotation axes

    Returns:
    pose_aa: (..., 30, 3) - torch tensor, axis-angle representation
    '''

    # Convert DOF to axis-angle by scaling rotation axes

    dof_aa = g5_rotation_axis * dof.reshape(-1, 29, 1)
    dof_aa = dof_aa.reshape(*dof.shape[:-1], 29, 3)  # (..., 29, 3)

    # Convert quaternion to axis-angle for root rotation
    root_rot_aa = quaternion_to_axis_angle(root_rot)  # (..., 3)
    root_rot_aa = root_rot_aa.unsqueeze(-2)  # (..., 1, 3)

    # Concatenate root rotation with DOF axis-angles
    pose_aa = torch.cat((root_rot_aa, dof_aa), dim=-2)  # (..., 30, 3)

    return pose_aa


class EnDecoder(nn.Module):
    def __init__(self, stats_name="DEFAULT_01", noise_pose_k=10):
        super().__init__()
        # Load mean, std
        stats = getattr(stats_compose, stats_name)
        Log.info(f"[EnDecoder] Use {stats_name} for statistics!")
        self.register_buffer("mean", torch.tensor(stats["mean"]).float(), False)
        self.register_buffer("std", torch.tensor(stats["std"]).float(), False)

        # option
        self.noise_pose_k = noise_pose_k

        # smpl
        self.smplx_model = make_smplx("supermotion_v437coco17")
        parents = self.smplx_model.parents[:22]
        self.register_buffer("parents_tensor", parents, False)
        self.parents = parents.tolist()

    def get_noisyobs(self, data, return_type="r6d"):
        """
        Noisy observation contains local pose with noise
        Args:
            data (dict):
                body_pose: (B, L, J*3) or (B, L, J, 3)
        Returns:
            noisy_bosy_pose: (B, L, J, 6) or (B, L, J, 3) or (B, L, 3, 3) depends on return_type
        """
        body_pose = data["body_pose"]  # (B, L, 63)
        B, L, _ = body_pose.shape
        body_pose = body_pose.reshape(B, L, -1, 3)

        # (B, L, J, C)
        return_mapping = {"R": 0, "r6d": 1, "aa": 2}
        return_id = return_mapping[return_type]
        noisy_bosy_pose = gaussian_augment(body_pose, self.noise_pose_k, to_R=True)[return_id]
        return noisy_bosy_pose

    def normalize_body_pose_r6d(self, body_pose_r6d):
        """body_pose_r6d: (B, L, {J*6}/{J, 6}) ->  (B, L, J*6)"""
        B, L = body_pose_r6d.shape[:2]
        body_pose_r6d = body_pose_r6d.reshape(B, L, -1)
        if self.mean.shape[-1] == 1:  # no mean, std provided
            return body_pose_r6d
        body_pose_r6d = (body_pose_r6d - self.mean[:126]) / self.std[:126].clamp(min=1e-8)  # (B, L, C)
        return body_pose_r6d

    def fk_v2(self, body_pose, betas, global_orient=None, transl=None, get_intermediate=False):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)

        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

        if transl is not None:
            local_skeleton[..., 0, :] += transl  # B, L, 22, 3

        mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
        fk_mat = matrix.forward_kinematics(mat, self.parents)  # B, L, 22, 4, 4
        joints = matrix.get_position(fk_mat)  # B, L, 22, 3
        if not get_intermediate:
            return joints
        else:
            return joints, mat, fk_mat

    def get_local_pos(self, betas):
        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
        return local_skeleton

    def encode(self, inputs):
        """
        definition: {
                body_pose_r6d,  # (B, L, (J-1)*6) -> 0:126
                betas, # (B, L, 10) -> 126:136
                global_orient_r6d,  # (B, L, 6) -> 136:142  incam
                global_orient_gv_r6d: # (B, L, 6) -> 142:148  gv
                local_transl_vel,  # (B, L, 3) -> 148:151, smpl-coord
            }
        """
        B, L = inputs["smpl_params_c"]["body_pose"].shape[:2]
        # cam
        smpl_params_c = inputs["smpl_params_c"]
        body_pose = smpl_params_c["body_pose"].reshape(B, L, 21, 3)
        body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(-2)
        betas = smpl_params_c["betas"]
        global_orient_R = axis_angle_to_matrix(smpl_params_c["global_orient"])
        global_orient_r6d = matrix_to_rotation_6d(global_orient_R)

        # global
        R_c2gv = inputs["R_c2gv"]  # (B, L, 3, 3)
        global_orient_gv_r6d = matrix_to_rotation_6d(R_c2gv @ global_orient_R)

        # local_transl_vel
        smpl_params_w = inputs["smpl_params_w"]
        local_transl_vel = get_local_transl_vel(smpl_params_w["transl"], smpl_params_w["global_orient"])
        if False:  # debug
            transl_recover = rollout_local_transl_vel(
                local_transl_vel, smpl_params_w["global_orient"], smpl_params_w["transl"][:, [0]]
            )
            print((transl_recover - smpl_params_w["transl"]).abs().max())

        # returns
        x = torch.cat([body_pose_r6d, betas, global_orient_r6d, global_orient_gv_r6d, local_transl_vel], dim=-1)
        x_norm = (x - self.mean) / self.std.clamp(min=1e-8)
        return x_norm

    def encode_translw(self, inputs):
        """
        definition: {
                body_pose_r6d,  # (B, L, (J-1)*6) -> 0:126
                betas, # (B, L, 10) -> 126:136
                global_orient_r6d,  # (B, L, 6) -> 136:142  incam
                global_orient_gv_r6d: # (B, L, 6) -> 142:148  gv
                local_transl_vel,  # (B, L, 3) -> 148:151, smpl-coord
            }
        """
        # local_transl_vel
        smpl_params_w = inputs["smpl_params_w"]
        local_transl_vel = get_local_transl_vel(smpl_params_w["transl"], smpl_params_w["global_orient"])

        # returns
        x = local_transl_vel
        x_norm = (x - self.mean[-3:]) / self.std[-3:].clamp(min=1e-8)
        return x_norm

    def decode_translw(self, x_norm):
        return x_norm * self.std[-3:] + self.mean[-3:]

    def decode(self, x_norm):
        """x_norm: (B, L, C)"""
        B, L, C = x_norm.shape
        x = (x_norm * self.std) + self.mean

        body_pose_r6d = x[:, :, :126]
        betas = x[:, :, 126:136]
        global_orient_r6d = x[:, :, 136:142]
        global_orient_gv_r6d = x[:, :, 142:148]
        local_transl_vel = x[:, :, 148:151]

        body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6)))
        body_pose = body_pose.flatten(-2)
        global_orient_c = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))
        global_orient_gv = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_gv_r6d))

        output = {
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": global_orient_c,
            "global_orient_gv": global_orient_gv,
            "local_transl_vel": local_transl_vel,
        }

        return output
    # ------------------------------------------------------------------
    # G1 VITH-format encode / decode / FK
    # pred_x layout (189-D):
    #   [0:174]   body_pose_r6d   — 29 joints, each: [0,pos,0] → R → 6D
    #   [174:180] global_orient_r6d   — root quat(wxyz) → R → 6D
    #   [180:186] global_orient_gv_r6d — yaw-free root orientation → 6D
    #   [186:189] local_transl_vel     — Δtransl in body frame
    # ------------------------------------------------------------------

    def setup_g1_fk(self, mjcf_path: str, device=None, sampled_vertices_path: str | None = None):
        """Initialise G1 FK engine from MJCF or URDF.  Call once before training."""
        # `motion_lib` is a top-level package at the ViT-H repo root and not
        # pip-installed. If Hydra (or anything else) changes the cwd, `sys.path[0]`
        # ('') no longer resolves to the project root and the import below fails.
        # Make the import robust by adding the repo root to sys.path explicitly.
        import sys as _sys
        from pathlib import Path as _Path
        _repo_root = str(_Path(__file__).resolve().parents[4])   # .../ViT-H
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from motion_lib.torch_h1_humanoid_batch import Humanoid_Batch
        dev = device or next(self.parameters(), torch.zeros(1)).device
        self._g1_humanoid = Humanoid_Batch(
            extend_hand=False, extend_head=False,
            mjcf_file=mjcf_path, device=dev,
        )
        self._g1_byd_to_mjc = torch.tensor(_G1_BYD_TO_MJC, dtype=torch.long, device=dev)
        # Per-joint rotation axes in MJC order, used to map scalar DOF θ → axis-angle
        self._g5_rotation_axis = torch.tensor(
            _G1_G5_ROTATION_AXIS, dtype=torch.float32, device=dev
        )  # (29, 3)
        # URDF-DFS body index → BYD-body slot (used at FK output time so the
        # 30 returned positions are aligned with npz body_pos_w).
        self._g1_urdf_dfs_to_byd_body = torch.tensor(
            _G1_URDF_DFS_TO_BYD_BODY, dtype=torch.long, device=dev,
        )
        if sampled_vertices_path is not None:
            sampled_vertices = torch.load(
                sampled_vertices_path,
                map_location=dev,
                weights_only=True,
            )
            self.register_buffer(
                "_g1_vertices_local",
                sampled_vertices["vertices_local"].to(device=dev, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "_g1_vertex_body_indices_urdf",
                sampled_vertices["body_indices_urdf"].to(device=dev, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_g1_vertex_body_indices_byd",
                sampled_vertices["body_indices_byd"].to(device=dev, dtype=torch.long),
                persistent=False,
            )

    # --- helpers -------------------------------------------------------

    @staticmethod
    def _dof_to_r6d(joint_pos: torch.Tensor) -> torch.Tensor:
        """Convert 1-D DOF scalars to 6D rotation matrices.

        Each DOF `θ` is encoded as an axis-angle [0, θ, 0] (Y-axis rotation),
        then lifted to a rotation matrix and projected to 6D.

        Args:
            joint_pos: (B, L, 29) DOF values in byd order
        Returns:
            (B, L, 29*6=174) flattened 6D rotations
        """
        B, L, J = joint_pos.shape
        aa = torch.zeros(B, L, J, 3, device=joint_pos.device, dtype=joint_pos.dtype)
        aa[..., 1] = joint_pos                             # Y-axis only
        rotmat = axis_angle_to_matrix(aa)                  # (B, L, J, 3, 3)
        return matrix_to_rotation_6d(rotmat).flatten(-2)   # (B, L, J*6)

    # --- encode --------------------------------------------------------

    def encode_g1(self, inputs: dict) -> torch.Tensor:
        """Encode G1 inputs to normalised pred_x (B, L, 189).

        Layout (matches original VITH semantics):
            [0:174]   body_pose_r6d        — joint-local DOFs (frame-invariant)
            [174:180] global_orient_r6d    — root rotation in CAMERA frame
            [180:186] global_orient_gv_r6d — root rotation in GRAVITY-VIEW frame
            [186:189] local_transl_vel     — root displacement in body frame
                                              (derived from world-frame transl)

        Reads:
            inputs["g1_target"]:
                g1_joint_pos:   (B, L, 29)
                g1_body_pos_w:  (B, L, N, 3)   world frame
                g1_body_quat_w: (B, L, N, 4)   world frame, wxyz
            inputs["T_w2c"]:    (B, L, 4, 4)   world → camera transform
            inputs["R_c2gv"]:   (B, L, 3, 3)   camera → gravity-view rotation
        """
        tgt = inputs.get("g1_target", inputs)
        joint_pos    = tgt["g1_joint_pos"]      # (B, L, 29)
        body_pos_w   = tgt["g1_body_pos_w"]     # (B, L, N, 3)
        body_quat_w  = tgt["g1_body_quat_w"]    # (B, L, N, 4) wxyz

        # 1. body_pose_r6d: DOF → [0,θ,0] → R6D (frame-invariant)
        body_pose_r6d = self._dof_to_r6d(joint_pos)   # (B, L, 174)

        # 2. World-frame root rotation
        root_wxyz = body_quat_w[:, :, 0]              # (B, L, 4) wxyz
        root_R_w  = quaternion_to_matrix(root_wxyz)   # (B, L, 3, 3) world

        # 3. Camera-frame root rotation: R_c = R_w2c @ R_w
        
        
        g1_T_w2c  = tgt.get("g1_T_w2c", inputs["T_w2c"])  # (B, L, 4, 4)
        R_w2c_g1  = g1_T_w2c[..., :3, :3]                 # (B, L, 3, 3)
        root_R_c  = R_w2c_g1 @ root_R_w                   # (B, L, 3, 3) camera
        global_orient_r6d = matrix_to_rotation_6d(root_R_c)  # (B, L, 6)

        # 4. Gravity-view-frame root rotation: R_gv = R_c2gv @ R_c
        
        R_c2gv    = inputs["R_c2gv"]                   # (B, L, 3, 3)
        global_orient_gv_r6d = matrix_to_rotation_6d(R_c2gv @ root_R_c)  # (B, L, 6)

        # 5. local_transl_vel: world-frame transl in body frame (camera-independent)
        root_pos    = body_pos_w[:, :, 0]               # (B, L, 3)
        root_aa_w   = matrix_to_axis_angle(root_R_w)    # (B, L, 3) world
        local_transl_vel = get_local_transl_vel(root_pos, root_aa_w)  # (B, L, 3)

        x = torch.cat([body_pose_r6d, global_orient_r6d,
                       global_orient_gv_r6d, local_transl_vel], dim=-1)  # (B, L, 189)

        if hasattr(self, "g1_pred_mean"):
            x = (x - self.g1_pred_mean) / (self.g1_pred_std + 1e-8)
        return x

    # --- decode --------------------------------------------------------

    def decode_g1_new(self, x_norm: torch.Tensor) -> dict:
        """Decode normalised pred_x (B, L, 189) back to G1-format dict.

        Note: ``global_orient_rotmat`` is in CAMERA frame (matches encode_g1).
        For FK / world-frame rollout, callers must convert via T_w2c:
            R_w = R_w2c.T @ R_c

        Returns:
            body_pose_r6d:        (B, L, 29, 6)
            global_orient_r6d:    (B, L, 6)            CAMERA frame
            global_orient_gv_r6d: (B, L, 6)            GRAVITY-VIEW frame
            local_transl_vel:     (B, L, 3)            body-frame Δt of WORLD transl
            # convenience extras:
            body_pose_rotmat:     (B, L, 29, 3, 3)
            global_orient_rotmat: (B, L, 3, 3)         CAMERA frame
        """
        B, L, _ = x_norm.shape
        if hasattr(self, "g1_pred_mean"):
            x = x_norm * (self.g1_pred_std + 1e-8) + self.g1_pred_mean
        else:
            x = x_norm

        bp_r6d    = x[:, :, :G1_BODY_POSE_DIM].reshape(B, L, G1_N_DOFS, 6)  # (B,L,29,6)
        go_r6d    = x[:, :, G1_BODY_POSE_DIM:G1_BODY_POSE_DIM + 6]
        go_gv_r6d = x[:, :, G1_BODY_POSE_DIM + 6:G1_BODY_POSE_DIM + 12]
        ltv       = x[:, :, G1_BODY_POSE_DIM + 12:G1_BODY_POSE_DIM + 15]

        bp_rotmat = rotation_6d_to_matrix(bp_r6d)   # (B, L, 29, 3, 3)
        go_rotmat = rotation_6d_to_matrix(go_r6d)   # (B, L, 3, 3)

        return {
            "body_pose_r6d":        bp_r6d,
            "global_orient_r6d":    go_r6d,
            "global_orient_gv_r6d": go_gv_r6d,
            "local_transl_vel":     ltv,
            "body_pose_rotmat":     bp_rotmat,
            "global_orient_rotmat": go_rotmat,
        }

    # --- FK ------------------------------------------------------------

    def fk_g1(
        self,
        dof_pos: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        body_order: str = "byd",
        world_up: str = "z",
        return_vertices: bool = False,
    ):
        """G1 forward kinematics — clean version aligned with the npz convention.

        Input contract (matches blend.py + npz on disk):
          dof_pos  : (B, N, 29)   scalar joint angles, **BYD joint order**
          root_pos : (B, N, 3)    pelvis position, world coords
          root_rot : (B, N, 4)    pelvis orientation quat **wxyz**

        ``world_up='z'`` (default — what the npz on disk actually uses; verified
        sub-mm against `body_pos_w`) feeds the URDF FK natively. ``world_up='y'``
        applies a R_x(+90°) similarity (left-mul on world basis + right-mul on
        pelvis local basis) so identity y-up quat = pelvis upright.

        Output bodies are re-gathered to BYD body order by default so slot i
        lines up with `npz['body_pos_w'][:, i, :]`. Pass ``body_order='urdf'``
        for the URDF-DFS order (which is what `forward_kinematics_batch` natively
        produces).

        Returns:
            positions_world: (B, N, 30, 3)
            rotations_world: (B, N, 30, 3, 3)
            vertices_world: (B, N, 384, 3), only when return_vertices is True.
        """
        if not hasattr(self, "_g1_humanoid"):
            raise RuntimeError("Call setup_g1_fk(mjcf_path) before fk_g1.")
        if body_order not in ("byd", "urdf"):
            raise ValueError(f"body_order must be 'byd' or 'urdf', got {body_order!r}")
        if world_up not in ("y", "z"):
            raise ValueError(f"world_up must be 'y' or 'z', got {world_up!r}")
        if return_vertices and not hasattr(self, "_g1_vertices_local"):
            raise RuntimeError(
                "Pass sampled_vertices_path to setup_g1_fk() before requesting vertices."
            )

        device = root_pos.device
        dtype = root_pos.dtype

        # Humanoid_Batch is a plain Python object, not nn.Module — Lightning's
        # `.to(device)` doesn't migrate its tensors. Lazy-sync them on first
        # call so CPU ↔ GPU mismatch inside forward_kinematics_batch is impossible.
        h = self._g1_humanoid
        if h._parents.device != device:
            h._parents            = h._parents.to(device)
            h._offsets            = h._offsets.to(device)
            h._local_rotation     = h._local_rotation.to(device)
            h._local_rotation_mat = h._local_rotation_mat.to(device)
            if hasattr(h, "joints_range") and torch.is_tensor(h.joints_range):
                h.joints_range = h.joints_range.to(device)

        # ---- (1) BYD scalar dofs → MJC-ordered scalar dofs ---------------
        byd_to_mjc = self._g1_byd_to_mjc.to(device)
        g5_axis    = self._g5_rotation_axis.to(device).unsqueeze(0)         # (1, 29, 3)
        dof_mjc    = dof_pos[..., byd_to_mjc]                                # (B, N, 29)

        # ---- (2) optional y-up → z-up frame change on the root ----------
        
        
        
        
        
        if world_up == "y":
            R_y2z = torch.tensor(
                [[1.0, 0.0,  0.0],
                 [0.0, 0.0, -1.0],
                 [0.0, 1.0,  0.0]],
                dtype=dtype, device=device,
            )
            root_pos_in = root_pos @ R_y2z.T            
            root_R      = R_y2z @ quaternion_to_matrix(root_rot)   
            root_rot_in = matrix_to_quaternion(root_R)
        else:
            root_pos_in = root_pos
            root_rot_in = root_rot

        # ---- (3) verbatim compute_fk pipeline ----------------------------
        pose_aa   = _dof_to_pose_aa_torch(dof_mjc, root_rot_in, g5_axis)    # (B, N, 30, 3)
        pose_quat = axis_angle_to_quaternion(pose_aa)                       # wxyz
        pose_mat  = quaternion_to_matrix(pose_quat)                         # (B, N, 30, 3, 3)
        pos_w, rot_w = self._g1_humanoid.forward_kinematics_batch(
            pose_mat[:, :, 1:], pose_mat[:, :, 0:1], root_pos_in,
        )                                                                    # (B, N, 30, 3) URDF-DFS

        
        
        
        if world_up == "y":
            pos_w = pos_w @ R_y2z
            rot_w = R_y2z.T @ rot_w

        # ---- (5) URDF-DFS → BYD body order (default) --------------------
        if body_order == "byd":
            gather = self._g1_urdf_dfs_to_byd_body.to(device)
            pos_w  = pos_w[..., gather, :]
            rot_w  = rot_w[..., gather, :, :]

        if return_vertices:
            vertex_body_indices = (
                self._g1_vertex_body_indices_byd
                if body_order == "byd"
                else self._g1_vertex_body_indices_urdf
            ).to(device)
            vertices_local = self._g1_vertices_local.to(device=device, dtype=dtype)
            vertex_body_positions = pos_w.index_select(dim=-2, index=vertex_body_indices)
            vertex_body_rotations = rot_w.index_select(dim=-3, index=vertex_body_indices)
            vertices_world = torch.einsum(
                "...vij,vj->...vi",
                vertex_body_rotations,
                vertices_local,
            ) + vertex_body_positions
            return pos_w, rot_w, vertices_world

        return pos_w, rot_w

    # --- stats for new G1 pred_x (189-D) --------------------------------

    def set_g1_pred_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Register normalization statistics for the G1 prediction vector."""
        mean = mean.float().clone()
        std  = std.float().clone()
        self.register_buffer("g1_pred_mean", mean, persistent=False)
        self.register_buffer("g1_pred_std",  std,  persistent=False)

    def set_g1_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Register G1 mean/std (1D tensors) used to un-normalize model outputs.

        Args:
            mean: tensor of shape (C,)
            std: tensor of shape (C,)
        """
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean).float()
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std).float()
        self.register_buffer("g1_mean", mean.float(), False)
        self.register_buffer("g1_std", std.float(), False)
        self.g1_output_dim = mean.shape[-1]

    def compute_g1_stats_from_npz(self, npz_path):
        """Load G1 mean/std from stats NPZ file and register them.

        Expected format: keys 'mean' (C,) and 'std' (C,), where
        C = joint_pos(29) + body_pos_w(90) + body_quat_w_6d(180) = 299.
        """
        import numpy as _np

        data = _np.load(npz_path)
        keys = list(data.keys())

        if "mean" in keys and "std" in keys:
            mean = torch.tensor(data["mean"]).float()
            std = torch.tensor(data["std"]).float().clamp(min=1e-6)
            self.set_g1_stats(mean, std)
            return

        raise ValueError(
            f"Unrecognised g1_stats format. Keys: {keys}. "
            f"Expected 'mean'+'std' in {npz_path}. "
            f"Regenerate with: python mean_std.py"
        )

    def decode_g1(self, x_norm, joint_pos_dim=29, n_bodies=30):
        """Decode normalized features into G1-format outputs.

        Encoding order (must match pipeline):
            [joint_pos(29) | body_pos_w_flat(90) | body_quat_w_6d_flat(180)] = 299D

        Args:
            x_norm: (B, L, 299) normalized model output
            joint_pos_dim: joint position dims (default 29)
            n_bodies: number of rigid bodies (default 30)

        Returns:
            dict with ``joint_pos``, ``body_pos_w``, ``body_quat_w`` (xyzw)
        """
        if not hasattr(self, "g1_mean") or not hasattr(self, "g1_std"):
            raise RuntimeError("G1 stats not set. Call set_g1_stats or compute_g1_stats_from_npz first.")

        B, L, C = x_norm.shape
        x = x_norm * self.g1_std + self.g1_mean

        idx = 0
        joint_pos = x[:, :, idx : idx + joint_pos_dim]       # (B, L, 29)
        idx += joint_pos_dim

        body_pos_flat = x[:, :, idx : idx + n_bodies * 3]     # (B, L, 90)
        idx += n_bodies * 3
        body_pos_w = body_pos_flat.reshape(B, L, n_bodies, 3) # (B, L, 30, 3)

        body_quat_6d_flat = x[:, :, idx : idx + n_bodies * 6] # (B, L, 180)
        idx += n_bodies * 6
        body_quat_6d = body_quat_6d_flat.reshape(B, L, n_bodies, 6)

        # 6D → matrix → quaternion wxyz → xyzw
        body_rot_mat = rotation_6d_to_matrix(body_quat_6d)     # (B, L, 30, 3, 3)
        body_quat_wxyz = matrix_to_quaternion(body_rot_mat)     # (B, L, 30, 4)
        # wxyz → xyzw
        body_quat_w = torch.cat([body_quat_wxyz[..., 1:4], body_quat_wxyz[..., 0:1]], dim=-1)

        return {
            "joint_pos": joint_pos,     # (B, L, 29)
            "body_pos_w": body_pos_w,   # (B, L, 30, 3)
            "body_quat_w": body_quat_w, # (B, L, 30, 4) xyzw
        }


group_name = "endecoder/offline"
cfg_base = builds(EnDecoder, populate_full_signature=True)
MainStore.store(name="v1_no_stdmean", node=cfg_base, group=group_name)
MainStore.store(name="v1", node=cfg_base(stats_name="MM_V1"), group=group_name)
MainStore.store(
    name="v1_amass_local_bedlam_cam",
    node=cfg_base(stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM"),
    group=group_name,
)

MainStore.store(name="v2", node=cfg_base(stats_name="MM_V2"), group=group_name)
MainStore.store(name="v2_1", node=cfg_base(stats_name="MM_V2_1"), group=group_name)
