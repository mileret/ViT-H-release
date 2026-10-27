import torch 
import numpy as np
import motion_lib.motion_utils.rotation_conversions as tRot
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters
import motion_lib.rotation3d as pRot

# H1_ROTATION_AXIS = torch.tensor([[
#     [0, 0, 1], # l_hip_yaw
#     [1, 0, 0], # l_hip_roll
#     [0, 1, 0], # l_hip_pitch
#
#     [0, 1, 0], # kneel
#     [0, 1, 0], # ankle
#
#     [0, 0, 1], # r_hip_yaw
#     [1, 0, 0], # r_hip_roll
#     [0, 1, 0], # r_hip_pitch
#
#     [0, 1, 0], # kneel
#     [0, 1, 0], # ankle
#
#     [0, 0, 1], # torso
#
#     [0, 1, 0], # l_shoulder_pitch
#     [1, 0, 0], # l_roll_pitch
#     [0, 0, 1], # l_yaw_pitch
#
#     [0, 1, 0], # l_elbow
#
#     [0, 1, 0], # r_shoulder_pitch
#     [1, 0, 0], # r_roll_pitch
#     [0, 0, 1], # r_yaw_pitch
#
#     [0, 1, 0], # r_elbow
# ]])
# H1_ROTATION_AXIS = np.array([[
#     [0, 1, 0],  # l_hip_pitch
#     [1, 0, 0],  # l_hip_roll
#     [0, 0, 1],  # l_hip_yaw
#
#     [0, 1, 0],  # kneel
#     [0, 1, 0],  # ankle
#     [1, 0, 0],  # ankle_g1
#
#     [0, 1, 0],  # r_hip_pitch
#     [1, 0, 0],  # r_hip_roll
#     [0, 0, 1],  # r_hip_yaw
#
#     [0, 1, 0],  # kneel
#     [0, 1, 0],  # ankle
#     [1, 0, 0],  # ankle_g1
#
#     [0, 0, 1],  # torso
#
#     [0, 1, 0],  # l_shoulder_pitch
#     [1, 0, 0],  # l_roll_pitch
#     [0, 0, 1],  # l_yaw_pitch
#
#     [0, 1, 0],  # l_elbow
#     [1, 0, 0],  # l_elbow_g1
#
#     [0, 1, 0],  # r_shoulder_pitch
#     [1, 0, 0],  # r_roll_pitch
#     [0, 0, 1],  # r_yaw_pitch
#
#     [0, 1, 0],  # r_elbow
#     [1, 0, 0],  # l_elbow_g1
# ]])

class Humanoid_Batch:

    def __init__(self, mjcf_file = f"resources/robots/h1/h1.xml", extend_hand = True, extend_head = False, device = torch.device("cpu")):
        # Auto-detect URDF vs MJCF by file extension. Both parsers return the
        # same dict shape (node_names / parent_indices / local_translation /
        # local_rotation / joints_range), so the rest of FK is identical.
        if str(mjcf_file).lower().endswith('.urdf'):
            self.mjcf_data = mjcf_data = self.from_urdf(mjcf_file)
        else:
            self.mjcf_data = mjcf_data = self.from_mjcf(mjcf_file)
        self.extend_hand = extend_hand
        self.extend_head = extend_head
        if extend_hand:
            self.model_names = mjcf_data['node_names'] + ["left_hand_link", "right_hand_link"]
            self._parents = torch.cat((mjcf_data['parent_indices'], torch.tensor([18, 23]))).to(device) # Adding the hands joints
            arm_length = 0.3
            self._offsets = torch.cat((mjcf_data['local_translation'], torch.tensor([[arm_length, 0, 0], [arm_length, 0, 0]])), dim = 0)[None, ].to(device)
            self._local_rotation = torch.cat((mjcf_data['local_rotation'], torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])), dim = 0)[None, ].to(device)
            self._remove_idx = 2
        else:
            self._parents = mjcf_data['parent_indices']
            self.model_names = mjcf_data['node_names']
            self._offsets = mjcf_data['local_translation'][None, ].to(device)
            self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)
            
        if extend_head:
            self._remove_idx = 3
            self.model_names = self.model_names + ["head_link"]
            self._parents = torch.cat((self._parents, torch.tensor([0]).to(device))).to(device) # Adding the hands joints
            head_length = 0.45
            self._offsets = torch.cat((self._offsets, torch.tensor([[[0, 0, head_length]]]).to(device)), dim = 1).to(device)
            self._local_rotation = torch.cat((self._local_rotation, torch.tensor([[[1, 0, 0, 0]]]).to(device)), dim = 1).to(device)
            
        
        self.joints_range = mjcf_data['joints_range'].to(device)
        self._local_rotation_mat = tRot.quaternion_to_matrix(self._local_rotation).float() # w, x, y ,z
        
    def from_mjcf(self, path):
        # function from Poselib: 
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
            
        xml_joint_root = xml_body_root.find("joint")
        
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint")
            for joint in all_joints:
                if not joint.attrib.get("range") is None: 
                    joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
            
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            return node_index
        
        _add_xml_node(xml_body_root, -1, 0)
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "joints_range": torch.from_numpy(np.array(joints_range))
        }

    def from_urdf(self, path):
        """Parse a URDF and return the same dict shape as `from_mjcf`.

        Tree topology follows non-fixed joints only — for G1 this matches the
        MJCF body tree exactly (root + 29 revolute children = 30 nodes). Fixed
        joints (sensors, hand palms, foot frames) are skipped because they are
        not part of the actuated kinematic chain.

        Each node carries the joint origin (xyz, rpy→quat-wxyz) of the joint
        that connects it to its parent; the root has zero offset and identity
        rotation (FK overrides the root with `root_positions` anyway).
        """
        tree = ETree.parse(path)
        urdf_root = tree.getroot()

        # Collect non-fixed joints in declaration order. URDF rpy convention:
        # R = Rz(yaw) * Ry(pitch) * Rx(roll) → standard wxyz quaternion below.
        joint_info = {}                           # child_link -> dict
        joint_decl_order = []                     # for stable child ordering
        for j in urdf_root.findall('joint'):
            jtype = j.attrib.get('type')
            if jtype == 'fixed':
                continue
            parent_name = j.find('parent').attrib['link']
            child_name = j.find('child').attrib['link']
            origin = j.find('origin')
            if origin is not None:
                xyz = np.fromstring(origin.attrib.get('xyz', '0 0 0'), dtype=float, sep=' ')
                rpy = np.fromstring(origin.attrib.get('rpy', '0 0 0'), dtype=float, sep=' ')
            else:
                xyz = np.zeros(3, dtype=float)
                rpy = np.zeros(3, dtype=float)
            r, p, y = rpy
            cr, sr = np.cos(r / 2), np.sin(r / 2)
            cp, sp = np.cos(p / 2), np.sin(p / 2)
            cy, sy = np.cos(y / 2), np.sin(y / 2)
            quat = np.array([
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ], dtype=float)

            jrange = None
            limit = j.find('limit')
            if limit is not None and 'lower' in limit.attrib and 'upper' in limit.attrib:
                jrange = np.array([float(limit.attrib['lower']),
                                   float(limit.attrib['upper'])], dtype=float)
            joint_info[child_name] = {
                'parent': parent_name, 'pos': xyz, 'quat': quat, 'range': jrange,
            }
            joint_decl_order.append(child_name)

        # Find root link: one that is a parent of some non-fixed joint but
        # never a child of one.
        all_link_names = [l.attrib.get('name') for l in urdf_root.findall('link')]
        children_set = set(joint_info.keys())
        parents_set = set(info['parent'] for info in joint_info.values())
        root_candidates = [n for n in all_link_names
                           if n not in children_set and n in parents_set]
        if not root_candidates:
            raise ValueError("URDF parsed incorrectly: no root link with kinematic children.")
        root_link = root_candidates[0]

        # Build parent → ordered children map (preserve URDF declaration order)
        children_of = {}
        for c, info in joint_info.items():
            children_of.setdefault(info['parent'], []).append(c)
        order_idx = {n: i for i, n in enumerate(joint_decl_order)}
        for kids in children_of.values():
            kids.sort(key=lambda n: order_idx[n])

        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []

        def _dfs(name, parent_idx):
            cur_idx = len(node_names)
            node_names.append(name)
            parent_indices.append(parent_idx)
            if parent_idx == -1:
                local_translation.append(np.zeros(3, dtype=float))
                local_rotation.append(np.array([1, 0, 0, 0], dtype=float))
            else:
                info = joint_info[name]
                local_translation.append(info['pos'])
                local_rotation.append(info['quat'])
                if info['range'] is not None:
                    joints_range.append(info['range'])
            for c in children_of.get(name, []):
                _dfs(c, cur_idx)

        _dfs(root_link, -1)

        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "joints_range": torch.from_numpy(np.array(joints_range)) if joints_range
                            else torch.zeros((0, 2)),
        }


    def fk_batch(self, pose, trans, convert_to_mat=True, return_full = False, dt=1/30):
        device, dtype = pose.device, pose.dtype
        pose_input = pose.clone()
        B, seq_len = pose.shape[:2]
        pose = pose[..., :len(self._parents), :] # H1 fitted joints might have extra joints
        if self.extend_hand and self.extend_head and pose.shape[-2] == 32:
            pose = torch.cat([pose, torch.zeros(B, seq_len, 1, 3).to(device).type(dtype)], dim = -2) # adding hand and head joints

        if convert_to_mat:
            pose_quat = tRot.axis_angle_to_quaternion(pose)
            pose_mat = tRot.quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose
        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1  # Exclude root
        
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        
        return_dict = EasyDict()
        
        
        wbody_rot = tRot.wxyz_to_xyzw(tRot.matrix_to_quaternion(wbody_mat))
        if self.extend_hand:
            if return_full:
                return_dict.global_velocity_extend = self._compute_velocity(wbody_pos, dt) 
                return_dict.global_angular_velocity_extend = self._compute_angular_velocity(wbody_rot, dt)
                
            return_dict.global_translation_extend = wbody_pos.clone()
            return_dict.global_rotation_mat_extend = wbody_mat.clone()
            return_dict.global_rotation_extend = wbody_rot
            
            wbody_pos = wbody_pos[..., :-self._remove_idx, :]
            wbody_mat = wbody_mat[..., :-self._remove_idx, :, :]
            wbody_rot = wbody_rot[..., :-self._remove_idx, :]
        
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot
            
        if return_full:
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)  # Isaac gym is [x, y, z, w]. All the previous functions are [w, x, y, z]
            rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)
            return_dict.local_rotation = tRot.wxyz_to_xyzw(pose_quat)
            return_dict.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
            return_dict.global_root_angular_velocity = rigidbody_angular_velocity[..., 0, :]
            return_dict.global_angular_velocity = rigidbody_angular_velocity
            return_dict.global_velocity = rigidbody_linear_velocity
            
            if self.extend_hand or self.extend_head:
                return_dict.dof_pos = pose.sum(dim = -1)[..., 1:][..., :-self._remove_idx] # you can sum it up since unitree's each joint has 1 dof. Last two are for hands. doesn't really matter. 
            else:
                return_dict.dof_pos = pose.sum(dim = -1)[..., 1:] # you can sum it up since unitree's each joint has 1 dof. Last two are for hands. doesn't really matter. 
            
            dof_vel = ((return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1] )/dt)
            # return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim = 1)
            return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -1:]], dim=1)
            return_dict.fps = int(1/dt)
        
        return return_dict
    

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """
        
        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []

        expanded_offsets = (self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype))
        # print(expanded_offsets.shape, J)

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[self._parents[i]])
                rot_mat = torch.matmul(rotations_world[self._parents[i]], torch.matmul(self._local_rotation_mat[:,  (i):(i + 1)], rotations[:, :, (i - 1):i, :]))
                # rot_mat = torch.matmul(rotations_world[self._parents[i]], rotations[:, :, (i - 1):i, :])
                # print(rotations[:, :, (i - 1):i, :].shape, self._local_rotation_mat.shape)
                
                positions_world.append(jpos)
                rotations_world.append(rot_mat)
        
        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world
    
    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if guassian_filter:
            velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)
        
        return velocity
    
    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        # assume the second last dimension is the time axis
        # import ipdb; ipdb.set_trace()
        diff_quat_data = pRot.quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = pRot.quat_mul_norm(r[..., 1:, :, :], pRot.quat_inverse(r[..., :-1, :, :]))
        diff_angle, diff_axis = pRot.quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),)

        return angular_velocity  

