from pathlib import Path
import numpy as np
import torch
from hmr4d.utils.pylogger import Log
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from time import time

from hmr4d.configs import MainStore, builds
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.vis.renderer_utils import simple_render_mesh_background
from hmr4d.utils.video_io_utils import read_video_np, save_video

import hmr4d.utils.matrix as matrix
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.dataset.bedlam.utils import mid2featname, mid2vname
from hmr4d.utils.geo_transform import compute_cam_angvel, apply_T_on_points
from hmr4d.utils.geo.hmr_global import get_T_w2c_from_wcparams, get_c_rootparam, get_R_c2gv
from hmr4d.dataset.pure_motion.g1_amass import (
    _MJC_TO_BYD,
    _first_frame_inv_rot,
    _apply_ry_neg90_on_quat_wxyz,
    _apply_ry_neg90_on_vec,
    _finite_diff_ang_vel,
    _finite_diff_lin_vel,
    _quat_xyzw_to_wxyz,
    apply_ay_to_az_on_quat_wxyz,
    apply_ay_to_az_on_vec,
    apply_az_to_ay_on_quat_wxyz,
    apply_az_to_ay_on_vec,
    canonicalize_g1_first_frame,
    canonicalize_smpl_first_frame,
    compute_smpl_pelvis_offset,
    fk_g1_pth_slice,
    get_smplx_pelvis_buffers,
)


def _torch_load_cpu(path, mmap=True):
    kwargs = {"map_location": "cpu"}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        try:
            return torch.load(path, **kwargs)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except RuntimeError as exc:
        # Legacy non-zip torch.save files cannot be memory mapped.
        if mmap and "mmap" in str(exc).lower():
            return _torch_load_cpu(path, mmap=False)
        raise


def _normalize_pth_paths(paths):
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [Path(path) for path in paths if path is not None]


def _load_and_merge_seq_stores(paths, label, dataset_name, required=False):
    paths = _normalize_pth_paths(paths)
    merged = {}
    for path in paths:
        if not path.exists():
            Log.warning(f"[{dataset_name}] {label} pth not found: {path}")
            continue
        Log.info(f"[{dataset_name}] Loading {label}: {path}")
        store = _torch_load_cpu(path)
        duplicate = set(merged) & set(store)
        if duplicate:
            examples = sorted(duplicate)[:3]
            raise ValueError(
                f"[{dataset_name}] duplicate keys while merging {label}: "
                f"path={path}, count={len(duplicate)}, examples={examples}"
            )
        merged.update(store)
    if required and not merged:
        raise FileNotFoundError(f"[{dataset_name}] no readable {label} pth from: {paths}")
    return merged if merged else None


def _as_tensor_f32(x):
    return torch.as_tensor(x, dtype=torch.float32).clone()


def _first_existing_path(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return p
    return None


def _pick_entry_tensor(entry, names, start, end):
    for name in names:
        if name in entry:
            return _as_tensor_f32(entry[name][start:end])
    raise KeyError(f"missing any of {names} in entry keys={list(entry.keys())}")


def _maybe_slice_tensor(x, start, end):
    if x is None:
        return None
    x = _as_tensor_f32(x)
    return x[start:end]


def _bedlam2_human_to_offline_yup_rot(device, dtype):
    # Inverse of the bbox preproc's x_negz_y transform:
    # video_world = (x, -z, y).  VITH world = (x, z, -y).
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        device=device,
        dtype=dtype,
    )


def _convert_bedlam2_T_w2c_to_yup(T_w2c_video):
    R_yup_to_video = _bedlam2_human_to_offline_yup_rot(T_w2c_video.device, T_w2c_video.dtype).T
    T = T_w2c_video.clone()
    T[..., :3, :3] = T_w2c_video[..., :3, :3] @ R_yup_to_video
    return T


def _get_bedlam_smpl_T_w2c(smplx_raw, start, end):
    """Reconstruct BEDLAM world-to-camera transforms from paired SMPL-X data.

    The BEDLAM bbox PTH provides valid image-space boxes, but its
    stored camera transform is not in the same convention as the paired
    world-space SMPL-X motion.  BEDLAM's own camera-space root parameters are
    the authoritative bridge between those two coordinate systems.
    """
    required = ("global_orient_incam", "trans_incam", "cam_ext", "skeleton")
    missing = [name for name in required if name not in smplx_raw]
    if missing:
        raise KeyError(f"BEDLAM SMPL-X entry is missing camera fields: {missing}")

    global_orient_c = _as_tensor_f32(smplx_raw["global_orient_incam"][start:end])
    trans_incam = _as_tensor_f32(smplx_raw["trans_incam"][start:end])
    cam_ext = _as_tensor_f32(smplx_raw["cam_ext"][start:end])
    skeleton = _as_tensor_f32(smplx_raw["skeleton"])
    pose = _as_tensor_f32(smplx_raw["pose"][start:end])
    trans = _as_tensor_f32(smplx_raw["trans"][start:end])
    return get_T_w2c_from_wcparams(
        global_orient_w=pose[:, :3],
        transl_w=trans,
        global_orient_c=global_orient_c,
        transl_c=trans_incam + cam_ext[:, :3, 3],
        offset=skeleton[0],
    )


def _canonicalize_T_w2c_smpl_first_frame(T_w2c, global_orient_aa, transl, pelvis_offset=None, yaw_only=True):
    """Transform a real video camera into the same canonical world as SMPL.

    Keep this transform paired with ``canonicalize_smpl_first_frame``. SMPL-X
    global orientation rotates the body about the pelvis, so the pelvis offset
    is the (unrotated) pivot. The inverse camera compensation must therefore add
    ``transl0 + pelvis_offset``. Multiplying the offset by A or A.T introduces
    a constant camera-frame bias.
    """
    R0 = axis_angle_to_matrix(global_orient_aa[0])
    A = _first_frame_inv_rot(R0, yaw_only=yaw_only, up_axis=1, fwd_axis=2)
    center = transl[0].clone()
    if pelvis_offset is not None:
        center = center + pelvis_offset.to(center)

    T = T_w2c.clone()
    R_old = T_w2c[..., :3, :3]
    T[..., :3, :3] = R_old @ A.transpose(-1, -2)
    T[..., :3, 3] = T_w2c[..., :3, 3] + torch.einsum("fij,j->fi", R_old, center)
    return T


def _load_seq_feature(seq_store, key, start, end, names, default):
    if seq_store is None:
        return default, False
    entry = seq_store.get(key)
    if entry is None:
        return default, False
    if isinstance(entry, torch.Tensor):
        return _maybe_slice_tensor(entry, start, end), True
    base_start = int(entry.get("start_end", (0, 0))[0]) if isinstance(entry, dict) and "start_end" in entry else 0
    s = start - base_start
    e = end - base_start
    for name in names:
        if name in entry:
            return _maybe_slice_tensor(entry[name], s, e), True
    return default, False


def _load_imgfeat_entry(imgfeat_root, key, meta):
    if imgfeat_root is None:
        return None
    candidates = [
        imgfeat_root / f"{key}.pt",
        imgfeat_root / f"{key}.pth",
        imgfeat_root / meta.get("scene", "") / f"{meta.get('sequence', '')}.pt",
        imgfeat_root / meta.get("scene", "") / f"{meta.get('sequence', '')}.pth",
        imgfeat_root / meta.get("scene", "") / f"{meta.get('sequence', '')}-{meta.get('body', '')}.pt",
        imgfeat_root / meta.get("scene", "") / f"{meta.get('sequence', '')}-{meta.get('body', '')}.pth",
    ]
    p = _first_existing_path(candidates)
    if p is None:
        return None
    return _torch_load_cpu(p)


def _slice_imgfeat_entry(entry, start, end):
    if isinstance(entry, torch.Tensor):
        return _maybe_slice_tensor(entry, start, end), True
    base_start = int(entry.get("start_end", (0, 0))[0]) if isinstance(entry, dict) else 0
    s = start - base_start
    e = end - base_start
    for name in ("features", "f_imgseq", "imgfeat"):
        if isinstance(entry, dict) and name in entry:
            return _maybe_slice_tensor(entry[name], s, e), True
    return None, False


class BedlamDatasetV2(ImgfeatMotionDatasetBase):
    """mid_to_valid_range and features are newly generated."""

    MIDINDEX_TO_LOAD = {
        "all60": ("mid_to_valid_range_all60.pt", "imgfeats/bedlam_all60"),
        "maxspan60": ("mid_to_valid_range_maxspan60.pt", "imgfeats/bedlam_maxspan60"),
    }

    def __init__(
        self,
        mid_indices=["all60", "maxspan60"],
        lazy_load=True,  # Load from disk when needed
        random1024=False,  # Faster loading for debugging
    ):
        self.root = Path("inputs/BEDLAM/hmr4d_support")
        self.min_motion_frames = 60
        self.max_motion_frames = 120
        self.lazy_load = lazy_load
        self.random1024 = random1024

        # speficify mid_index to handle
        if not isinstance(mid_indices, list):
            mid_indices = [mid_indices]
        self.mid_indices = mid_indices
        assert all([m in self.MIDINDEX_TO_LOAD for m in mid_indices])

        super().__init__()

    def _load_dataset(self):
        Log.info(f"[BEDLAM] Loading from {self.root}")
        tic = time()
        # Load mid to valid range
        self.mid_to_valid_range = {}
        self.mid_to_imgfeat_dir = {}
        for m in self.mid_indices:
            fn, feat_dir = self.MIDINDEX_TO_LOAD[m]
            mid_to_valid_range_ = torch.load(self.root / fn)
            self.mid_to_valid_range.update(mid_to_valid_range_)
            self.mid_to_imgfeat_dir.update({mid: self.root / feat_dir for mid in mid_to_valid_range_})

        # Load motionfiles
        Log.info(f"[BEDLAM] Start loading motion files")
        if self.random1024:  # Debug, faster loading
            try:
                Log.info(f"[BEDLAM] Loading 1024 samples for debugging ...")
                self.motion_files = torch.load(self.root / "smplpose_v2_random1024.pth")
            except:
                Log.info(f"[BEDLAM] Not found, saving 1024 samples to disk ...")
                self.motion_files = torch.load(self.root / "smplpose_v2.pth")
                keys = list(self.motion_files.keys())
                keys = np.random.choice(keys, 1024, replace=False)
                self.motion_files = {k: self.motion_files[k] for k in keys}
                torch.save(self.motion_files, self.root / "smplpose_v2_random1024.pth")
            self.mid_to_valid_range = {k: v for k, v in self.mid_to_valid_range.items() if k in self.motion_files}
        else:
            self.motion_files = torch.load(self.root / "smplpose_v2.pth")
        Log.info(f"[BEDLAM] Motion files loaded. Elapsed: {time() - tic:.2f}s")

    def _get_idx2meta(self):
        # sum_frame = sum([e-s for s, e in self.mid_to_valid_range.values()])
        self.idx2meta = list(self.mid_to_valid_range.keys())
        Log.info(f"[BEDLAM] {len(self.idx2meta)} sequences. ")

    def _load_data(self, idx):
        mid = self.idx2meta[idx]
        # neutral smplx : "pose": (F, 63), "trans": (F, 3), "beta": (10),
        #           and : "skeleton": (J, 3)
        data = self.motion_files[mid].copy()

        # Random select a subset
        range1, range2 = self.mid_to_valid_range[mid]  # [range1, range2)
        mlength = range2 - range1
        min_motion_len = self.min_motion_frames
        max_motion_len = self.max_motion_frames

        if mlength < min_motion_len:  # the minimal mlength is 30 when generating data
            start = range1
            length = mlength
        else:
            effect_max_motion_len = min(max_motion_len, mlength)
            length = np.random.randint(min_motion_len, effect_max_motion_len + 1)  # [low, high)
            start = np.random.randint(range1, range2 - length + 1)
        end = start + length
        data["start_end"] = (start, end)
        data["length"] = length

        # Update data to a subset
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and len(v.shape) > 1 and k != "skeleton":
                data[k] = v[start:end]

        # Load img(as feature) : {mid -> 'features', 'bbx_xys', 'img_wh', 'start_end'}
        imgfeat_dir = self.mid_to_imgfeat_dir[mid]
        f_img_dict = torch.load(imgfeat_dir / mid2featname(mid))

        # remap (start, end)
        start_mapped = start - f_img_dict["start_end"][0]
        end_mapped = end - f_img_dict["start_end"][0]

        data["f_imgseq"] = f_img_dict["features"][start_mapped:end_mapped].float()  # (L, 1024)
        data["bbx_xys"] = f_img_dict["bbx_xys"][start_mapped:end_mapped].float()  # (L, 4)
        data["img_wh"] = f_img_dict["img_wh"]  # (2)
        data["kp2d"] = torch.zeros((end - start), 17, 3)  # (L, 17, 3)  # do not provide kp2d

        return data

    def _process_data(self, data, idx):
        length = data["length"]

        # SMPL params in cam
        body_pose = data["pose"][:, 3:]  # (F, 63)
        betas = data["beta"].repeat(length, 1)  # (F, 10)
        global_orient = data["global_orient_incam"]  # (F, 3)
        transl = data["trans_incam"] + data["cam_ext"][:, :3, 3]  # (F, 3), bedlam convention
        smpl_params_c = {"body_pose": body_pose, "betas": betas, "transl": transl, "global_orient": global_orient}

        # SMPL params in world
        global_orient_w = data["pose"][:, :3]  # (F, 3)
        transl_w = data["trans"]  # (F, 3)
        smpl_params_w = {"body_pose": body_pose, "betas": betas, "transl": transl_w, "global_orient": global_orient_w}

        gravity_vec = torch.tensor([0, -1, 0], dtype=torch.float32)  # (3), BEDLAM is ay
        T_w2c = get_T_w2c_from_wcparams(
            global_orient_w=global_orient_w,
            transl_w=transl_w,
            global_orient_c=global_orient,
            transl_c=transl,
            offset=data["skeleton"][0],
        )  # (F, 4, 4)
        R_c2gv = get_R_c2gv(T_w2c[:, :3, :3], gravity_vec)  # (F, 3, 3)

        # cam_angvel (slightly different from WHAM)
        cam_angvel = compute_cam_angvel(T_w2c[:, :3, :3])  # (F, 6)

        # Returns: do not forget to make it batchable! (last lines)
        max_len = self.max_motion_frames
        return_data = {
            "meta": {"data_name": "bedlam", "idx": idx},
            "length": length,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,  # (F, 3, 3)
            "gravity_vec": gravity_vec,  # (3)
            "bbx_xys": data["bbx_xys"],  # (F, 3)
            "K_fullimg": data["cam_int"],  # (F, 3, 3)
            "f_imgseq": data["f_imgseq"],  # (F, D)
            "kp2d": data["kp2d"],  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "mask": {
                "valid": get_valid_mask(max_len, length),
                "vitpose": False,
                "bbx_xys": True,
                "f_imgseq": True,
                "spv_incam_only": False,
            },
        }

        if False:  # check transformation, wis3d: sampled motion (global, incam)
            wis3d = make_wis3d(name="debug-data-bedlam")
            smplx = make_smplx("supermotion")

            # global
            smplx_out = smplx(**smpl_params_w)
            w_gt_joints = smplx_out.joints
            add_motion_as_lines(w_gt_joints, wis3d, name="w-gt_joints")

            # incam
            smplx_out = smplx(**smpl_params_c)
            c_gt_joints = smplx_out.joints
            add_motion_as_lines(c_gt_joints, wis3d, name="c-gt_joints")

            # Check transformation works correctly
            print("T_w2c", (apply_T_on_points(w_gt_joints, T_w2c) - c_gt_joints).abs().max())
            R_c, t_c = get_c_rootparam(
                smpl_params_w["global_orient"], smpl_params_w["transl"], T_w2c, data["skeleton"][0]
            )
            print("transl_c", (t_c - smpl_params_c["transl"]).abs().max())
            R_diff = matrix_to_axis_angle(
                (axis_angle_to_matrix(R_c) @ axis_angle_to_matrix(smpl_params_c["global_orient"]).transpose(-1, -2))
            ).norm(dim=-1)
            print("global_orient_c", R_diff.abs().max())  # < 1e-6

            skeleton_beta = smplx.get_skeleton(smpl_params_c["betas"])
            print("Skeleton", (skeleton_beta[0] - data["skeleton"]).abs().max())  # (1.2e-7)

        if False:  # cam-overlay
            smplx = make_smplx("supermotion")

            # *. original bedlam param
            # mid = self.idx2meta[idx]
            # video_path = "-".join(mid.replace("bedlam_data/", "inputs/bedlam/").split("-")[:-1])
            # npz_file = "inputs/bedlam/processed_labels/20221024_3-10_100_batch01handhair_static_highSchoolGym.npz"
            # params = np.load(npz_file, allow_pickle=True)
            # mid2index = {}
            # for j in tqdm(range(len(params["video_name"]))):
            #     k = params["video_name"][j] + "-" + params["sub"][j]
            #     mid2index[k] = j
            # betas = params['shape'][mid2index[mid]][:length]
            # global_orient_incam = torch.from_numpy(params['pose_cam'][121][:, :3])
            # body_pose = torch.from_numpy(params['pose_cam'][121][:, 3:66])
            # transl_incam = torch.from_numpy(params["trans_cam"][121])
            smplx_out = smplx(**smpl_params_c)

            # ----- Render Overlay ----- #
            mid = self.idx2meta[idx]
            images = read_video_np(self.root / "videos" / mid2vname(mid), data["start_end"][0], data["start_end"][1])
            render_dict = {
                "K": data["cam_int"][:1],  # only support batch-size 1
                "faces": smplx.faces,
                "verts": smplx_out.vertices,
                "background": images,
            }
            img_overlay = simple_render_mesh_background(render_dict)
            save_video(img_overlay, "tmp.mp4", crf=23)

        # Batchable
        return_data["smpl_params_c"] = repeat_to_max_len_dict(return_data["smpl_params_c"], max_len)
        return_data["smpl_params_w"] = repeat_to_max_len_dict(return_data["smpl_params_w"], max_len)
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
        return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)
        return return_data


class Bedlam2G1Dataset(ImgfeatMotionDatasetBase):
    """BEDLAM-family video observations paired with G1 targets.

    Bbox/camera, ViTPose, and image features may each be split over multiple
    disjoint PTH files. H36M also uses this implementation with a two-frame
    observation offset and without the BEDLAM Unreal/video-world conversion.
    """

    def __init__(
        self,
        data_dir="/path/to/training_set/bedlam2/part_00",
        bbox_pth_path="/path/to/training_set/bedlam2/part_00/bbox_00.pth",
        bbox_pth_paths=None,
        vitpose_pth_path=None,
        vitpose_pth_paths=None,
        imgfeat_pth_path=None,
        imgfeat_pth_paths=None,
        imgfeat_dir=None,
        smplx_pth_name="motion.pth",
        g1_pth_name="g1.pth",
        motion_frames=120,
        min_motion_frames=60,
        limit_size=None,
        betas_dim=10,
        root_body_id=0,
        # PHUMA BEDLAM2 G1 comes from the MuJoCo/GMR qpos stream, which is
        # already z-up.  Setting this true would rotate the robot twice.
        g1_input_is_yup=False,
        dof_pos_order="mjc",
        canonicalize_first_frame=True,
        floor_adjust=True,
        yaw_only_canon=True,
        align_target="pelvis",
        split="train",
        split_seed=42,
        split_ratios=(0.9, 0.05, 0.05),
        full_sequence=False,
        subset_split=None,
        subset_n=None,
        subset_seed=42,
        convert_T_w2c_to_yup=True,
        camera_source="bbox",
        observation_frame_offset=0,
        dataset_name="BEDLAM2_G1",
        data_name="bedlam2_g1",
        smplx_neutral_npz_path="inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    ):
        assert split in ("train", "val", "test"), split
        assert dof_pos_order in ("mjc", "byd"), dof_pos_order
        assert abs(sum(split_ratios) - 1.0) < 1e-6
        self.data_dir = Path(data_dir)
        self.smplx_pth_path = self.data_dir / smplx_pth_name
        self.g1_pth_path = self.data_dir / g1_pth_name
        self.bbox_pth_paths = _normalize_pth_paths(bbox_pth_paths if bbox_pth_paths is not None else bbox_pth_path)
        self.vitpose_pth_paths = _normalize_pth_paths(
            vitpose_pth_paths if vitpose_pth_paths is not None else vitpose_pth_path
        )
        self.imgfeat_pth_paths = _normalize_pth_paths(
            imgfeat_pth_paths if imgfeat_pth_paths is not None else imgfeat_pth_path
        )
        self.bbox_pth_path = self.bbox_pth_paths[0] if self.bbox_pth_paths else None
        self.vitpose_pth_path = self.vitpose_pth_paths[0] if self.vitpose_pth_paths else None
        self.imgfeat_pth_path = self.imgfeat_pth_paths[0] if self.imgfeat_pth_paths else None
        self.imgfeat_dir = Path(imgfeat_dir) if imgfeat_dir is not None else None
        self.motion_frames = int(motion_frames)
        self.min_motion_frames = int(min_motion_frames)
        self.limit_size = limit_size
        self.betas_dim = int(betas_dim)
        self.root_body_id = int(root_body_id)
        self.g1_input_is_yup = bool(g1_input_is_yup)
        self.dof_pos_order = dof_pos_order
        self.canonicalize_first_frame = bool(canonicalize_first_frame)
        self.floor_adjust = bool(floor_adjust)
        self.yaw_only_canon = bool(yaw_only_canon)
        self.align_target = align_target
        self.split = split
        self.split_seed = int(split_seed)
        self.split_ratios = tuple(float(r) for r in split_ratios)
        self.full_sequence = bool(full_sequence) and split != "train"
        self.subset_split = subset_split
        self.subset_n = int(subset_n) if subset_n is not None else None
        self.subset_seed = int(subset_seed)
        self.convert_T_w2c_to_yup = bool(convert_T_w2c_to_yup)
        if camera_source not in ("bbox", "bedlam_smpl"):
            raise ValueError(f"camera_source must be 'bbox' or 'bedlam_smpl', got {camera_source!r}")
        self.camera_source = camera_source
        self.observation_frame_offset = int(observation_frame_offset)
        if self.observation_frame_offset < 0:
            raise ValueError(f"observation_frame_offset must be >= 0, got {self.observation_frame_offset}")
        self.dataset_name = str(dataset_name)
        self.data_name = str(data_name)

        self._J_template_pelvis, self._J_shapedirs_pelvis = get_smplx_pelvis_buffers(
            smplx_neutral_npz_path,
            num_betas=self.betas_dim,
        )
        self.smplx_lite = make_smplx("supermotion_smpl24") if self.floor_adjust else None

        super().__init__()

    def __len__(self):
        if self.limit_size is not None:
            return min(int(self.limit_size), len(self.idx2meta))
        return len(self.idx2meta)

    def _load_dataset(self):
        Log.info(f"[{self.dataset_name}] Loading SMPL-X: {self.smplx_pth_path}")
        self.smplx_files = _torch_load_cpu(self.smplx_pth_path)
        Log.info(f"[{self.dataset_name}] Loading G1: {self.g1_pth_path}")
        self.g1_files = _torch_load_cpu(self.g1_pth_path)

        self.bbox_files = _load_and_merge_seq_stores(
            self.bbox_pth_paths, "video bbox/camera", self.dataset_name, required=True
        )
        self.vitpose_files = _load_and_merge_seq_stores(
            self.vitpose_pth_paths, "ViTPose kp2d", self.dataset_name
        )
        self.imgfeat_files = _load_and_merge_seq_stores(
            self.imgfeat_pth_paths, "image features", self.dataset_name
        )

        common = sorted(set(self.smplx_files) & set(self.g1_files) & set(self.bbox_files))
        if not common:
            raise ValueError(
                f"[{self.dataset_name}] no common keys among SMPL-X, G1, and bbox pth files. "
                f"smplx={len(self.smplx_files)} g1={len(self.g1_files)} bbox={len(self.bbox_files)}"
            )
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(common))
        n_total = len(common)
        n_train = int(self.split_ratios[0] * n_total)
        n_val = int(self.split_ratios[1] * n_total)
        if self.split_ratios[2] <= 0:
            n_val = n_total - n_train
        split_indices = {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val :],
        }
        selected_split = self.subset_split or self.split
        assert selected_split in ("train", "val", "test")
        keys = [common[i] for i in split_indices[selected_split]]
        if self.subset_n is not None and self.subset_n < len(keys):
            rng2 = np.random.default_rng(self.subset_seed)
            pick = rng2.choice(len(keys), size=self.subset_n, replace=False)
            pick.sort()
            keys = [keys[i] for i in pick]
        self.split_keys = keys
        Log.info(
            f"[{self.dataset_name}] split={self.split} selected={len(self.split_keys)}/{n_total} "
            f"keys, camera_source={self.camera_source}, obs_offset={self.observation_frame_offset}, "
            f"vitpose={self.vitpose_files is not None}, "
            f"imgfeat={self.imgfeat_files is not None or self.imgfeat_dir is not None}"
        )

    def _seq_len(self, key):
        bbox_len = int(self.bbox_files[key]["bbx_xys"].shape[0]) - self.observation_frame_offset
        return min(
            int(self.smplx_files[key]["pose"].shape[0]),
            int(self.g1_files[key]["root_pos"].shape[0]),
            max(bbox_len, 0),
        )

    def _get_idx2meta(self):
        self.idx2meta = []
        skipped = 0
        for key in self.split_keys:
            length = self._seq_len(key)
            if length < self.min_motion_frames:
                skipped += 1
                continue
            if self.full_sequence:
                self.idx2meta.append({"key": key, "usable_len": length, "seg_id": 0, "num_seg": 1})
            else:
                n_seg = max(length // self.motion_frames, 1)
                for seg_id in range(n_seg):
                    self.idx2meta.append({"key": key, "usable_len": length, "seg_id": seg_id, "num_seg": n_seg})
        Log.info(f"[{self.dataset_name}] idx2meta={len(self.idx2meta)} skipped={skipped}")

    def _slice_bounds(self, idx):
        meta = self.idx2meta[idx]
        usable_len = int(meta["usable_len"])
        tgt_len = self.motion_frames
        if self.split == "train":
            lo = min(self.min_motion_frames, usable_len)
            hi = min(tgt_len, usable_len)
            length = np.random.randint(lo, hi + 1) if hi > lo else lo
            start = np.random.randint(0, usable_len - length + 1) if length < usable_len else 0
            return start, start + length
        if self.full_sequence:
            return 0, usable_len
        length = min(tgt_len, usable_len)
        start = int(meta.get("seg_id", 0)) * tgt_len
        if start + length > usable_len:
            start = max(0, usable_len - length)
        return start, start + length

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        key = meta["key"]
        start, end = self._slice_bounds(idx)
        obs_start = start + self.observation_frame_offset
        obs_end = end + self.observation_frame_offset
        smplx_raw = self.smplx_files[key]
        g1_raw = self.g1_files[key]
        bbox_raw = self.bbox_files[key]

        pose = _as_tensor_f32(smplx_raw["pose"][start:end])
        trans = _as_tensor_f32(smplx_raw["trans"][start:end])
        beta_key = "beta" if "beta" in smplx_raw else "betas"
        beta = _as_tensor_f32(smplx_raw.get(beta_key, torch.zeros(self.betas_dim)))[: self.betas_dim]
        length = int(pose.shape[0])
        smpl = {
            "body_pose": pose[:, 3:66],
            "betas": beta.unsqueeze(0).expand(length, -1).clone(),
            "global_orient": pose[:, :3],
            "transl": trans,
        }

        if self.camera_source == "bedlam_smpl":
            T_w2c = _get_bedlam_smpl_T_w2c(smplx_raw, start, end)
            if "cam_int" in smplx_raw:
                K_fullimg = _as_tensor_f32(smplx_raw["cam_int"][start:end])
            else:
                K_fullimg = _as_tensor_f32(bbox_raw["K_fullimg"][obs_start:obs_end])
        else:
            T_w2c = _as_tensor_f32(bbox_raw["T_w2c"][obs_start:obs_end])
            if self.convert_T_w2c_to_yup:
                T_w2c = _convert_bedlam2_T_w2c_to_yup(T_w2c)
            K_fullimg = _as_tensor_f32(bbox_raw["K_fullimg"][obs_start:obs_end])
        bbx_xys = _as_tensor_f32(bbox_raw["bbx_xys"][obs_start:obs_end])
        bbox_len = int(bbox_raw["bbx_xys"].shape[0])
        bbox_valid_all = bbox_raw.get("valid", torch.ones(bbox_len, dtype=torch.bool))
        bbox_valid = torch.as_tensor(bbox_valid_all[obs_start:obs_end]).bool()

        kp2d_default = torch.zeros(length, 17, 3, dtype=torch.float32)
        kp2d, has_vitpose = _load_seq_feature(
            self.vitpose_files,
            key,
            obs_start,
            obs_end,
            names=("kp2d", "vitpose", "keypoints", "keypoints2d"),
            default=kp2d_default,
        )

        f_imgseq_default = torch.zeros(length, 1024, dtype=torch.float32)
        f_imgseq, has_imgfeat = _load_seq_feature(
            self.imgfeat_files,
            key,
            obs_start,
            obs_end,
            names=("f_imgseq", "features", "imgfeat"),
            default=f_imgseq_default,
        )
        if not has_imgfeat and self.imgfeat_dir is not None:
            imgfeat_entry = _load_imgfeat_entry(self.imgfeat_dir, key, smplx_raw.get("metadata", {}))
            if imgfeat_entry is not None:
                f_imgseq_loaded, has_imgfeat = _slice_imgfeat_entry(imgfeat_entry, obs_start, obs_end)
                if has_imgfeat:
                    f_imgseq = f_imgseq_loaded

        fps = float(g1_raw.get("fps", bbox_raw.get("fps", 30.0)))
        dof_pos = _as_tensor_f32(g1_raw["dof_pos"][start:end])
        if self.dof_pos_order == "mjc":
            dof_pos = dof_pos[..., _MJC_TO_BYD]
        root_pos_raw = _pick_entry_tensor(g1_raw, ("root_pos", "root_trans"), start, end)
        root_rot_raw = _pick_entry_tensor(g1_raw, ("root_rot", "root_ori", "root_quat"), start, end)
        if root_rot_raw.shape[-1] == 4:
            root_rot_wxyz_raw = _quat_xyzw_to_wxyz(root_rot_raw)
        else:
            raise ValueError(f"[{self.dataset_name}] expected root quaternion dim 4, got {root_rot_raw.shape}")

        if self.g1_input_is_yup:
            root_pos_z = apply_ay_to_az_on_vec(root_pos_raw)
            root_rot_wxyz_z = apply_ay_to_az_on_quat_wxyz(root_rot_wxyz_raw)
        else:
            root_pos_z = root_pos_raw
            root_rot_wxyz_z = root_rot_wxyz_raw

        with torch.no_grad():
            body_pos_w_z, body_quat_w_z = fk_g1_pth_slice(dof_pos, root_pos_z, root_rot_wxyz_z, device=dof_pos.device)
        g1 = {
            "joint_pos": dof_pos,
            "joint_vel": _finite_diff_lin_vel(dof_pos, fps),
            "body_pos_w": body_pos_w_z,
            "body_quat_w": body_quat_w_z,
            "body_lin_vel_w": _finite_diff_lin_vel(body_pos_w_z, fps),
            "body_ang_vel_w": _finite_diff_ang_vel(body_quat_w_z, fps),
        }

        if self.canonicalize_first_frame:
            smpl_global_orient_precanon = smpl["global_orient"].clone()
            smpl_transl_precanon = smpl["transl"].clone()
            g1 = canonicalize_g1_first_frame(
                g1, root_body_id=self.root_body_id, yaw_only=self.yaw_only_canon, up_axis=2, fwd_axis=0
            )
            pelvis_offset = None
            if self.align_target == "pelvis":
                pelvis_offset = compute_smpl_pelvis_offset(smpl["betas"][0], self._J_template_pelvis, self._J_shapedirs_pelvis)
            T_w2c = _canonicalize_T_w2c_smpl_first_frame(
                T_w2c,
                smpl_global_orient_precanon,
                smpl_transl_precanon,
                pelvis_offset=pelvis_offset,
                yaw_only=self.yaw_only_canon,
            )
            smpl["global_orient"], smpl["transl"] = canonicalize_smpl_first_frame(
                smpl["global_orient"], smpl["transl"], pelvis_offset=pelvis_offset,
                yaw_only=self.yaw_only_canon, up_axis=1, fwd_axis=2
            )

        # G1 FK is z-up internally.  Convert to VITH's y-up world and align
        # the robot's +x forward direction with the SMPL +z forward convention.
        for name in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
            g1[name] = apply_az_to_ay_on_vec(g1[name])
            g1[name] = _apply_ry_neg90_on_vec(g1[name])
        g1["body_quat_w"] = apply_az_to_ay_on_quat_wxyz(g1["body_quat_w"])
        g1["body_quat_w"] = _apply_ry_neg90_on_quat_wxyz(g1["body_quat_w"])
        if not self.canonicalize_first_frame:
            g1_root_first = g1["body_pos_w"][0, self.root_body_id].clone()
            smpl_transl_first = smpl["transl"][0].clone()
            g1["body_pos_w"] = g1["body_pos_w"] + (smpl_transl_first - g1_root_first)[None, None, :]

        if self.floor_adjust and self.canonicalize_first_frame:
            g1_floor_y = float(-g1["body_pos_w"][0, :, 1].min())
            g1["body_pos_w"][:, :, 1] += g1_floor_y
            with torch.no_grad():
                j24 = self.smplx_lite(
                    smpl["body_pose"][:1],
                    smpl["betas"][:1],
                    smpl["global_orient"][:1],
                    smpl["transl"][:1],
                )[0]
            smpl_floor_y = float(-j24[:, 1].min())
            smpl["transl"][:, 1] += smpl_floor_y
            floor_shift = torch.tensor([0.0, smpl_floor_y, 0.0], dtype=T_w2c.dtype, device=T_w2c.device)
            T_w2c[..., :3, 3] -= torch.einsum("fij,j->fi", T_w2c[..., :3, :3], floor_shift)

        return {
            "key": key,
            "length": length,
            "smpl": smpl,
            "T_w2c": T_w2c,
            "K_fullimg": K_fullimg,
            "bbx_xys": bbx_xys,
            "bbox_valid": bbox_valid,
            "kp2d": kp2d,
            "has_vitpose": bool(has_vitpose),
            "f_imgseq": f_imgseq,
            "has_imgfeat": bool(has_imgfeat),
            "g1": g1,
            "fps": fps,
            "meta": {
                "data_name": self.data_name,
                "idx": idx,
                "key": key,
                "camera_source": self.camera_source,
                "observation_frame_offset": self.observation_frame_offset,
                "camera_height_offset": bbox_raw.get(
                    "camera_height_offset", smplx_raw.get("camera_height_offset", 0.0)
                ),
                "mp4_path": bbox_raw.get("mp4_path", smplx_raw.get("mp4_path")),
                "camera_path": bbox_raw.get("camera_path", smplx_raw.get("camera_path")),
            },
        }

    def _process_data(self, data, idx):
        length = data["length"]
        smpl_w = data["smpl"]
        T_w2c = data["T_w2c"]
        gravity_vec = torch.tensor([0, -1, 0], dtype=torch.float32)
        R_c2gv = get_R_c2gv(T_w2c[:, :3, :3], gravity_vec)
        cam_angvel = compute_cam_angvel(T_w2c[:, :3, :3])

        offset = compute_smpl_pelvis_offset(
            smpl_w["betas"][0], self._J_template_pelvis, self._J_shapedirs_pelvis
        ).to(smpl_w["transl"])
        global_orient_c, transl_c = get_c_rootparam(
            smpl_w["global_orient"], smpl_w["transl"], T_w2c, offset
        )
        smpl_c = {
            "body_pose": smpl_w["body_pose"].clone(),
            "betas": smpl_w["betas"].clone(),
            "global_orient": global_orient_c,
            "transl": transl_c,
        }

        g1_target = {
            "g1_joint_pos": data["g1"]["joint_pos"],
            "g1_joint_vel": data["g1"]["joint_vel"],
            "g1_body_pos_w": data["g1"]["body_pos_w"],
            "g1_body_quat_w": data["g1"]["body_quat_w"],
            "g1_body_lin_vel_w": data["g1"]["body_lin_vel_w"],
            "g1_body_ang_vel_w": data["g1"]["body_ang_vel_w"],
        }
        g1_T_w2c = T_w2c.clone()
        g1_T_w2c[..., :3, 3] = 0.9 * T_w2c[..., :3, 3]
        g1_target["g1_T_w2c"] = g1_T_w2c

        out = {
            "meta": data["meta"],
            "length": length,
            "fps": torch.tensor(float(data["fps"]), dtype=torch.float32),
            "smpl_params_c": smpl_c,
            "smpl_params_w": smpl_w,
            "T_w2c": T_w2c,
            "R_c2gv": R_c2gv,
            "gravity_vec": gravity_vec,
            "bbx_xys": data["bbx_xys"],
            "K_fullimg": data["K_fullimg"],
            "f_imgseq": data["f_imgseq"],
            "kp2d": data["kp2d"],
            "cam_angvel": cam_angvel,
            "g1_target": g1_target,
            "mask": {
                "valid": data["bbox_valid"] & get_valid_mask(length, length),
                "vitpose": bool(data["has_vitpose"]),
                "bbx_xys": True,
                "f_imgseq": bool(data["has_imgfeat"]),
                "spv_incam_only": False,
            },
        }

        max_len = length if self.full_sequence else self.motion_frames
        valid = get_valid_mask(max_len, length)
        valid[:length] &= data["bbox_valid"]
        out["smpl_params_c"] = repeat_to_max_len_dict(out["smpl_params_c"], max_len)
        out["smpl_params_w"] = repeat_to_max_len_dict(out["smpl_params_w"], max_len)
        for name in ("T_w2c", "R_c2gv", "bbx_xys", "K_fullimg", "f_imgseq", "kp2d", "cam_angvel"):
            out[name] = repeat_to_max_len(out[name], max_len)
        out["mask"]["valid"] = valid
        for name, value in list(g1_target.items()):
            if isinstance(value, torch.Tensor) and value.dim() >= 1:
                g1_target[name] = repeat_to_max_len(value, max_len)
        return out


class BedlamG1Dataset(Bedlam2G1Dataset):
    """BEDLAM SMPL-X/G1 pairs with real video observations."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("dataset_name", "BEDLAM_G1")
        kwargs.setdefault("data_name", "bedlam_g1")
        kwargs.setdefault("camera_source", "bedlam_smpl")
        kwargs.setdefault("convert_T_w2c_to_yup", False)
        kwargs.setdefault("observation_frame_offset", 0)
        super().__init__(*args, **kwargs)


group_name = "train_datasets/imgfeat_bedlam"
MainStore.store(name="v2", node=builds(BedlamDatasetV2), group=group_name)
MainStore.store(name="v2_random1024", node=builds(BedlamDatasetV2, random1024=True), group=group_name)
MainStore.store(name="bedlam_g1", node=builds(BedlamG1Dataset), group="train_datasets")
MainStore.store(name="bedlam_g1", node=builds(BedlamG1Dataset), group="test_datasets")
MainStore.store(name="bedlam2_g1", node=builds(Bedlam2G1Dataset), group="train_datasets")
MainStore.store(name="bedlam2_g1", node=builds(Bedlam2G1Dataset), group="test_datasets")
