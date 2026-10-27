import torch


# Current ViT-H diffusion training builds obs from `supermotion_v437coco17`.
# COCO17 order:
#   0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
#   5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
#   9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
#   13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle.
# Drop only nose/left_ear/right_ear, so eyes are kept.
COCO17_NO_NOSE_EARS_IDS = (1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
COCO17_NO_NOSE_EARS_NUM_JOINTS = len(COCO17_NO_NOSE_EARS_IDS)


def select_coco17_no_nose_ears(obs: torch.Tensor) -> torch.Tensor:
    """Select the 14-joint obs order used by diffusion conditioning.

    Accepts either full COCO17 obs (..., 17, C) or already-selected obs
    (..., 14, C). The confidence/mask channel, when present, is preserved.
    """
    num_joints = obs.shape[-2]
    if num_joints == COCO17_NO_NOSE_EARS_NUM_JOINTS:
        return obs
    if num_joints != 17:
        raise ValueError(
            f"Expected COCO17 obs with 17 joints or selected obs with "
            f"{COCO17_NO_NOSE_EARS_NUM_JOINTS} joints, got {num_joints}."
        )
    ids = torch.as_tensor(COCO17_NO_NOSE_EARS_IDS, device=obs.device)
    return obs.index_select(-2, ids)
