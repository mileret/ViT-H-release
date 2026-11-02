"""Dependency-light helpers for validating and resampling video timelines."""

import numpy as np


VIDEO_RESAMPLE_VERSION = "pyav-timestamp-nearest-v1"


def select_frame_indices_by_timestamp(timestamps, duration, target_fps):
    """Select nearest decoded frames on a uniform target time grid."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    target_fps = float(target_fps)
    duration = float(duration)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError("Video contains no timestamped frames.")
    if not np.isfinite(timestamps).all() or not np.isfinite(duration):
        raise ValueError("Video contains non-finite timestamps or duration.")
    if target_fps <= 0 or duration <= 0:
        raise ValueError(f"Invalid target_fps/duration: {target_fps}, {duration}")
    if abs(float(timestamps[0])) > 1e-6:
        raise ValueError("timestamps must be relative to the first frame (start at zero).")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("Video frame timestamps must be strictly increasing.")
    if duration <= float(timestamps[-1]):
        raise ValueError(
            f"Video duration ({duration:.9f}) must extend beyond its last frame "
            f"timestamp ({timestamps[-1]:.9f})."
        )

    target_count = max(1, int(np.ceil(duration * target_fps - 1e-9)))
    target_times = np.arange(target_count, dtype=np.float64) / target_fps
    midpoints = (timestamps[:-1] + timestamps[1:]) * 0.5
    return np.searchsorted(midpoints, target_times, side="right").astype(np.int64)
