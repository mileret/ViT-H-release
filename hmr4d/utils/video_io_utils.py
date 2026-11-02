import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm

from hmr4d.utils.video_timeline import (
    VIDEO_RESAMPLE_VERSION,
    select_frame_indices_by_timestamp,
)

def get_video_lwh(video_path):
    try:
        L, H, W, _ = iio.improps(video_path, plugin="pyav").shape
        if L <= 0 or W <= 0 or H <= 0:
            raise ValueError(f"Invalid video dimensions/frame count for {video_path}: {(L, W, H)}")
        return L, W, H
    except Exception as exc:
        if not _looks_like_missing_pyav(exc):
            raise
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}") from exc
        L = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if L <= 0 or W <= 0 or H <= 0:
            raise ValueError(f"Invalid video dimensions/frame count for {video_path}: {(L, W, H)}")
        return L, W, H


def probe_video_timeline(video_path):
    """Decode frame timestamps and return validated source timeline metadata."""
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "PyAV is required for timestamp-aware video preprocessing. "
            "Install the release requirements (av==13.0.0)."
        ) from exc

    video_path = str(video_path)
    timestamps = []
    last_frame_end = None
    packet_ends = {}
    stream_end = None
    average_fps = None
    width = height = None
    with av.open(video_path) as container:
        streams = list(container.streams.video)
        if not streams:
            raise ValueError(f"No video stream found: {video_path}")
        stream = streams[0]
        width, height = int(stream.width), int(stream.height)
        # Stream duration is in stream.time_base units, relative to start_time.
        # Container duration may include longer audio streams and is not usable.
        if stream.duration is not None and stream.start_time is not None and stream.time_base is not None:
            stream_end = (stream.start_time + stream.duration) * stream.time_base
        if stream.average_rate is not None:
            try:
                average_fps = float(stream.average_rate)
            except (TypeError, ValueError, ZeroDivisionError):
                average_fps = None
        # Demux also yields decoder-flush packets. Match by presentation time,
        # not packet order: codecs with B frames reorder their packets.
        for packet in container.demux(stream):
            if packet.pts is not None and packet.time_base is not None and packet.duration and packet.duration > 0:
                packet_time = packet.pts * packet.time_base
                packet_ends[packet_time] = packet_time + packet.duration * packet.time_base
            for frame in packet.decode():
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(
                        f"Video frame has no parseable timestamp (PTS/time_base): {video_path}"
                    )
                timestamp = frame.pts * frame.time_base
                if not np.isfinite(float(timestamp)):
                    raise ValueError(f"Non-finite frame timestamp in {video_path}")
                timestamps.append(timestamp)
                # PyAV 13 has no VideoFrame.duration. Never retain a previous
                # frame's duration when the final frame has no duration field.
                frame_duration = getattr(frame, "duration", None)
                last_frame_end = (
                    timestamp + frame_duration * frame.time_base
                    if frame_duration is not None and frame_duration > 0 else None
                )

    if not timestamps:
        raise ValueError(f"Video contains no decoded frames: {video_path}")
    first_timestamp, last_timestamp = timestamps[0], timestamps[-1]
    duration = None
    for duration_source, end in (
        ("frame", last_frame_end),
        ("packet", packet_ends.get(last_timestamp)),
        ("video_stream", stream_end),
    ):
        if end is not None and np.isfinite(float(end)) and end > last_timestamp:
            duration = float(end - first_timestamp)
            break
    if duration is None:
        raise ValueError(
            f"Cannot reliably determine final-frame duration from frame, packet, "
            f"or video-stream metadata: {video_path}"
        )
    timestamps = np.asarray([float(t - first_timestamp) for t in timestamps], dtype=np.float64)
    deltas = np.diff(timestamps)
    if deltas.size and (not np.isfinite(deltas).all() or np.any(deltas <= 0)):
        raise ValueError(f"Video timestamps are not strictly increasing: {video_path}")
    if width <= 0 or height <= 0 or duration <= 0:
        raise ValueError(
            f"Invalid decoded video metadata for {video_path}: "
            f"size={width}x{height}, duration={duration}"
        )
    intervals = np.append(deltas, duration - timestamps[-1])
    is_vfr = bool(
        intervals.size > 1
        and np.max(np.abs(intervals - np.median(intervals)))
        > max(1e-6, abs(float(np.median(intervals))) * 1e-3)
    )
    timestamp_hash = hashlib.sha256(
        json.dumps(timestamps.tolist(), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "timestamps": timestamps,
        "duration": duration,
        "duration_source": duration_source,
        "start_seconds": float(first_timestamp),
        "decoded_frames": int(timestamps.size),
        "width": width,
        "height": height,
        "average_fps": average_fps,
        "is_vfr": is_vfr,
        "timestamp_sha256": timestamp_hash,
    }


def resample_video_to_fps(source_path, output_path, target_fps=30.0, crf=23):
    """Create a CFR video by selecting source frames using decoded timestamps.

    The output is written atomically. A failed or interrupted conversion never
    replaces a previously complete staged video.
    """
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "PyAV is required for timestamp-aware video preprocessing. "
            "Install the release requirements (av==13.0.0)."
        ) from exc

    target_fps = float(target_fps)
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError(f"Invalid target FPS: {target_fps}")
    source_path = Path(source_path)
    output_path = Path(output_path)
    timeline = probe_video_timeline(source_path)
    selected = select_frame_indices_by_timestamp(
        timeline["timestamps"], timeline["duration"], target_fps
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.stem}.tmp.{os.getpid()}{output_path.suffix or '.mp4'}"
    )
    writer = None
    written = 0
    try:
        writer = get_writer(temp_path, fps=target_fps, crf=crf)
        next_output = 0
        with av.open(str(source_path)) as container:
            stream = container.streams.video[0]
            for source_index, frame in enumerate(container.decode(stream)):
                count = 0
                while next_output < len(selected) and int(selected[next_output]) == source_index:
                    count += 1
                    next_output += 1
                if count:
                    image = frame.to_ndarray(format="rgb24")
                    for _ in range(count):
                        writer.write_frame(image)
                        written += 1
        if next_output != len(selected) or written != len(selected):
            raise RuntimeError(
                f"Decoded frame count changed between timeline probe and resampling: "
                f"selected={len(selected)}, written={written}"
            )
        writer.close()
        writer = None
        os.replace(temp_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        temp_path.unlink(missing_ok=True)
        raise

    staged_length, staged_width, staged_height = get_video_lwh(output_path)
    if staged_length != written:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Staged video frame-count mismatch: encoded={written}, probed={staged_length}"
        )
    return {
        "strategy": VIDEO_RESAMPLE_VERSION,
        "source_decoded_frames": timeline["decoded_frames"],
        "source_average_fps": timeline["average_fps"],
        "source_duration_seconds": timeline["duration"],
        "source_duration_source": timeline["duration_source"],
        "source_start_seconds": timeline["start_seconds"],
        "source_is_vfr": timeline["is_vfr"],
        "source_timestamp_sha256": timeline["timestamp_sha256"],
        "target_fps": target_fps,
        "target_frames": written,
        "target_duration_seconds": written / target_fps,
        "width": staged_width,
        "height": staged_height,
    }


def _looks_like_missing_pyav(exc):
    msg = str(exc).lower()
    return "pyav" in msg or "plugin" in msg or "no module named 'av'" in msg


def _read_video_np_cv2(video_path, start_frame=0, end_frame=-1, scale=1.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_idx = start_frame
    while True:
        if end_frame != -1 and frame_idx >= end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        frames.append(frame[..., ::-1])
        frame_idx += 1
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")
    return np.stack(frames, axis=0)


def read_video_np(video_path, start_frame=0, end_frame=-1, scale=1.0):
    """
    Args:
        video_path: str
    Returns:
        frames: np.array, (N, H, W, 3) RGB, uint8
    """
    # If video path not exists, an error will be raised by ffmpegs
    filter_args = []
    should_check_length = False

    # 1. Trim
    if not (start_frame == 0 and end_frame == -1):
        if end_frame == -1:
            filter_args.append(("trim", f"start_frame={start_frame}"))
        else:
            should_check_length = True
            filter_args.append(("trim", f"start_frame={start_frame}:end_frame={end_frame}"))

    # 2. Scale
    if scale != 1.0:
        filter_args.append(("scale", f"iw*{scale}:ih*{scale}"))

    # Excute then check
    try:
        frames = iio.imread(video_path, plugin="pyav", filter_sequence=filter_args)
    except Exception as exc:
        if not _looks_like_missing_pyav(exc):
            raise
        frames = _read_video_np_cv2(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            scale=scale,
        )
    if should_check_length:
        assert len(frames) == end_frame - start_frame

    return frames


def get_video_reader(video_path):
    return iio.imiter(video_path, plugin="pyav")


def read_images_np(image_paths, verbose=False):
    """
    Args:
        image_paths: list of str
    Returns:
        images: np.array, (N, H, W, 3) RGB, uint8
    """
    if verbose:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in tqdm(image_paths)]
    else:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in image_paths]
    images = np.stack(images, axis=0)
    return images


def save_video(images, video_path, fps=30, crf=17):
    """
    Args:
        images: (N, H, W, 3) RGB, uint8
        crf: 17 is visually lossless, 23 is default, +6 results in half the bitrate
    0 is lossless, https://trac.ffmpeg.org/wiki/Encode/H.264#crf
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().astype(np.uint8)
    elif isinstance(images, list):
        images = np.array(images).astype(np.uint8)

    with iio.imopen(video_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        writer._video_stream.options = {"crf": str(crf)}
        writer.write(images)


def get_writer(video_path, fps=30, crf=17):
    """remember to .close()"""
    writer = iio.imopen(video_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)
    writer._video_stream.options = {"crf": str(crf)}
    return writer


def copy_file(video_path, out_video_path, overwrite=True):
    if not overwrite and Path(out_video_path).exists():
        return
    shutil.copy(video_path, out_video_path)


def merge_videos_horizontal(in_video_paths: list, out_video_path: str):
    import ffmpeg

    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "hstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    ffmpeg.run(output, overwrite_output=True, quiet=True)


def merge_videos_vertical(in_video_paths: list, out_video_path: str):
    import ffmpeg

    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "vstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    ffmpeg.run(output, overwrite_output=True, quiet=True)
