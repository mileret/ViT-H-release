#!/usr/bin/env python3
"""Run the G1 motion tracker on one NPZ file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vith_tracker import track_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-onnx", type=Path, required=True)
    parser.add_argument("--xml-path", type=Path, required=True)
    parser.add_argument("--grounding-xml", type=Path, required=True)
    parser.add_argument("--input-fps", type=float)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--ort-threads", type=int, default=1)
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Render the tracked motion to an MP4 file with MuJoCo.",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        help="Video path; defaults to the NPZ output path with an .mp4 suffix.",
    )
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
    result = track_file(
        input_path=args.input,
        output_path=output,
        policy_onnx=args.policy_onnx,
        xml_path=args.xml_path,
        grounding_xml=args.grounding_xml,
        input_fps=args.input_fps,
        control_hz=args.control_hz,
        ort_threads=args.ort_threads,
        render_video=args.render_video,
        video_output=args.video_output,
        video_width=args.video_width,
        video_height=args.video_height,
    )
    succeeded = int(result["is_success"].sum())
    total = int(result["is_success"].size)
    print(f"saved {output} ({succeeded}/{total} successful)")
    if args.render_video:
        video = args.video_output or output.with_suffix(".mp4")
        print(f"rendered video from {video.expanduser().resolve()}")


if __name__ == "__main__":
    main()
