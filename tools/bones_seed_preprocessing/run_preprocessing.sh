#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash run_preprocessing.sh /absolute/path/to/bones-seed /absolute/path/to/output [workers]" >&2
    exit 2
fi
RAW_ROOT=$(realpath "$1")
OUTPUT_ROOT=$(realpath -m "$2")
WORKERS=${3:-16}
BUNDLE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SONIC_ROOT="$BUNDLE_ROOT/GR00T-WholeBodyControl"
PROCESS_ROOT="$SONIC_ROOT/gear_sonic/data_process"

python "$PROCESS_ROOT/extract_soma_joints_with_eyes_global_from_bvh.py" \
    --input "$RAW_ROOT/soma_proportional/bvh" --output "$OUTPUT_ROOT/soma" \
    --fps 30 --num_workers "$WORKERS" --skip_existing

python "$PROCESS_ROOT/filter_and_copy_bones_data.py" \
    --source "$OUTPUT_ROOT/soma" --dest "$OUTPUT_ROOT/soma_filtered" --workers "$WORKERS"

python "$PROCESS_ROOT/convert_soma_csv_to_motion_lib.py" \
    --input "$RAW_ROOT/g1/csv" --output "$OUTPUT_ROOT/robot" \
    --fps 30 --fps_source 120 --individual --num_workers "$WORKERS"

python "$PROCESS_ROOT/filter_and_copy_bones_data.py" \
    --source "$OUTPUT_ROOT/robot" --dest "$OUTPUT_ROOT/robot_filtered" --workers "$WORKERS"

python "$BUNDLE_ROOT/src/humanoid_recon/data_process/export_sonic_paired_motions.py" \
    --sonic-root "$SONIC_ROOT" --motion-file "$OUTPUT_ROOT/robot_filtered" \
    --soma-motion-file "$OUTPUT_ROOT/soma_filtered" --output-dir "$OUTPUT_ROOT/paired" \
    --target-fps 30
