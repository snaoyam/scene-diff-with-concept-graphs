#!/usr/bin/env bash
# Step 3: score one pair's step-2 object_masks.pkl against the official SceneDiff
# benchmark ground truth (scene_diff/scripts/evaluate_multiview.py, called directly for
# a single scene by run_scene_diff_benchmark.py).
#
# Usage:
#   run_scene_diff_benchmark <pair_name>
#   ./run-scene-diff-benchmark.sh <pair_name>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONCEPT_GRAPHS_SLAM_DIR="$(cd "${CONCEPT_GRAPHS_ROOT:-$SCRIPT_DIR/../concept-graphs}/conceptgraph/slam" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

run_scene_diff_benchmark() {
    local pair_name="$1"
    local output_root_arg=()
    [[ -n "$OUTPUT_ROOT" ]] && output_root_arg=(--output_root "$OUTPUT_ROOT")
    python "$CONCEPT_GRAPHS_SLAM_DIR/run_scene_diff_benchmark.py" \
        --pair_name "$pair_name" "${output_root_arg[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    if [[ $# -ne 1 ]]; then
        echo "Usage: $(basename "$0") <pair_name>" >&2
        exit 1
    fi
    run_scene_diff_benchmark "$1"
fi
