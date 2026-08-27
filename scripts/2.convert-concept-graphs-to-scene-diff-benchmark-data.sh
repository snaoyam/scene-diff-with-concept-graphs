#!/usr/bin/env bash
# Step 2: match one pair's before/after ConceptGraphs and export object_masks.pkl in
# the format scene_diff/scripts/evaluate_multiview.py expects (see
# convert_concept_graphs_to_scene_diff_benchmark_data.py for the matching logic).
#
# Usage:
#   convert_concept_graphs_to_scene_diff_benchmark_data <pair_name> [extra CLI args...]
#   ./convert-concept-graphs-to-scene-diff-benchmark-data.sh <pair_name> [extra CLI args...]
#
# Extra args (e.g. --ablation_disable_confidence_visibility_filter -- see
# scripts/run_ablation.sh) are forwarded straight onto the underlying script's own argparse CLI.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONCEPT_GRAPHS_SLAM_DIR="$(cd "${CONCEPT_GRAPHS_ROOT:-$SCRIPT_DIR/../concept-graphs}/conceptgraph/slam" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

convert_concept_graphs_to_scene_diff_benchmark_data() {
    local pair_name="$1"
    shift
    local output_root_arg=()
    [[ -n "$OUTPUT_ROOT" ]] && output_root_arg=(--output_root "$OUTPUT_ROOT")
    python "$CONCEPT_GRAPHS_SLAM_DIR/convert_concept_graphs_to_scene_diff_benchmark_data.py" \
        --pair_name "$pair_name" "${output_root_arg[@]}" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    if [[ $# -lt 1 ]]; then
        echo "Usage: $(basename "$0") <pair_name> [extra CLI args...]" >&2
        exit 1
    fi
    convert_concept_graphs_to_scene_diff_benchmark_data "$@"
fi
