#!/usr/bin/env bash
# Step 2: match one pair's before/after ConceptGraphs and export object_masks.pkl in
# the format scene_diff/scripts/evaluate_multiview.py expects (see
# convert_concept_graphs_to_scene_diff_benchmark_data.py for the matching logic).
#
# Usage:
#   convert_concept_graphs_to_scene_diff_benchmark_data <pair_name>
#   ./convert-concept-graphs-to-scene-diff-benchmark-data.sh <pair_name>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

convert_concept_graphs_to_scene_diff_benchmark_data() {
    local pair_name="$1"
    conda run -n scene_diff --no-capture-output \
        python "$SCRIPT_DIR/convert_concept_graphs_to_scene_diff_benchmark_data.py" \
            --pair_name "$pair_name"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    if [[ $# -ne 1 ]]; then
        echo "Usage: $(basename "$0") <pair_name>" >&2
        exit 1
    fi
    convert_concept_graphs_to_scene_diff_benchmark_data "$1"
fi
