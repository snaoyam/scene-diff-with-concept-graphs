#!/usr/bin/env bash
# Step 1: build the before/after ConceptGraphs for one SceneDiff pair.
#
# Meant to be sourced (defines construct_concept_graphs, does nothing else) or run
# directly (python conceptgraph/slam/rerun_realtime_mapping.py scene_pair=<pair> handles
# building both the "before" and "after" graphs in one call -- see
# conceptgraph/slam/rerun_realtime_mapping.py's main()).
#
# Usage:
#   construct_concept_graphs <pair_name>
#   ./construct-concept-graphs.sh <pair_name>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONCEPT_GRAPHS_DIR="$(cd "${CONCEPT_GRAPHS_ROOT:-$SCRIPT_DIR/../concept-graphs}" && pwd)"

construct_concept_graphs() {
    local pair_name="$1"
    (cd "$CONCEPT_GRAPHS_DIR" && \
        python conceptgraph/slam/rerun_realtime_mapping.py "scene_pair=${pair_name}")
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    if [[ $# -ne 1 ]]; then
        echo "Usage: $(basename "$0") <pair_name>" >&2
        exit 1
    fi
    construct_concept_graphs "$1"
fi
