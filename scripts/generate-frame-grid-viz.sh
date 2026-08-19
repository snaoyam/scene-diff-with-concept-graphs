#!/usr/bin/env bash
# Backfills frame_grid_viz/ (the combined detected/filtered/fused/scenegraph grid,
# see conceptgraph/utils/combine_frame_viz.py) for ConceptGraph runs that already
# exist on disk, without re-running construct-concept-graphs.sh (step 1, the slow
# SLAM+detection step) or steps 2/3.
#
# Usage:
#   ./scripts/generate-frame-grid-viz.sh                # all pairs from scene-pairs.sh
#   ./scripts/generate-frame-grid-viz.sh <pair_name> ... # only the given pairs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/outputs"

# Same isolated-snapshot detection as run.sh: prefer the ./scripts/setup.sh
# snapshot+venv if present, so this uses the exact code/deps run.sh would use.
ISOLATED_RUN_DIR="$SCRIPT_DIR/../.isolated-runs"
if [[ -d "$ISOLATED_RUN_DIR/venv" ]]; then
    ISOLATED_RUN_DIR="$(cd "$ISOLATED_RUN_DIR" && pwd)"
    echo "isolated code snapshot detected at $ISOLATED_RUN_DIR -- using it (run ./scripts/setup.sh to refresh)"
    export PATH="$ISOLATED_RUN_DIR/venv/bin:$PATH"
    export CONCEPT_GRAPHS_ROOT="${CONCEPT_GRAPHS_ROOT:-$ISOLATED_RUN_DIR/concept-graphs}"
fi

CONCEPT_GRAPHS_DIR="$(cd "${CONCEPT_GRAPHS_ROOT:-$SCRIPT_DIR/../concept-graphs}" && pwd)"

generate_frame_grid_viz() {
    local pair_name="$1"
    local concept_graphs_dir="$OUTPUT_ROOT/$pair_name/concept_graphs"
    local ran_any=0

    for variant in before after; do
        local exp_out_path="$concept_graphs_dir/$variant/exps/r_mapping_pilot"
        if [[ ! -d "$exp_out_path" ]]; then
            continue
        fi
        ran_any=1
        echo "[$pair_name/$variant] generating frame_grid_viz"
        (cd "$CONCEPT_GRAPHS_DIR" && python -m conceptgraph.utils.combine_frame_viz "$exp_out_path")
    done

    if [[ "$ran_any" -eq 0 ]]; then
        echo "[$pair_name] no concept_graphs data found under $concept_graphs_dir, skipping"
    fi
}

if [[ $# -gt 0 ]]; then
    pairs=("$@")
else
    source "$SCRIPT_DIR/scene-pairs.sh"
    pairs=("${SCENE_PAIRS[@]}")
fi

for pair_name in "${pairs[@]}"; do
    generate_frame_grid_viz "$pair_name"
done
