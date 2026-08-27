#!/usr/bin/env bash
# Step 4: stitch the 8 per-frame debug/visualization images already produced for one
# pair's before/after scans (detected_masks, filtered_masks, fused_masks,
# fused_masks_with_nodes, scenegraph_viz, scenegraph_viz_with_edges, debug_masks, gt_mask_viz)
# into one lossless 2x4 grid image per frame -- see
# conceptgraph/utils/combine_scene_vis_grid.py for the layout and compositing logic.
#
# Usage:
#   combine_scene_vis_grid <pair_name>
#   ./combine-scene-vis-grid.sh <pair_name>
#
# GROUND_TRUTH_ROOT overrides where <pair>/gt_mask_viz/<before|after>/ is read from
# (default: <repo>/ground-truth -- same as scripts/visualize-gt-masks.sh's own default
# output root). Always computed and passed here rather than left for
# combine_scene_vis_grid.py to guess from its own file location: under the
# ./scripts/setup.sh isolated snapshot, that file lives one directory deeper
# (.isolated-runs/concept-graphs/...) than in the live checkout (concept-graphs/...),
# so counting parents from __file__ can't land on the right root in both cases --
# SCRIPT_DIR here is always the true repo's scripts/ (setup.sh never snapshots
# scripts/), so it always resolves correctly regardless of which code copy is running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONCEPT_GRAPHS_UTILS_DIR="$(cd "${CONCEPT_GRAPHS_ROOT:-$SCRIPT_DIR/../concept-graphs}/conceptgraph/utils" && pwd)"
# run.sh sets/exports this; standalone invocation falls back to the yaml's own default
# (rerun_realtime_mapping.yaml's output_root) by simply not passing an override.
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
GROUND_TRUTH_ROOT="${GROUND_TRUTH_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/ground-truth}"

combine_scene_vis_grid() {
    local pair_name="$1"
    local output_root_arg=()
    [[ -n "$OUTPUT_ROOT" ]] && output_root_arg=(--output_root "$OUTPUT_ROOT")
    python "$CONCEPT_GRAPHS_UTILS_DIR/combine_scene_vis_grid.py" \
        --pair_name "$pair_name" "${output_root_arg[@]}" --ground_truth_root "$GROUND_TRUTH_ROOT"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    if [[ $# -ne 1 ]]; then
        echo "Usage: $(basename "$0") <pair_name>" >&2
        exit 1
    fi
    combine_scene_vis_grid "$1"
fi
