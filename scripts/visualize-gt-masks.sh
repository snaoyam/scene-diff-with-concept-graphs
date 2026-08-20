#!/usr/bin/env bash
# Draws SceneDiff GT (segments.pkl) object masks onto video1/video2 frames, color-coded
# by change type: added=green, removed=red, moved=blue. Frames are subsampled by
# RESAMPLE_RATE the same way scene_diff/data/scenediff_to_conceptgraph.py subsamples when
# building the ConceptGraph Dataset, so gt_mask_viz/video{1,2}/frame_{i}.png lines up with
# Datasets/scenediff/<pair>/{before,after}/color/{i}.jpg.
# See scene_diff/scripts/visualize_gt_masks.py for the actual logic.
#
# Usage:
#   ./scripts/visualize-gt-masks.sh                # all pairs from scene-pairs.sh
#   ./scripts/visualize-gt-masks.sh <pair_name> ... # only the given pairs
#
# OUTPUT_ROOT overrides where gt_mask_viz/ is written per pair (default: ./outputs).
# RESAMPLE_RATE overrides the frame subsampling rate (default: 10, matching
# scenediff_to_conceptgraph.py's --resample_rate default used to build the Dataset).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_DIFF_DIR="$(cd "${SCENE_DIFF_ROOT:-$SCRIPT_DIR/../scene_diff}" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/ground-truth}"
RESAMPLE_RATE="${RESAMPLE_RATE:-10}"

# Same isolated-snapshot detection as run.sh / generate-frame-grid-viz.sh: prefer the
# ./scripts/setup.sh snapshot+venv if present, so this uses the exact code/deps run.sh would use.
ISOLATED_RUN_DIR="$SCRIPT_DIR/../.isolated-runs"
if [[ -d "$ISOLATED_RUN_DIR/venv" ]]; then
    ISOLATED_RUN_DIR="$(cd "$ISOLATED_RUN_DIR" && pwd)"
    echo "isolated code snapshot detected at $ISOLATED_RUN_DIR -- using it (run ./scripts/setup.sh to refresh)"
    export PATH="$ISOLATED_RUN_DIR/venv/bin:$PATH"
    export SCENE_DIFF_ROOT="${SCENE_DIFF_ROOT:-$ISOLATED_RUN_DIR/scene_diff}"
    SCENE_DIFF_DIR="$(cd "$SCENE_DIFF_ROOT" && pwd)"
fi

visualize_gt_masks() {
    local pair_name="$1"
    python "$SCENE_DIFF_DIR/scripts/visualize_gt_masks.py" \
        --pair_name "$pair_name" \
        --data_root "$SCENE_DIFF_DIR/data/scenediff_benchmark/data" \
        --output_root "$OUTPUT_ROOT" \
        --resample_rate "$RESAMPLE_RATE"
}

if [[ $# -gt 0 ]]; then
    pairs=("$@")
else
    source "$SCRIPT_DIR/scene-pairs.sh"
    pairs=("${SCENE_PAIRS[@]}")
fi

failed_scenes=()
for pair_name in "${pairs[@]}"; do
    echo "=== [$pair_name] ==="
    if ! visualize_gt_masks "$pair_name"; then
        echo "[$pair_name] failed"
        failed_scenes+=("$pair_name")
    fi
    echo
done

if [[ ${#failed_scenes[@]} -gt 0 ]]; then
    echo "failed: ${failed_scenes[*]}"
    exit 1
fi
