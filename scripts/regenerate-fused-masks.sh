#!/usr/bin/env bash
# Regenerate fused_masks/ visualizations for every pair in scene-pairs.sh, reusing each
# pair's existing concept_graphs/{before,after} data instead of re-computing from scratch.
# For when only vis.py / write_progressive_fused_mask() changed and the visualization
# output needs to be refreshed without re-running the (slow) ConceptGraph construction step.
#
# Usage: ./scripts/regenerate-fused-masks.sh

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/outputs"
timestamp="$(date '+%Y-%m-%d_%H-%M-%S')"

# Same isolated-snapshot auto-detection as run.sh -- see that file for details.
ISOLATED_RUN_DIR="$SCRIPT_DIR/../.isolated-runs"
if [[ -d "$ISOLATED_RUN_DIR/venv" ]]; then
    ISOLATED_RUN_DIR="$(cd "$ISOLATED_RUN_DIR" && pwd)"
    echo "isolated code snapshot detected at $ISOLATED_RUN_DIR -- using it (run ./scripts/setup.sh to refresh)"
    export PATH="$ISOLATED_RUN_DIR/venv/bin:$PATH"
    export CONCEPT_GRAPHS_ROOT="${CONCEPT_GRAPHS_ROOT:-$ISOLATED_RUN_DIR/concept-graphs}"
fi

source "$SCRIPT_DIR/scene-pairs.sh"
source "$SCRIPT_DIR/construct-concept-graphs.sh"

failed_scenes=()
skipped_scenes=()

for scene_id in "${SCENE_PAIRS[@]}"; do
    scene_output_dir="$OUTPUT_ROOT/$scene_id"
    concept_graphs_dir="$scene_output_dir/concept_graphs"

    # Check that step 1 (concept_graphs) output exists
    missing_step1=0
    for variant in before after; do
        if [[ ! -f "$concept_graphs_dir/$variant/exps/r_mapping_pilot/pcd_r_mapping_pilot.pkl.gz" ]]; then
            missing_step1=1
        fi
    done
    if [[ "$missing_step1" -eq 1 ]]; then
        echo "[$scene_id] no step-1 concept_graphs output found, skipping (run construct-concept-graphs.sh first)"
        skipped_scenes+=("$scene_id")
        continue
    fi

    # Delete only fused_masks directories to force regeneration with new visualization code
    rm -rf "$concept_graphs_dir/before/exps/r_mapping_pilot/fused_masks" \
           "$concept_graphs_dir/after/exps/r_mapping_pilot/fused_masks"

    echo
    echo "=== running [$scene_id] (fused_masks regeneration only) ==="

    {
        if ! construct_concept_graphs "$scene_id"; then
            echo "[$scene_id] failed"
            failed_scenes+=("$scene_id")
            continue
        fi
    } > "$scene_output_dir/terminal-outputs-fused-masks-$timestamp.txt" 2>&1

    echo "[$scene_id] fused_masks regenerated"
    echo
done

echo
if [[ ${#skipped_scenes[@]} -gt 0 ]]; then
    printf '  skipped (no step-1 output): %s\n' "${skipped_scenes[*]}"
fi
if [[ ${#failed_scenes[@]} -gt 0 ]]; then
    printf '  failed: %s\n' "${failed_scenes[*]}"
    exit 1
fi
