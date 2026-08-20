#!/usr/bin/env bash
# Regenerate fused_masks/, fused_masks_with_nodes/, scenegraph_viz/, and
# scenegraph_viz_with_edges/ for scenes that have already been fully processed
# (i.e. have benchmark_result/eval_result.txt), reusing each scene's existing
# concept_graphs/{before,after} detections instead of re-running detection from
# scratch. For when only write_progressive_fused_mask() (rerun_realtime_mapping.py)
# or scenegraph_viz.py changed and these four visualization outputs need to be
# refreshed without re-running the (slow) YOLO/SAM/VLM detection step or steps 2/3
# (convert-concept-graphs-to-scene-diff-benchmark-data.sh, run-scene-diff-benchmark.sh)
# -- those are untouched by a pure-visualization change.
#
# This still re-runs the (fast, CPU-only) per-frame matching/merging loop in
# rerun_realtime_mapping.py, because fused_masks/fused_masks_with_nodes are written
# progressively as that loop runs (not derivable from the final saved pcd alone) --
# see write_progressive_fused_mask()'s docstring. It skips only the slow model
# inference (YOLO/SAM/VLM/CLIP/DINO) part, via the same cached-detection reuse
# check_run_detections() already uses whenever concept_graphs/<variant>/exps/.../
# detections/ exists on disk from the prior run.
#
# Usage:
#   ./scripts/regenerate-fused-and-scenegraph-viz.sh                # all pairs from scene-pairs.sh
#   ./scripts/regenerate-fused-and-scenegraph-viz.sh <pair_name> ... # only the given pairs

set -euo pipefail

CUDA_VISIBLE_DEVICES=6 #always only use GPU=6 and do not use any other GPU nodes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="$(date '+%Y-%m-%d_%H-%M-%S')"

# Same ISOLATED_RUN/OUTPUT_ROOT convention as run.sh -- see that file for details.
# Defaults to isolated-outputs/ since that's where already-completed scenes (the
# eval_result.txt this script gates on) live under the normal run.sh workflow.
ISOLATED_RUN="${ISOLATED_RUN:-1}"
if [[ "$ISOLATED_RUN" == "0" ]]; then
    OUTPUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/outputs"
else
    OUTPUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/isolated-outputs"
fi

ISOLATED_RUN_DIR="$SCRIPT_DIR/../.isolated-runs"
if [[ "$ISOLATED_RUN" == "0" ]]; then
    echo "ISOLATED_RUN=0 -- skipping isolated snapshot, using original environment"
elif [[ -d "$ISOLATED_RUN_DIR/venv" ]]; then
    ISOLATED_RUN_DIR="$(cd "$ISOLATED_RUN_DIR" && pwd)"
    echo "isolated code snapshot detected at $ISOLATED_RUN_DIR -- using it (run ./scripts/setup.sh to refresh)"
    export PATH="$ISOLATED_RUN_DIR/venv/bin:$PATH"
    export CONCEPT_GRAPHS_ROOT="${CONCEPT_GRAPHS_ROOT:-$ISOLATED_RUN_DIR/concept-graphs}"
fi

source "$SCRIPT_DIR/construct-concept-graphs.sh"

if [[ $# -gt 0 ]]; then
    pairs=("$@")
else
    source "$SCRIPT_DIR/scene-pairs.sh"
    pairs=("${SCENE_PAIRS[@]}")
fi

failed_scenes=()
skipped_scenes=()

for scene_id in "${pairs[@]}"; do
    scene_output_dir="$OUTPUT_ROOT/$scene_id"
    concept_graphs_dir="$scene_output_dir/concept_graphs"

    if [[ ! -f "$scene_output_dir/benchmark_result/eval_result.txt" ]]; then
        echo "[$scene_id] no benchmark_result/eval_result.txt found, skipping"
        skipped_scenes+=("$scene_id")
        continue
    fi

    missing_step1=0
    for variant in before after; do
        if [[ ! -f "$concept_graphs_dir/$variant/exps/r_mapping_pilot/pcd_r_mapping_pilot.pkl.gz" ]]; then
            missing_step1=1
        fi
    done
    if [[ "$missing_step1" -eq 1 ]]; then
        echo "[$scene_id] eval_result.txt exists but step-1 concept_graphs output is missing/incomplete, skipping"
        skipped_scenes+=("$scene_id")
        continue
    fi

    # Delete only the four viz directories being regenerated -- everything else
    # (detected_masks/, filtered_masks/, the pcd itself, benchmark_data/,
    # benchmark_result/, ...) is left untouched. Deleting first (rather than
    # relying on the frame loop to overwrite same-named files) avoids stale
    # leftovers from the old fused_masks/scenegraph_viz composite format if this
    # run somehow produces fewer frames than the one that wrote them.
    for variant in before after; do
        exp_out_path="$concept_graphs_dir/$variant/exps/r_mapping_pilot"
        rm -rf "$exp_out_path/fused_masks" \
               "$exp_out_path/fused_masks_with_nodes" \
               "$exp_out_path/scenegraph_viz" \
               "$exp_out_path/scenegraph_viz_with_edges"
    done

    echo
    echo "=== running [$scene_id] (fused_masks/fused_masks_with_nodes/scenegraph_viz/scenegraph_viz_with_edges regeneration only) ==="

    {
        if ! construct_concept_graphs "$scene_id"; then
            echo "[$scene_id] failed"
            failed_scenes+=("$scene_id")
            continue
        fi
    } > "$scene_output_dir/terminal-outputs-viz-regen-$timestamp.txt" 2>&1

    echo "[$scene_id] viz regenerated"
    echo
done

echo
if [[ ${#skipped_scenes[@]} -gt 0 ]]; then
    printf '  skipped (no eval_result.txt / incomplete step-1 output): %s\n' "${skipped_scenes[*]}"
fi
if [[ ${#failed_scenes[@]} -gt 0 ]]; then
    printf '  failed: %s\n' "${failed_scenes[*]}"
    exit 1
fi
