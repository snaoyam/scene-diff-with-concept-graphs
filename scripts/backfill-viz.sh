#!/usr/bin/env bash
# Scans isolated-outputs/ (or outputs/ with ISOLATED_RUN=0) for scenes whose
# regenerate-fused-and-scenegraph-viz.sh and/or combine-scene-vis-grid.sh output is
# missing or incomplete, and runs only those two scripts for only the scenes that
# actually need them -- so it's safe to re-run any time (after either script's
# underlying logic changes, or after new scenes finish steps 1-3) without wastefully
# re-running either for scenes whose output is already complete.
#
# "Needs regenerate-fused-and-scenegraph-viz.sh": the scene has eval_result.txt and a
# step-1 pcd for both variants (regenerate-fused-and-scenegraph-viz.sh's own
# prerequisite), and for either variant, any of fused_masks/, fused_masks_with_nodes/,
# scenegraph_viz/, scenegraph_viz_with_edges/ is missing or has fewer files than
# detected_masks/ -- detected_masks/ is written unconditionally for every frame
# (rerun_realtime_mapping.py's main loop, before any filtering/caching), so it's the
# reference frame count for "how many frames this scan actually has".
#
# "Needs combine-scene-vis-grid.sh": the scene has eval_result.txt, and for either
# variant, vis/<variant>/ is missing or has fewer files than
# benchmark_data/debug_masks/<variant>/ -- debug_masks/ always covers every frame (see
# save_debug_mask_visualizations in convert_concept_graphs_to_scene_diff_benchmark_data.py),
# so it's the reference frame count for step 4.
#
# regenerate-fused-and-scenegraph-viz.sh runs first (batched into one call covering
# every scene that needs it) so any scene it touches gets combine-scene-vis-grid.sh run
# against the freshly-regenerated fused_masks/scenegraph_viz afterwards, not stale ones
# -- "needs combine-scene-vis-grid.sh" is (re)checked after regeneration finishes.
#
# Usage:
#   ./scripts/backfill-viz.sh                 # scans isolated-outputs/ (default)
#   ISOLATED_RUN=0 ./scripts/backfill-viz.sh   # scans outputs/ instead

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same OUTPUT_ROOT convention as run.sh / regenerate-fused-and-scenegraph-viz.sh, so
# this scans exactly the tree those two scripts themselves read from and write to.
ISOLATED_RUN="${ISOLATED_RUN:-1}"
export ISOLATED_RUN
if [[ "$ISOLATED_RUN" == "0" ]]; then
    OUTPUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/outputs"
else
    OUTPUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/isolated-outputs"
fi

# Same isolated-snapshot detection as run.sh: prefer the ./scripts/setup.sh
# snapshot+venv if present, so this uses the exact code/deps run.sh would use.
ISOLATED_RUN_DIR="$SCRIPT_DIR/../.isolated-runs"
if [[ "$ISOLATED_RUN" != "0" && -d "$ISOLATED_RUN_DIR/venv" ]]; then
    ISOLATED_RUN_DIR="$(cd "$ISOLATED_RUN_DIR" && pwd)"
    echo "isolated code snapshot detected at $ISOLATED_RUN_DIR -- using it (run ./scripts/setup.sh to refresh)"
    export PATH="$ISOLATED_RUN_DIR/venv/bin:$PATH"
    export CONCEPT_GRAPHS_ROOT="${CONCEPT_GRAPHS_ROOT:-$ISOLATED_RUN_DIR/concept-graphs}"
    export SCENE_DIFF_ROOT="${SCENE_DIFF_ROOT:-$ISOLATED_RUN_DIR/scene_diff}"
fi

_count_files() {  # $1 = dir
    [[ -d "$1" ]] || { echo 0; return; }
    find "$1" -maxdepth 1 -type f | wc -l
}

_scene_eligible() {  # $1 = scene_id -- step-1 output present for both variants
    local scene_output_dir="$OUTPUT_ROOT/$1"
    local variant
    for variant in before after; do
        [[ -f "$scene_output_dir/concept_graphs/$variant/exps/r_mapping_pilot/pcd_r_mapping_pilot.pkl.gz" ]] || return 1
    done
    return 0
}

_needs_regen() {  # $1 = scene_id
    local scene_output_dir="$OUTPUT_ROOT/$1"
    local variant exp_out_path ref_count d
    for variant in before after; do
        exp_out_path="$scene_output_dir/concept_graphs/$variant/exps/r_mapping_pilot"
        ref_count="$(_count_files "$exp_out_path/detected_masks")"
        [[ "$ref_count" -eq 0 ]] && continue
        for d in fused_masks fused_masks_with_nodes scenegraph_viz scenegraph_viz_with_edges; do
            [[ "$(_count_files "$exp_out_path/$d")" -lt "$ref_count" ]] && return 0
        done
    done
    return 1
}

_needs_vis_grid() {  # $1 = scene_id
    local scene_output_dir="$OUTPUT_ROOT/$1"
    local variant ref_count
    for variant in before after; do
        ref_count="$(_count_files "$scene_output_dir/benchmark_data/debug_masks/$variant")"
        [[ "$ref_count" -eq 0 ]] && continue
        [[ "$(_count_files "$scene_output_dir/vis/$variant")" -lt "$ref_count" ]] && return 0
    done
    return 1
}

if [[ ! -d "$OUTPUT_ROOT" ]]; then
    echo "OUTPUT_ROOT $OUTPUT_ROOT does not exist" >&2
    exit 1
fi

scene_ids=()
for d in "$OUTPUT_ROOT"/*/; do
    [[ -d "$d" ]] || continue
    scene_ids+=("$(basename "$d")")
done
echo "scanning ${#scene_ids[@]} scene(s) under $OUTPUT_ROOT"

regen_needed=()
for scene_id in "${scene_ids[@]}"; do
    if [[ -f "$OUTPUT_ROOT/$scene_id/benchmark_result/eval_result.txt" ]] \
        && _scene_eligible "$scene_id" && _needs_regen "$scene_id"; then
        regen_needed+=("$scene_id")
    fi
done

# if [[ ${#regen_needed[@]} -gt 0 ]]; then
#     echo
#     echo "=== regenerate-fused-and-scenegraph-viz needed for: ${regen_needed[*]} ==="
#     "$SCRIPT_DIR/regenerate-fused-and-scenegraph-viz.sh" "${regen_needed[@]}"
#     regen_status=$?
# else
#     echo "no scene needs regenerate-fused-and-scenegraph-viz.sh"
#     regen_status=0
# fi

regen_status=0

source "$SCRIPT_DIR/combine-scene-vis-grid.sh"

vis_needed=()
for scene_id in "${scene_ids[@]}"; do
    vis_needed+=("$scene_id")
done

vis_failed=()
if [[ ${#vis_needed[@]} -gt 0 ]]; then
    echo
    echo "=== combine-scene-vis-grid needed for: ${vis_needed[*]} ==="
    for scene_id in "${vis_needed[@]}"; do
        echo "--- [$scene_id] combine-scene-vis-grid ---"
        if ! combine_scene_vis_grid "$scene_id"; then
            echo "[$scene_id] failed"
            vis_failed+=("$scene_id")
        fi
    done
else
    echo "no scene needs combine-scene-vis-grid.sh"
fi

echo
if [[ "$regen_status" -ne 0 || ${#vis_failed[@]} -gt 0 ]]; then
    [[ ${#vis_failed[@]} -gt 0 ]] && printf 'combine-scene-vis-grid failed: %s\n' "${vis_failed[*]}"
    [[ "$regen_status" -ne 0 ]] && echo "regenerate-fused-and-scenegraph-viz.sh exited non-zero"
    exit 1
fi
