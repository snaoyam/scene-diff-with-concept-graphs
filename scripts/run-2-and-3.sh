#!/usr/bin/env bash
# Re-run steps 2+3 (convert + benchmark) for every pair in scene-pairs.sh, reusing each
# pair's existing step-1 concept_graphs/{before,after} output instead of regenerating it.
# For when only convert_concept_graphs_to_scene_diff_benchmark_data.py /
# run_scene_diff_benchmark.py changed and outputs/*/benchmark_result/eval_result.txt
# needs to be refreshed without re-running the (slow) ConceptGraph construction step.
#
# Usage: ./scripts/run-2-and-3.sh

CUDA_VISIBLE_DEVICES=6 #always only use GPU=6 and do not use any other GPU nodes
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
    export SCENE_DIFF_ROOT="${SCENE_DIFF_ROOT:-$ISOLATED_RUN_DIR/scene_diff}"
fi

source "$SCRIPT_DIR/scene-pairs.sh"
source "$SCRIPT_DIR/convert-concept-graphs-to-scene-diff-benchmark-data.sh"
source "$SCRIPT_DIR/run-scene-diff-benchmark.sh"

failed_scenes=()
skipped_scenes=()

for scene_id in "${SCENE_PAIRS[@]}"; do
    scene_output_dir="$OUTPUT_ROOT/$scene_id"
    concept_graphs_dir="$scene_output_dir/concept_graphs"

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

    # Regenerating regardless of what's already there -- that's the point of this script.
    rm -rf "$scene_output_dir/benchmark_data" "$scene_output_dir/benchmark_result"

    echo
    echo "=== running [$scene_id] (steps 2+3 only) ==="

    {
        echo "=== [$scene_id] 2/3 convert-concept-graphs-to-scene-diff-benchmark-data ==="
        if ! convert_concept_graphs_to_scene_diff_benchmark_data "$scene_id"; then
            echo "[$scene_id] failed"
            failed_scenes+=("$scene_id")
            continue
        fi

        echo "=== [$scene_id] 3/3 run-scene-diff-benchmark ==="
        if ! run_scene_diff_benchmark "$scene_id"; then
            echo "[$scene_id] failed"
            failed_scenes+=("$scene_id")
            continue
        fi
    } > "$scene_output_dir/terminal-outputs-$timestamp.txt" 2>&1

    echo "[$scene_id] complete"
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
