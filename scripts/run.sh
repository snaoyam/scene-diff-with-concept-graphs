#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/scene-pairs.sh"
source "$SCRIPT_DIR/construct-concept-graphs.sh"
source "$SCRIPT_DIR/convert-concept-graphs-to-scene-diff-benchmark-data.sh"
source "$SCRIPT_DIR/run-scene-diff-benchmark.sh"

failed_scenes=()

for scene_id in "${SCENE_PAIRS[@]}"; do
    # run
    # 1. construct-concept-graphs.sh
    # 2. convert-concept-graphs-to-scene-diff-benchmark-data.sh
    # 3. run-scene-diff-benchmark.sh
    echo "=== [$scene_id] 1/3 construct-concept-graphs ==="
    if ! construct_concept_graphs "$scene_id"; then
        failed_scenes+=("$scene_id (construct)")
        continue
    fi

    echo "=== [$scene_id] 2/3 convert-concept-graphs-to-scene-diff-benchmark-data ==="
    if ! convert_concept_graphs_to_scene_diff_benchmark_data "$scene_id"; then
        failed_scenes+=("$scene_id (convert)")
        continue
    fi

    echo "=== [$scene_id] 3/3 run-scene-diff-benchmark ==="
    if ! run_scene_diff_benchmark "$scene_id"; then
        failed_scenes+=("$scene_id (benchmark)")
        continue
    fi
done

echo
if [[ ${#failed_scenes[@]} -gt 0 ]]; then
    printf '  failed: %s\n' "${failed_scenes[*]}"
    exit 1
fi
