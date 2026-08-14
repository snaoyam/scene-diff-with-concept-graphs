#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/scene-pairs.sh"

failed_scenes=()

for scene_id in "${SCENE_PAIRS[@]}"; do
    # run 
    # 1. construct-concept-graphs.sh
    # 2. convert-concept-graphs-to-scene-diff-benchmark-data.sh
    # 3. run-scene-diff-benchmark.sh
done

echo
if [[ ${#failed_scenes[@]} -gt 0 ]]; then
    printf '  failed: %s\n' "${failed_scenes[*]}"
    exit 1
fi
