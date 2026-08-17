#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/outputs"

source "$SCRIPT_DIR/scene-pairs.sh"

for scene_id in "${SCENE_PAIRS[@]}"; do
    if [[ ! -f "$OUTPUT_ROOT/$scene_id/benchmark_result/eval_result.txt" ]]; then
        echo "$scene_id"
    fi
done
