#!/usr/bin/env bash
set -uo pipefail

# Converts every SceneDiff benchmark scene pair under scenediff_benchmark/data
# into ConceptGraph-ready RGB-D sequences via scenediff_to_conceptgraph.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RAW_DIR="$PROJECT_ROOT/scene_diff/data/scenediff_benchmark/data"
OUT_DATASET_ROOT="$PROJECT_ROOT/Datasets/scenediff"
DATACONFIG_ROOT="$PROJECT_ROOT/concept-graphs/conceptgraph/dataset/dataconfigs/scenediff"

# One pair per line so individual scenes can be commented out.
SCENE_PAIRS=($(find "$RAW_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort))

for pair_name in "${SCENE_PAIRS[@]}"; do
    echo ""
    echo "=== pair=${pair_name} ==="
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python "$SCRIPT_DIR/scenediff_to_conceptgraph.py" \
        --raw_dir "$RAW_DIR" \
        --out_dataset_root "$OUT_DATASET_ROOT" \
        --dataconfig_root "$DATACONFIG_ROOT" \
        --pairs "$pair_name"
done
