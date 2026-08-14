#!/usr/bin/env bash
set -uo pipefail

# Runs the SceneDiff algorithm (predict_multiview.py) and the official benchmark
# scoring (evaluate_multiview.py) for one or more named scene pairs from the
# SceneDiff benchmark, one scene at a time. Split (val/test) and set
# (varied/kitchen) are auto-detected per scene from splits/*.json.
#
# Usage:
#   ./run.sh SCENE_ID [SCENE_ID ...]
#   ./run.sh                          # uses SCENE_PAIRS below
#
# Examples:
#   ./run.sh living_room_17_living_room_18
#   CUDA_VISIBLE_DEVICES=2 ./run.sh living_room_17_living_room_18 bed_3_bed_4
#
# Predictions accumulate in $OUTPUT_DIR/<scene_id>/object_masks.pkl (so a
# scene already predicted is skipped on rerun -- see predict_multiview.py's
# filter_already_processed_scenes). Each scene is still evaluated in
# isolation: the eval step only ever scores the scene just run, regardless
# of what else has accumulated in $OUTPUT_DIR.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/node_data/urp26su_dongwoo/anaconda3/envs/scene_diff/bin/python}"
# GPU 0 is often saturated by other users' jobs on this shared node -- check
# `nvidia-smi` and override if needed, e.g. `CUDA_VISIBLE_DEVICES=2 ./run.sh ...`.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG="${CONFIG:-configs/scenediff_config.yml}"
OUTPUT_DIR="${OUTPUT_DIR:-output/scenediff_benchmark}"

# One scene pair per line so individual scenes can be commented out.
# SCENE_PAIRS=(
#     living_room_17_living_room_18
#     bathroom_1_bathroom_2
#     bed_3_bed_4
#     bus_1_bus_2
#     coffee_table_1_coffee_table_2
# )
source "/node_data/urp26su_dongwoo/concept-graphs-project/scene_pairs.sh"

# SCENE_PAIRS=("${SCENE_PAIRS[@]:0:50}")
REVERSED_SCENE_PAIRS=()
for (( i=${#SCENE_PAIRS[@]}-1; i>=0; i-- )); do
    REVERSED_SCENE_PAIRS+=("${SCENE_PAIRS[i]}")
done
SCENE_PAIRS=("${REVERSED_SCENE_PAIRS[@]}")

if [[ $# -gt 0 ]]; then
    SCENE_PAIRS=("$@")
fi


TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir -p "$OUTPUT_DIR"

for scene_id in "${SCENE_PAIRS[@]}"; do
    echo ""
    echo "=== scene_id=${scene_id} ==="

    if compgen -G "$OUTPUT_DIR/$scene_id/eval_result_frame_dup_*_obj_dup_*.txt" > /dev/null; then
        echo "SKIP: already mapped (${scene_id})."
        continue
    fi

    read -r split_name set_name < <(
        "$PYTHON" - "$scene_id" <<'PYEOF'
import json, sys
scene_id = sys.argv[1]
for split_name, fname in (("val", "val_split.json"), ("test", "test_split.json")):
    d = json.load(open(f"data/scenediff_benchmark/splits/{fname}"))
    for set_name, lst in d.items():
        if scene_id in lst:
            print(split_name, set_name)
            sys.exit(0)
sys.exit(1)
PYEOF
    ) || { echo "  '$scene_id' not found in val_split.json or test_split.json, skipping" >&2; continue; }
    echo "  split=$split_name set=$set_name"

    # Single-scene split file + config override so predict_multiview.py only
    # touches this one scene (it otherwise processes a whole split/set).
    scene_split_json="$TMP_DIR/${scene_id}_split.json"
    scene_config="$TMP_DIR/${scene_id}_config.yml"
    "$PYTHON" - "$scene_id" "$split_name" "$set_name" "$CONFIG" "$scene_split_json" "$scene_config" <<'PYEOF'
import sys, json, yaml
scene_id, split_name, set_name, config_path, split_json_path, out_config_path = sys.argv[1:]

json.dump({"varied": [], "kitchen": [], set_name: [scene_id]}, open(split_json_path, "w"))

cfg = yaml.safe_load(open(config_path))
cfg["dataset"]["splits"][split_name] = split_json_path
yaml.safe_dump(cfg, open(out_config_path, "w"))
PYEOF

    echo "  --- predict ---"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" scripts/predict_multiview.py \
        --config "$scene_config" \
        --splits "$split_name" --sets "$set_name" \
        --output_dir "$OUTPUT_DIR" || { echo "  predict failed for $scene_id" >&2; continue; }

    if [[ ! -f "$OUTPUT_DIR/$scene_id/object_masks.pkl" ]]; then
        echo "  no object_masks.pkl produced for $scene_id, skipping eval" >&2
        continue
    fi

    # Isolate eval to just this scene via a scratch dir of one symlink, so
    # results aren't silently averaged in with other scenes sitting in
    # $OUTPUT_DIR from earlier runs.
    eval_pred_dir="$TMP_DIR/${scene_id}_pred"
    mkdir -p "$eval_pred_dir"
    ln -sfn "$(cd "$OUTPUT_DIR/$scene_id" && pwd)" "$eval_pred_dir/$scene_id"

    echo "  --- evaluate ---"
    mkdir -p "$OUTPUT_DIR/$scene_id"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" scripts/evaluate_multiview.py \
        --pred_dir "$eval_pred_dir" \
        --splits "$split_name" --sets "$set_name" \
        --output_path "$OUTPUT_DIR/$scene_id/eval_result.txt" \
        --visualize False

    result_file=$(ls "$OUTPUT_DIR/$scene_id/eval_result"_frame_dup_*_obj_dup_*.txt 2>/dev/null | tail -n1)
    echo "  predictions:  $OUTPUT_DIR/$scene_id/object_masks.pkl"
    echo "  eval result:  ${result_file:-$OUTPUT_DIR/$scene_id/eval_result.txt}"
done
