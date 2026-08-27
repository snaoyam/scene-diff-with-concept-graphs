#!/usr/bin/env bash
#activate conceptgraph env (/node_data/urp26su_dongwoo/concept-graphs-project/environment.yml)
conda activate conceptgraph

CUDA_VISIBLE_DEVICES=6 #always only use GPU=6 and do not use any other GPU nodes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="$(date '+%Y-%m-%d_%H-%M-%S')"

OUTPUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/outputs"

source "$SCRIPT_DIR/scene-pairs.sh"
source "$SCRIPT_DIR/1.construct-concept-graphs.sh"
source "$SCRIPT_DIR/2.convert-concept-graphs-to-scene-diff-benchmark-data.sh"
source "$SCRIPT_DIR/3.run-scene-diff-benchmark.sh"
source "$SCRIPT_DIR/4.combine-scene-vis-grid.sh"

failed_scenes=()

for scene_id in "${SCENE_PAIRS[@]}"; do
    scene_output_dir="$OUTPUT_ROOT/$scene_id"

    if [[ -f "$scene_output_dir/benchmark_result/eval_result.txt" ]]; then
        echo "[$scene_id] eval_result.txt already exists, skipping"
        continue
    fi

    # 이전 실행이 중간에 실패해 남은 미완료 폴더는 삭제하고 처음부터 다시 실행
    concept_graphs_dir="$scene_output_dir/concept_graphs"
    for variant in before after; do
        variant_pcd="$concept_graphs_dir/$variant/exps/r_mapping_pilot/pcd_r_mapping_pilot.pkl.gz"
        if [[ -d "$concept_graphs_dir/$variant" && ! -f "$variant_pcd" ]]; then
            echo "[$scene_id] removing incomplete concept_graphs/$variant"
            rm -rf "$concept_graphs_dir/$variant"
        fi
    done

    if [[ -d "$scene_output_dir/benchmark_data" && ! -f "$scene_output_dir/benchmark_data/object_masks.pkl" ]]; then
        echo "[$scene_id] removing incomplete benchmark_data"
        rm -rf "$scene_output_dir/benchmark_data"
    fi

    if [[ -d "$scene_output_dir/benchmark_result" ]]; then
        echo "[$scene_id] removing incomplete benchmark_result"
        rm -rf "$scene_output_dir/benchmark_result"
    fi

    echo
    echo "=== running [$scene_id] ==="
    mkdir -p "$scene_output_dir"

    {
        # run
        # 1. construct-concept-graphs.sh
        # 2. convert-concept-graphs-to-scene-diff-benchmark-data.sh
        # 3. run-scene-diff-benchmark.sh
        # 4. combine-scene-vis-grid.sh
        echo "=== [$scene_id] 1/4 construct-concept-graphs ==="
        if ! construct_concept_graphs "$scene_id"; then
            echo "[$scene_id] failed"
            failed_scenes+=("$scene_id")
            continue
        fi

        echo "=== [$scene_id] 2/4 convert-concept-graphs-to-scene-diff-benchmark-data ==="
        if ! convert_concept_graphs_to_scene_diff_benchmark_data "$scene_id"; then
            echo "[$scene_id] failed"
            failed_scenes+=("$scene_id")
            continue
        fi

        echo "=== [$scene_id] 3/4 run-scene-diff-benchmark ==="
        if ! run_scene_diff_benchmark "$scene_id"; then
            echo "[$scene_id] failed"
            failed_scenes+=("$scene_id")
            continue
        fi

        echo "=== [$scene_id] 4/4 combine-scene-vis-grid ==="
        if ! combine_scene_vis_grid "$scene_id"; then
            echo "[$scene_id] failed"
            failed_scenes+=("$scene_id")
            continue
        fi
    } > "$scene_output_dir/terminal-outputs-$timestamp.txt" 2>&1

    echo "[$scene_id] complete"
    echo
done

echo
if [[ ${#failed_scenes[@]} -gt 0 ]]; then
    printf '  failed: %s\n' "${failed_scenes[*]}"
    exit 1
fi
