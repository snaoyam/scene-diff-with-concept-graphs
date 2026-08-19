#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=6 #always use only GPU=6 and do not use any other GPU nodes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="$(date '+%Y-%m-%d_%H-%M-%S')"

# ./scripts/setup.sh가 만들어둔 격리 스냅샷+venv가 있으면 자동으로 사용 (수동 activate/env var 불필요).
# 다른 터미널에서 setup.sh를 한 번만 실행해도, 이후 어느 터미널에서 run.sh를 실행하든 적용된다.
# ISOLATED_RUN=0으로 실행하면 격리 스냅샷이 있어도 무시하고 원본 환경에서 실행하며 outputs/에 쓴다.
# 그 외(기본값)에는 격리 스냅샷을 쓰고 isolated-outputs/에 써서 outputs/와 절대 섞이지 않는다.
# OUTPUT_ROOT는 scene_pair=...처럼 output_root=...로 매 파이썬 호출에 명시적으로 전달된다
# (construct-concept-graphs.sh 등 참고) -- 3단계 모두 같은 값을 봐야 하므로 여기서만 정한다.
ISOLATED_RUN="${ISOLATED_RUN:-1}"
if [[ "$ISOLATED_RUN" == "0" ]]; then
    OUTPUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/outputs-test"
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
    export SCENE_DIFF_ROOT="${SCENE_DIFF_ROOT:-$ISOLATED_RUN_DIR/scene_diff}"
fi

source "$SCRIPT_DIR/scene-pairs.sh"
source "$SCRIPT_DIR/construct-concept-graphs.sh"
source "$SCRIPT_DIR/convert-concept-graphs-to-scene-diff-benchmark-data.sh"
source "$SCRIPT_DIR/run-scene-diff-benchmark.sh"

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
        echo "=== [$scene_id] 1/3 construct-concept-graphs ==="
        if ! construct_concept_graphs "$scene_id"; then
            echo "[$scene_id] failed"
            failed_scenes+=("$scene_id")
            continue
        fi

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
if [[ ${#failed_scenes[@]} -gt 0 ]]; then
    printf '  failed: %s\n' "${failed_scenes[*]}"
    exit 1
fi
