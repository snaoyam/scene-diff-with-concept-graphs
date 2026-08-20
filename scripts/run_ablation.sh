#!/usr/bin/env bash
# Ablation study: isolates the contribution of each of the three ConceptGraph
# improvements (scene-specific vocabulary discovery, geometry-only object fusion,
# confidence/visibility-filtered change detection) by reverting one at a time via the
# ablation_* hydra keys in conceptgraph/hydra_configs/rerun_realtime_mapping.yaml, plus
# an "original_conceptgraph" config with all three reverted at once (also usable as the
# Original-ConceptGraph baseline row in the results table) and a "full" config (all
# three on -- the normal pipeline) for reference.
#
# Runs stage 1 (construct-concept-graphs) -> stage 2 (convert-to-benchmark-data) ->
# stage 3 (run-scene-diff-benchmark) for each (config, scene pair), skipping anything
# whose eval_result.txt already exists -- same convention as scripts/run.sh, which this
# reuses stage-by-stage. Scene-graph visualization (combine-scene-vis-grid, run.sh's
# stage 4) is skipped here since the ablation only needs scores, not images.
#
# Scene pairs: whatever scripts/scene-pairs.sh defines (SCENE_PAIRS) -- the same set
# run.sh itself uses. Edit scene-pairs.sh (e.g. uncomment its smaller hand-picked list)
# to control which/how many scenes this runs, in one place shared with run.sh. Note this
# script runs that same set 5 times over (once per ablation config, "full" mostly reused
# from run.sh's own output), so the full 105-pair val split here is far slower than a
# single run.sh pass -- narrow scene-pairs.sh's list first if you need a quicker turnaround.
#
# Usage:
#   ./scripts/run_ablation.sh
#
# Same ISOLATED_RUN convention as run.sh: ISOLATED_RUN=1 (default) runs the
# .isolated-runs/ snapshot against isolated-outputs*/; ISOLATED_RUN=0 runs the live
# source tree against outputs*/. Either way, each ablation config writes to its own
# output root (isolated-outputs-ablation-<config>/, sibling to the normal
# isolated-outputs/) EXCEPT "full", which reads/writes the SAME root scripts/run.sh
# uses -- so scenes run.sh has already finished are reused here, not recomputed. Do not
# run this at the same time as scripts/run.sh on overlapping scenes: both pin
# CUDA_VISIBLE_DEVICES=6, and "full" writes into run.sh's own output directory.
#
# IMPORTANT: the ablation_* hydra keys only exist in the LIVE concept-graphs/ source
# tree, not in an already-built .isolated-runs/ snapshot (setup.sh copies the source at
# snapshot-build time, and the snapshot is refreshed only by the user, never by an
# agent). If your snapshot predates the ablation_* keys, re-run ./scripts/setup.sh
# yourself first -- this script checks for that and refuses to run otherwise.

set -euo pipefail
CUDA_VISIBLE_DEVICES=6 #always only use GPU=6 and do not use any other GPU nodes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="$(date '+%Y-%m-%d_%H-%M-%S')"

ISOLATED_RUN="${ISOLATED_RUN:-1}"
if [[ "$ISOLATED_RUN" == "0" ]]; then
    BASE_OUTPUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/outputs"
else
    BASE_OUTPUT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/isolated-outputs"
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

# Refuse to run against a snapshot/tree that predates the ablation_* keys instead of
# failing confusingly mid-scene with a hydra "key not in struct" error.
RESOLVED_CG_ROOT="${CONCEPT_GRAPHS_ROOT:-$(cd "$SCRIPT_DIR/../concept-graphs" && pwd)}"
ABLATION_YAML="$RESOLVED_CG_ROOT/conceptgraph/hydra_configs/rerun_realtime_mapping.yaml"
if ! grep -q "^ablation_disable_vocabulary_discovery:" "$ABLATION_YAML" 2>/dev/null; then
    echo "ERROR: $ABLATION_YAML has no ablation_disable_vocabulary_discovery key." >&2
    echo "The code snapshot at $RESOLVED_CG_ROOT predates the ablation_* hydra options." >&2
    echo "Run ./scripts/setup.sh yourself to refresh .isolated-runs/ (this script will not do it for you), then re-run." >&2
    exit 1
fi

source "$SCRIPT_DIR/construct-concept-graphs.sh"
source "$SCRIPT_DIR/convert-concept-graphs-to-scene-diff-benchmark-data.sh"
source "$SCRIPT_DIR/run-scene-diff-benchmark.sh"
source "$SCRIPT_DIR/scene-pairs.sh"

# Parallel arrays: config name / stage-1 hydra overrides / stage-2 extra CLI args /
# output_root suffix (empty for "full" -- reuses run.sh's own output root as-is).
ABLATION_CONFIG_NAMES=(
  full
  original_conceptgraph
  no_vocabulary_discovery
  no_geometry_only_fusion
  no_confidence_visibility_filter
)
ABLATION_CONFIG_CONSTRUCT_OVERRIDES=(
  ""
  "ablation_disable_vocabulary_discovery=true ablation_disable_geometry_only_fusion=true"
  "ablation_disable_vocabulary_discovery=true"
  "ablation_disable_geometry_only_fusion=true"
  ""
)
ABLATION_CONFIG_CONVERT_ARGS=(
  ""
  "--ablation_disable_confidence_visibility_filter"
  ""
  ""
  "--ablation_disable_confidence_visibility_filter"
)
ABLATION_CONFIG_OUTPUT_SUFFIX=(
  ""
  "-ablation-original"
  "-ablation-no-vocab"
  "-ablation-no-fusion"
  "-ablation-no-confidence"
)

failed_runs=()

for cfg_idx in "${!ABLATION_CONFIG_NAMES[@]}"; do
    config_name="${ABLATION_CONFIG_NAMES[$cfg_idx]}"
    construct_overrides="${ABLATION_CONFIG_CONSTRUCT_OVERRIDES[$cfg_idx]}"
    convert_extra="${ABLATION_CONFIG_CONVERT_ARGS[$cfg_idx]}"
    OUTPUT_ROOT="${BASE_OUTPUT_ROOT}${ABLATION_CONFIG_OUTPUT_SUFFIX[$cfg_idx]}"
    export OUTPUT_ROOT

    echo
    echo "########## ablation config: $config_name (output_root=$OUTPUT_ROOT) ##########"

    for scene_id in "${SCENE_PAIRS[@]}"; do
        scene_output_dir="$OUTPUT_ROOT/$scene_id"

        if [[ -f "$scene_output_dir/benchmark_result/eval_result.txt" ]]; then
            echo "[$config_name/$scene_id] eval_result.txt already exists, skipping"
            continue
        fi

        # Same incomplete-run cleanup as run.sh.
        concept_graphs_dir="$scene_output_dir/concept_graphs"
        for variant in before after; do
            variant_pcd="$concept_graphs_dir/$variant/exps/r_mapping_pilot/pcd_r_mapping_pilot.pkl.gz"
            if [[ -d "$concept_graphs_dir/$variant" && ! -f "$variant_pcd" ]]; then
                echo "[$config_name/$scene_id] removing incomplete concept_graphs/$variant"
                rm -rf "$concept_graphs_dir/$variant"
            fi
        done
        if [[ -d "$scene_output_dir/benchmark_data" && ! -f "$scene_output_dir/benchmark_data/object_masks.pkl" ]]; then
            echo "[$config_name/$scene_id] removing incomplete benchmark_data"
            rm -rf "$scene_output_dir/benchmark_data"
        fi
        if [[ -d "$scene_output_dir/benchmark_result" ]]; then
            echo "[$config_name/$scene_id] removing incomplete benchmark_result"
            rm -rf "$scene_output_dir/benchmark_result"
        fi

        echo "=== running [$config_name/$scene_id] ==="
        mkdir -p "$scene_output_dir"

        {
            echo "=== [$config_name/$scene_id] 1/3 construct-concept-graphs ==="
            if ! construct_concept_graphs "$scene_id" $construct_overrides; then
                echo "[$config_name/$scene_id] failed"
                failed_runs+=("$config_name/$scene_id")
                continue
            fi

            echo "=== [$config_name/$scene_id] 2/3 convert-concept-graphs-to-scene-diff-benchmark-data ==="
            if ! convert_concept_graphs_to_scene_diff_benchmark_data "$scene_id" $convert_extra; then
                echo "[$config_name/$scene_id] failed"
                failed_runs+=("$config_name/$scene_id")
                continue
            fi

            echo "=== [$config_name/$scene_id] 3/3 run-scene-diff-benchmark ==="
            if ! run_scene_diff_benchmark "$scene_id"; then
                echo "[$config_name/$scene_id] failed"
                failed_runs+=("$config_name/$scene_id")
                continue
            fi
        } > "$scene_output_dir/terminal-outputs-ablation-$timestamp.txt" 2>&1

        echo "[$config_name/$scene_id] complete"
    done
done

echo
if [[ ${#failed_runs[@]} -gt 0 ]]; then
    echo "failed runs:"
    printf '  %s\n' "${failed_runs[@]}"
fi

echo
echo "########## ablation summary (mean over ${#SCENE_PAIRS[@]} scenes per config) ##########"
python3 - "$BASE_OUTPUT_ROOT" "${SCENE_PAIRS[@]}" <<'PYEOF'
import re
import statistics as st
import sys
from pathlib import Path

base_output_root = Path(sys.argv[1])
scene_pairs = sys.argv[2:]

configs = [
    ("full", ""),
    ("original_conceptgraph", "-ablation-original"),
    ("no_vocabulary_discovery", "-ablation-no-vocab"),
    ("no_geometry_only_fusion", "-ablation-no-fusion"),
    ("no_confidence_visibility_filter", "-ablation-no-confidence"),
]

def parse(path):
    text = path.read_text()
    def grab(pat):
        m = re.search(pat, text)
        return float(m.group(1)) if m else None
    return (
        grab(r"Metric 1: px/im IoU.*?:\s*([\d.]+)"),
        grab(r"Metric 2: obj/im AP.*?:\s*([\d.]+)"),
        grab(r"Metric 3a: obj/sc AP.*?:\s*([\d.]+)"),
    )

header = f"{'config':32s} {'n':>4s} {'px/im IoU':>10s} {'obj/im AP':>10s} {'obj/sc AP':>10s}"
print(header)
print("-" * len(header))
for name, suffix in configs:
    root = Path(str(base_output_root) + suffix)
    ious, ims, scs = [], [], []
    n = 0
    for scene in scene_pairs:
        f = root / scene / "benchmark_result" / "eval_result.txt"
        if not f.exists():
            continue
        iou, im_ap, sc_ap = parse(f)
        if iou is None:
            continue
        n += 1
        ious.append(iou); ims.append(im_ap); scs.append(sc_ap)
    if n == 0:
        print(f"{name:32s} {0:4d} {'--':>10s} {'--':>10s} {'--':>10s}  (no results yet at {root})")
        continue
    print(f"{name:32s} {n:4d} {st.mean(ious):10.4f} {st.mean(ims):10.4f} {st.mean(scs):10.4f}")
PYEOF
