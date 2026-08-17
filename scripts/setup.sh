#!/usr/bin/env bash
# Builds/refreshes an isolated snapshot of concept-graphs/conceptgraph and scene_diff,
# plus a venv whose `import conceptgraph` resolves to the snapshot instead of the live
# concept-graphs/conceptgraph/ directory. This lets a fresh scripts/run.sh invocation run
# against frozen code while concept-graphs/conceptgraph/ and scene_diff/ keep being edited
# live for other in-progress or future runs.
#
# Why this is needed:
# - conceptgraph is pip-installed with `pip install -e`, whose editable finder hardcodes
#   the absolute source path -- copying the code elsewhere does NOT change what
#   `import conceptgraph...` resolves to unless a separate venv registers its own editable
#   install pointing at the copy.
# - run_scene_diff_benchmark.py locates scene_diff/ via a SCENE_DIFF_ROOT-overridable path
#   (see concept-graphs/conceptgraph/slam/run_scene_diff_benchmark.py) and evaluate_multiview.py
#   locates its own scene_diff/utils.py relative to its own __file__, so copying both files
#   together to the same relative layout is enough to isolate them -- no editable install
#   needed for scene_diff since it isn't pip-installed at all, just sys.path-inserted.
#
# Usage:
#   ./scripts/setup.sh          # (re)build the snapshot + venv from the current code state
#
# Then, to run an isolated instance of the pipeline:
#   source .isolated-runs/venv/bin/activate
#   CUDA_VISIBLE_DEVICES=<n> \
#     CONCEPT_GRAPHS_ROOT="$(pwd)/.isolated-runs/concept-graphs" \
#     SCENE_DIFF_ROOT="$(pwd)/.isolated-runs/scene_diff" \
#     ./scripts/run.sh
#   deactivate
#
# Re-run this script any time you want the snapshot to pick up your latest edits (it fully
# refreshes the copied source files; already-running instances are unaffected either way).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE_CG="$PROJECT_ROOT/concept-graphs"
LIVE_SD="$PROJECT_ROOT/scene_diff"
SNAPSHOT_DIR="$PROJECT_ROOT/.isolated-runs"
SNAPSHOT_CG="$SNAPSHOT_DIR/concept-graphs"
SNAPSHOT_SD="$SNAPSHOT_DIR/scene_diff"
VENV_DIR="$SNAPSHOT_DIR/venv"
CONDA_PY="/node_data/urp26su_dongwoo/anaconda3/envs/conceptgraph/bin/python"

mkdir -p "$SNAPSHOT_CG/conceptgraph"

# frozen copy of the mutable python source (small; live edits never touch this copy)
for sub in slam utils dataset hydra_configs; do
    rsync -a --delete "$LIVE_CG/conceptgraph/$sub/" "$SNAPSHOT_CG/conceptgraph/$sub/"
done
cp "$LIVE_CG/conceptgraph/__init__.py" "$SNAPSHOT_CG/conceptgraph/__init__.py"
cp "$LIVE_CG/conceptgraph/scannet200_classes.txt" "$SNAPSHOT_CG/conceptgraph/scannet200_classes.txt"
cp "$LIVE_CG/conceptgraph/scannet200_classes_colors.json" "$SNAPSHOT_CG/conceptgraph/scannet200_classes_colors.json"
cp "$LIVE_CG/setup.py" "$SNAPSHOT_CG/setup.py"

# large/static assets (model weights) are never edited -- symlink instead of copying 1.6G+
for asset in weights sam_l.pt yolov8l-world.pt; do
    ln -sfn "$LIVE_CG/conceptgraph/$asset" "$SNAPSHOT_CG/conceptgraph/$asset"
done

# venv shares the conceptgraph conda env's site-packages (torch etc.) via
# --system-site-packages, but gets its own editable install of conceptgraph pointed at
# the snapshot above -- the shared conda env used by other running instances is untouched
if [[ ! -d "$VENV_DIR" ]]; then
    "$CONDA_PY" -m venv --system-site-packages "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install -e "$SNAPSHOT_CG" --no-deps --quiet

# scene_diff/scripts/evaluate_multiview.py is what step 3 (run_scene_diff_benchmark.py)
# actually calls; it only needs its own file plus scene_diff/utils.py (checked its imports
# -- no other scene_diff/ subpath is touched by evaluate_all_scenes()). scene_diff/data/ is
# 41G of benchmark video/ground-truth data, never edited -- symlink instead of copying.
mkdir -p "$SNAPSHOT_SD"
rsync -a --delete "$LIVE_SD/scripts/" "$SNAPSHOT_SD/scripts/"
cp "$LIVE_SD/utils.py" "$SNAPSHOT_SD/utils.py"
ln -sfn "$LIVE_SD/data" "$SNAPSHOT_SD/data"

echo "snapshot ready: $SNAPSHOT_CG , $SNAPSHOT_SD"
echo "verify with: $VENV_DIR/bin/python -c \"import conceptgraph; print(conceptgraph.__file__)\""
