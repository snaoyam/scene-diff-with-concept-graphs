"""
Step 3: score one pair's object_masks.pkl (produced by step 2,
convert_concept_graphs_to_scene_diff_benchmark_data.py) against the official SceneDiff
benchmark ground truth, by calling scene_diff/scripts/evaluate_multiview.py's own
evaluate_all_scenes() directly for that single scene.

evaluate_multiview.py's CLI only evaluates a whole split at once; importing it as a
module and calling evaluate_all_scenes([pair_name], ...) reuses its exact scoring logic
(px/im IoU, obj/im AP, obj/sc AP) for one scene at a time, which is what run.sh needs
since it processes pairs one at a time through steps 1/2/3.

--resample_rate must match what scene_diff/data/scenediff_to_conceptgraph.py used when
it produced Datasets/scenediff/<pair>/{before,after}/, since that determines the frame
index numbering ConceptGraph's image_idx (and thus step 2's object_masks.pkl frame_idx
keys) uses. Verified empirically at 10 (see plan/commit notes) -- override with
--resample_rate if the preprocessing is ever re-run with a different value.

debug_masks/{before,after}/ mask-overlay images -- green=added, red=removed,
blue=moved -- are written by step 2 (convert_concept_graphs_to_scene_diff_benchmark_data.py,
see save_debug_mask_visualizations() there) into benchmark_data/, since they only need
object_masks.pkl and the scene's own dataset, not this stage's scoring output. This
stage does not touch them.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import hydra

SCENE_DIFF_DIR = Path(os.environ.get(
    "SCENE_DIFF_ROOT", "/node_data/urp26su_dongwoo/concept-graphs-project/scene_diff"
))

sys.path.insert(0, str(SCENE_DIFF_DIR / "scripts"))
import evaluate_multiview  # noqa: E402

VIDEO_DATA_ROOT = SCENE_DIFF_DIR / "data" / "scenediff_benchmark" / "data"


def _load_output_root(output_root_override: str | None = None) -> Path:
    """Reads output_root from rerun_realtime_mapping.yaml -- the single source of
    truth for where pipeline outputs live -- via hydra.compose(), the same
    hydra_configs/ this script's sibling rerun_realtime_mapping.py loads with
    @hydra.main(config_path="../hydra_configs/", config_name="rerun_realtime_mapping").
    hydra.initialize() resolves config_path relative to this file, so it's the same
    "../hydra_configs" used there. output_root_override mirrors passing
    output_root=... on rerun_realtime_mapping.py's CLI, so this stage agrees with
    stage 1 on where a given run's outputs live even when that run didn't use the
    yaml's default output_root."""
    overrides = [f"output_root={output_root_override}"] if output_root_override else []
    with hydra.initialize(version_base=None, config_path="../hydra_configs"):
        cfg = hydra.compose(config_name="rerun_realtime_mapping", overrides=overrides)
    return Path(cfg.output_root).resolve()


def run_benchmark(pair_name: str, resample_rate: int, output_root: str | None = None):
    scene_root = _load_output_root(output_root) / pair_name
    benchmark_data_dir = scene_root / "benchmark_data"
    pred_path = benchmark_data_dir / "object_masks.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found -- run step 2 "
            "(convert-concept-graphs-to-scene-diff-benchmark-data.sh) for this pair first."
        )

    gt_path = VIDEO_DATA_ROOT / pair_name / "segments.pkl"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"{gt_path} not found -- evaluate_multiview.py silently skips scenes whose "
            "GT segments.pkl is missing (try/except: continue in evaluate_all_scenes), "
            "which produces a fake all-zero eval_result.txt instead of failing. Make sure "
            "scene_diff/data/scenediff_benchmark is downloaded and SCENE_DIFF_ROOT points at it."
        )

    result_dir = scene_root / "benchmark_result"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "eval_result.txt"

    # evaluate_multiview.py expects a flat <pred_dir>/<scene_name>/object_masks.pkl
    # layout (get_prediction_dir), but our own outputs/ nests object_masks.pkl one
    # level differently now (outputs/<pair>/benchmark_data/object_masks.pkl, no
    # further <pair>-named subfolder inside benchmark_data/). Bridge the two with a
    # single throwaway symlink rather than touching the official evaluator.
    with tempfile.TemporaryDirectory() as tmp_pred_root:
        (Path(tmp_pred_root) / pair_name).symlink_to(benchmark_data_dir, target_is_directory=True)

        args = argparse.Namespace(
            gt_dir=str(SCENE_DIFF_DIR / "data" / "scenediff_benchmark" / "data"),
            pred_dir=tmp_pred_root,
            video_dir=str(SCENE_DIFF_DIR / "data" / "scenediff_benchmark" / "data"),
            resample_rate=resample_rate,
            max_length=1024,
            iou_threshold=0.5,
            duplicate_match_threshold=1,
            per_frame_duplicate_match_threshold=1,
            mask_background=False,
            crop=False,
        )

        evaluate_multiview.evaluate_all_scenes([pair_name], args, str(result_path), visualize=False)

    print(f"[{pair_name}] benchmark result -> {result_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair_name", required=True)
    parser.add_argument("--resample_rate", type=int, default=10,
                         help="Must match the resample rate scenediff_to_conceptgraph.py used for this pair")
    # Defaults to None so the yaml stays the single source of truth; pass one only to
    # match a rerun_realtime_mapping.py run that itself overrode output_root=....
    parser.add_argument("--output_root", default=None,
                         help="Overrides the yaml's output_root, like output_root=... on rerun_realtime_mapping.py's CLI")
    args = parser.parse_args()

    run_benchmark(
        pair_name=args.pair_name,
        resample_rate=args.resample_rate,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
