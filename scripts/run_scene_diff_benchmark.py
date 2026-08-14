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
"""
import argparse
import sys
from pathlib import Path

SCENE_DIFF_DIR = Path("/node_data/urp26su_dongwoo/concept-graphs-project/scene_diff")

sys.path.insert(0, str(SCENE_DIFF_DIR / "scripts"))
import evaluate_multiview  # noqa: E402


def run_benchmark(pair_name: str, benchmark_data_root: Path, output_root: Path, resample_rate: int):
    pred_path = benchmark_data_root / pair_name / "object_masks.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found -- run step 2 "
            "(convert-concept-graphs-to-scene-diff-benchmark-data.sh) for this pair first."
        )

    args = argparse.Namespace(
        gt_dir=str(SCENE_DIFF_DIR / "data" / "scenediff_benchmark" / "data"),
        pred_dir=str(benchmark_data_root),
        video_dir=str(SCENE_DIFF_DIR / "data" / "scenediff_benchmark" / "data"),
        resample_rate=resample_rate,
        max_length=1024,
        iou_threshold=0.5,
        duplicate_match_threshold=1,
        per_frame_duplicate_match_threshold=1,
        mask_background=False,
        crop=False,
    )

    result_path = output_root / pair_name / "eval_result.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    evaluate_multiview.evaluate_all_scenes([pair_name], args, str(result_path), visualize=False)
    print(f"[{pair_name}] benchmark result -> {result_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair_name", required=True)
    parser.add_argument("--benchmark_data_root", type=Path, required=True,
                         help="e.g. outputs/benchmark_data (reads <root>/<pair>/object_masks.pkl)")
    parser.add_argument("--output_root", type=Path, required=True,
                         help="e.g. outputs/benchmark_result (writes <root>/<pair>/eval_result.txt)")
    parser.add_argument("--resample_rate", type=int, default=10,
                         help="Must match the resample rate scenediff_to_conceptgraph.py used for this pair")
    args = parser.parse_args()

    run_benchmark(
        pair_name=args.pair_name,
        benchmark_data_root=args.benchmark_data_root,
        output_root=args.output_root,
        resample_rate=args.resample_rate,
    )


if __name__ == "__main__":
    main()
