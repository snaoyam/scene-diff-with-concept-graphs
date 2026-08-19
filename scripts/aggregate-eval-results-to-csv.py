#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FIELD_PATTERNS = {
    "px_im_iou": re.compile(r"Metric 1: px/im IoU.*?:\s*([\d.]+)"),
    "px_im_tp": re.compile(r"TP:\s*(\d+),\s*FP:\s*(\d+),\s*FN:\s*(\d+)"),
    "obj_im_ap": re.compile(r"Metric 2: obj/im AP.*?:\s*([\d.]+)"),
    "obj_sc_ap_no_class": re.compile(r"Metric 3a: obj/sc AP \(without class requirement\):\s*([\d.]+)"),
    "obj_sc_ap_with_class": re.compile(r"Metric 3b: obj/sc AP \(with class requirement\):\s*([\d.]+)"),
    "det_tp": re.compile(r"True Positives:\s*(\d+)"),
    "det_fp": re.compile(r"False Positives:\s*(\d+)"),
    "det_fn": re.compile(r"False Negatives:\s*(\d+)"),
}


def parse_eval_result(path: Path) -> dict:
    text = path.read_text()

    result = {
        "px_im_iou": None,
        "px_im_tp": None,
        "px_im_fp": None,
        "px_im_fn": None,
        "obj_im_ap": None,
        "obj_sc_ap_no_class": None,
        "obj_sc_ap_with_class": None,
        "det_tp": None,
        "det_fp": None,
        "det_fn": None,
    }

    m = FIELD_PATTERNS["px_im_iou"].search(text)
    if m:
        result["px_im_iou"] = float(m.group(1))

    m = FIELD_PATTERNS["px_im_tp"].search(text)
    if m:
        result["px_im_tp"] = int(m.group(1))
        result["px_im_fp"] = int(m.group(2))
        result["px_im_fn"] = int(m.group(3))

    m = FIELD_PATTERNS["obj_im_ap"].search(text)
    if m:
        result["obj_im_ap"] = float(m.group(1))

    m = FIELD_PATTERNS["obj_sc_ap_no_class"].search(text)
    if m:
        result["obj_sc_ap_no_class"] = float(m.group(1))

    m = FIELD_PATTERNS["obj_sc_ap_with_class"].search(text)
    if m:
        result["obj_sc_ap_with_class"] = float(m.group(1))

    m = FIELD_PATTERNS["det_tp"].search(text)
    if m:
        result["det_tp"] = int(m.group(1))

    m = FIELD_PATTERNS["det_fp"].search(text)
    if m:
        result["det_fp"] = int(m.group(1))

    m = FIELD_PATTERNS["det_fn"].search(text)
    if m:
        result["det_fn"] = int(m.group(1))

    return result


def collect_results(root: Path, glob_pattern: str, scene_id_depth: int) -> dict:
    """scene_id_depth: how many parent levels above the eval file the scene-id directory is."""
    results = {}
    for eval_file in sorted(root.glob(glob_pattern)):
        scene_dir = eval_file
        for _ in range(scene_id_depth):
            scene_dir = scene_dir.parent
        results[scene_dir.name] = parse_eval_result(eval_file)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ConceptGraph-diff and SceneDiff benchmark eval_result.txt files into one CSV."
    )
    parser.add_argument(
        "--concept-graph-outputs",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Root dir containing <scene-id>/benchmark_result/eval_result.txt (default: ./outputs)",
    )
    parser.add_argument(
        "--scene-diff-outputs",
        type=Path,
        default=PROJECT_ROOT / "scene_diff" / "output" / "scenediff_benchmark",
        help="Root dir containing <scene-id>/eval_result_frame_dup_1_obj_dup_1.txt (default: ./scene_diff/output/scenediff_benchmark)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "eval_results_summary.csv",
        help="Path to write the combined CSV (default: ./outputs/eval_results_summary.csv)",
    )
    args = parser.parse_args()

    concept_graph_results = collect_results(
        args.concept_graph_outputs, "*/benchmark_result/eval_result.txt", scene_id_depth=2
    )
    scene_diff_results = collect_results(
        args.scene_diff_outputs, "*/eval_result_frame_dup_1_obj_dup_1.txt", scene_id_depth=1
    )

    scene_ids = sorted(set(concept_graph_results) | set(scene_diff_results))

    metric_fields = [
        "px_im_iou",
        "px_im_tp",
        "px_im_fp",
        "px_im_fn",
        "obj_im_ap",
        "obj_sc_ap_no_class",
        "obj_sc_ap_with_class",
        "det_tp",
        "det_fp",
        "det_fn",
    ]

    header = ["scene_id"]
    for source in ("concept_graph", "scene_diff"):
        header += [f"{source}_{field}" for field in metric_fields]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for scene_id in scene_ids:
            row = [scene_id]
            cg = concept_graph_results.get(scene_id, {})
            sd = scene_diff_results.get(scene_id, {})
            row += [cg.get(field) for field in metric_fields]
            row += [sd.get(field) for field in metric_fields]
            writer.writerow(row)

    print(f"Wrote {len(scene_ids)} scenes to {args.output}")
    missing_cg = sorted(set(scene_ids) - set(concept_graph_results))
    missing_sd = sorted(set(scene_ids) - set(scene_diff_results))
    if missing_cg:
        print(f"Missing concept-graph result for {len(missing_cg)} scene(s): {missing_cg}")
    if missing_sd:
        print(f"Missing scene-diff result for {len(missing_sd)} scene(s): {missing_sd}")


if __name__ == "__main__":
    main()
