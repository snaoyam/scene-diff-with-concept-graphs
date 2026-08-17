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

Also writes debug output under benchmark_result/: a copy of object_masks.pkl plus
debug_masks/{before,after,moved}/ mask-overlay images (green=added, red=removed,
blue=moved, each mask labeled with its object id) -- see _save_debug_visualizations().
"""
import argparse
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import hydra
import numpy as np
from pycocotools import mask as mask_utils

SCENE_DIFF_DIR = Path(os.environ.get(
    "SCENE_DIFF_ROOT", "/node_data/urp26su_dongwoo/concept-graphs-project/scene_diff"
))

sys.path.insert(0, str(SCENE_DIFF_DIR / "scripts"))
import evaluate_multiview  # noqa: E402

VIDEO_DATA_ROOT = SCENE_DIFF_DIR / "data" / "scenediff_benchmark" / "data"

# BGR (cv2 drawing order), matching evaluate_multiview.py's own Removed/Added/Moved colors.
CHANGE_COLORS_BGR = {
    "removed": (0, 0, 255),
    "added": (0, 255, 0),
    "moved": (255, 0, 0),
}


def _load_output_root() -> Path:
    """Reads output_root from rerun_realtime_mapping.yaml -- the single source of
    truth for where pipeline outputs live -- via hydra.compose(), the same
    hydra_configs/ this script's sibling rerun_realtime_mapping.py loads with
    @hydra.main(config_path="../hydra_configs/", config_name="rerun_realtime_mapping").
    hydra.initialize() resolves config_path relative to this file, so it's the same
    "../hydra_configs" used there."""
    with hydra.initialize(version_base=None, config_path="../hydra_configs"):
        cfg = hydra.compose(config_name="rerun_realtime_mapping")
    return Path(cfg.output_root).resolve()


def _draw_masks(rgb_frame, entries):
    """rgb_frame: HxWx3 float [0,1] RGB. entries: list of (obj_id, rle, color_bgr).
    Returns a uint8 BGR image with each mask alpha-blended, outlined, and labeled
    with its object id so the same id can be traced across before/after frames."""
    img = np.ascontiguousarray((rgb_frame[..., ::-1] * 255).clip(0, 255).astype(np.uint8))
    for obj_id, rle, color in entries:
        mask = mask_utils.decode(rle).astype(bool)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = img.copy()
        overlay[mask] = color
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 2)
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            text = f"id{obj_id}"
            cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _side_by_side(img_a, label_a, img_b, label_b):
    h = max(img_a.shape[0], img_b.shape[0])
    gap, top = 10, 30
    canvas = np.full((h + top, img_a.shape[1] + gap + img_b.shape[1], 3), 255, dtype=np.uint8)
    canvas[top:top + img_a.shape[0], :img_a.shape[1]] = img_a
    canvas[top:top + img_b.shape[0], img_a.shape[1] + gap:] = img_b
    cv2.putText(canvas, label_a, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, label_b, (img_a.shape[1] + gap + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def _save_debug_visualizations(pair_name: str, benchmark_data_dir: Path, result_dir: Path, resample_rate: int):
    """Dumps a copy of object_masks.pkl plus per-frame mask overlays into
    result_dir/debug_masks/ (before/, after/, moved/) so this run can be inspected
    without re-running anything. green=added, red=removed, blue=moved; each mask is
    labeled with its object id so a moved object's before/after position can be
    found by matching ids across before/ and after/. moved/ additionally places
    each moved object's before/after frame side by side."""
    pred_path = benchmark_data_dir / "object_masks.pkl"
    with open(pred_path, "rb") as f:
        pred_data = pickle.load(f)
    shutil.copy2(pred_path, result_dir / "object_masks.pkl")

    objects = {k: v for k, v in pred_data.items() if k not in ("H", "W")}
    if not objects:
        return
    H, W = pred_data["H"], pred_data["W"]

    frame_args = argparse.Namespace(resample_rate=resample_rate)
    rgb_frames_1, rgb_frames_2 = evaluate_multiview.load_rgb_frames(VIDEO_DATA_ROOT / pair_name, H, W, frame_args)
    rgb_frames = {"video_1": rgb_frames_1, "video_2": rgb_frames_2}

    debug_dir = result_dir / "debug_masks"
    before_dir, after_dir, moved_dir = debug_dir / "before", debug_dir / "after", debug_dir / "moved"
    for d in (before_dir, after_dir, moved_dir):
        d.mkdir(parents=True, exist_ok=True)

    frame_entries = {"video_1": {}, "video_2": {}}
    for obj_id, obj in objects.items():
        if "video_1" in obj and "video_2" in obj:
            color = CHANGE_COLORS_BGR["moved"]
        elif "video_1" in obj:
            color = CHANGE_COLORS_BGR["removed"]
        else:
            color = CHANGE_COLORS_BGR["added"]
        for video_key in ("video_1", "video_2"):
            for frame_idx, mask_info in obj.get(video_key, {}).items():
                frame_entries[video_key].setdefault(int(frame_idx), []).append((obj_id, mask_info["mask"], color))

    for video_key, out_dir in (("video_1", before_dir), ("video_2", after_dir)):
        for frame_idx, entries in frame_entries[video_key].items():
            frame = rgb_frames[video_key].get(frame_idx)
            if frame is None:
                continue
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.png"), _draw_masks(frame, entries))

    for obj_id, obj in objects.items():
        if "video_1" not in obj or "video_2" not in obj:
            continue
        before_idx, after_idx = min(obj["video_1"].keys()), min(obj["video_2"].keys())
        before_rgb, after_rgb = rgb_frames["video_1"].get(before_idx), rgb_frames["video_2"].get(after_idx)
        if before_rgb is None or after_rgb is None:
            continue
        color = CHANGE_COLORS_BGR["moved"]
        before_img = _draw_masks(before_rgb, [(obj_id, obj["video_1"][before_idx]["mask"], color)])
        after_img = _draw_masks(after_rgb, [(obj_id, obj["video_2"][after_idx]["mask"], color)])
        canvas = _side_by_side(before_img, "BEFORE", after_img, "AFTER")
        cv2.imwrite(str(moved_dir / f"obj_{obj_id:03d}.png"), canvas)

    print(f"[{pair_name}] debug visualizations -> {debug_dir}")


def run_benchmark(pair_name: str, resample_rate: int):
    scene_root = _load_output_root() / pair_name
    benchmark_data_dir = scene_root / "benchmark_data"
    pred_path = benchmark_data_dir / "object_masks.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found -- run step 2 "
            "(convert-concept-graphs-to-scene-diff-benchmark-data.sh) for this pair first."
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

    _save_debug_visualizations(pair_name, benchmark_data_dir, result_dir, resample_rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair_name", required=True)
    parser.add_argument("--resample_rate", type=int, default=10,
                         help="Must match the resample rate scenediff_to_conceptgraph.py used for this pair")
    args = parser.parse_args()

    run_benchmark(
        pair_name=args.pair_name,
        resample_rate=args.resample_rate,
    )


if __name__ == "__main__":
    main()
