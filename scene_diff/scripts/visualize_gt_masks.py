"""
Draws SceneDiff benchmark ground-truth object masks (segments.pkl) onto video1/video2
frames, color-coded by change type: added=green, removed=red, moved=blue.

An object counts as added/removed/moved purely from whether its id is a key of
video1_objects / video2_objects (present in both -> moved, video1 only -> removed,
video2 only -> added) -- the same derivation scene_diff/scripts/evaluate_multiview.py
and concept-graphs/conceptgraph/slam/run_scene_diff_benchmark.py use for their own
Removed/Added/Moved labels and debug visualizations.

Usage:
    python visualize_gt_masks.py --pair_name <pair_name> [--data_root ...] [--output_root ...]
"""
import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

SCENE_DIFF_DIR = Path(os.environ.get(
    "SCENE_DIFF_ROOT", "/node_data/urp26su_dongwoo/concept-graphs-project/scene_diff"
))
DEFAULT_DATA_ROOT = SCENE_DIFF_DIR / "data" / "scenediff_benchmark" / "data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs"

# BGR (cv2 drawing order), matches concept-graphs/conceptgraph/slam/run_scene_diff_benchmark.py's
# CHANGE_COLORS_BGR.
CHANGE_COLORS_BGR = {
    "removed": (0, 0, 255),
    "added": (0, 255, 0),
    "moved": (255, 0, 0),
}

# video{1,2}.mp4 is missing (only original_video{1,2}.MOV/.mov present) in some scene-pair dirs.
VIDEO_CANDIDATES = {
    1: ["video1.mp4", "original_video1.MOV", "original_video1.mov", "original_video1.mp4"],
    2: ["video2.mp4", "original_video2.MOV", "original_video2.mov", "original_video2.mp4"],
}


def find_video(pair_dir: Path, which: int) -> Path:
    for name in VIDEO_CANDIDATES[which]:
        candidate = pair_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no video{which} file found under {pair_dir}")


def classify_objects(video1_objects: dict, video2_objects: dict, objects_meta: list) -> dict:
    """Returns {obj_id: (color_bgr, display_text)}."""
    by_idx = {o["original_obj_idx"]: o for o in objects_meta}
    result = {}
    for obj_id in set(video1_objects) | set(video2_objects):
        in_v1, in_v2 = obj_id in video1_objects, obj_id in video2_objects
        change = "moved" if (in_v1 and in_v2) else ("removed" if in_v1 else "added")
        meta = by_idx.get(obj_id)
        label = meta["label"] if meta else f"obj{obj_id}"
        result[obj_id] = (CHANGE_COLORS_BGR[change], change, f"{change}:{label}")
    return result


def collect_frame_entries(video_objects: dict, classification: dict) -> dict:
    """{frame_idx: [(rle, color_bgr, text), ...]}"""
    frame_entries = {}
    for obj_id, frames in video_objects.items():
        color, _, text = classification[obj_id]
        for frame_idx, rle in frames.items():
            frame_entries.setdefault(int(frame_idx), []).append((rle, color, text))
    return frame_entries


def draw_masks(frame_bgr: np.ndarray, entries: list) -> np.ndarray:
    """entries: list of (rle, color_bgr, text). Returns a new annotated BGR frame with each
    mask alpha-blended, outlined, and labeled with its added/removed/moved:label text."""
    img = frame_bgr.copy()
    for rle, color, text in entries:
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
            cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_video(video_path: Path, frame_entries: dict, out_dir: Path) -> int:
    """Sequentially decodes video_path and writes an annotated PNG for every frame_idx
    present in frame_entries. Sequential cap.read() is used instead of random-access
    cap.set(CAP_PROP_POS_FRAMES, ...), which can land on the wrong frame for some codecs."""
    if not frame_entries:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")

    max_idx = max(frame_entries)
    saved = 0
    frame_idx = 0
    while frame_idx <= max_idx:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        entries = frame_entries.get(frame_idx)
        if entries:
            annotated = draw_masks(frame_bgr, entries)
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.png"), annotated)
            saved += 1
        frame_idx += 1
    cap.release()

    missing = sorted(idx for idx in frame_entries if idx >= frame_idx)
    if missing:
        print(f"  warning: {video_path.name} has only {frame_idx} frames, "
              f"skipped {len(missing)} out-of-range frame_idx(es): {missing}")
    return saved


def visualize_pair(pair_name: str, data_root: Path, output_root: Path) -> None:
    pair_dir = data_root / pair_name
    with open(pair_dir / "segments.pkl", "rb") as f:
        gt = pickle.load(f)

    video1_objects, video2_objects = gt["video1_objects"], gt["video2_objects"]
    classification = classify_objects(video1_objects, video2_objects, gt["objects"])
    n_added = sum(1 for _, change, _ in classification.values() if change == "added")
    n_removed = sum(1 for _, change, _ in classification.values() if change == "removed")
    n_moved = sum(1 for _, change, _ in classification.values() if change == "moved")

    out_dir = output_root / pair_name / "gt_mask_viz"
    n1 = render_video(find_video(pair_dir, 1), collect_frame_entries(video1_objects, classification),
                       out_dir / "video1")
    n2 = render_video(find_video(pair_dir, 2), collect_frame_entries(video2_objects, classification),
                       out_dir / "video2")

    print(f"[{pair_name}] added={n_added} removed={n_removed} moved={n_moved} "
          f"-> {n1} video1 frames, {n2} video2 frames saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair_name", required=True)
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    visualize_pair(args.pair_name, Path(args.data_root), Path(args.output_root))


if __name__ == "__main__":
    main()
