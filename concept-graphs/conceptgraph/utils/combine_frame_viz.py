'''
Stitches the four per-frame debug visualizations written by
slam/rerun_realtime_mapping.py (detected_masks/, filtered_masks/, fused_masks/,
scenegraph_viz/) into one grid image per frame, so a single image shows how one
frame moved through the whole pipeline:

    [detected_masks]  [filtered_masks]  [        (empty)        ]
    [        fused_masks (spans 2 cols)         ]  [scenegraph_viz]

fused_masks/ is itself already a side-by-side composite (real detections | real
+ reprojected), so it gets two grid cells' worth of width instead of one.

Usage:
    python -m conceptgraph.utils.combine_frame_viz <exp_out_path> [<exp_out_path> ...]

<exp_out_path> is the mapping experiment dir, e.g.
outputs/<scene-pair>/concept_graphs/{before,after}/exps/r_mapping_pilot -- the
same directory that contains detected_masks/, filtered_masks/, fused_masks/,
and scenegraph_viz/. Output is written to <exp_out_path>/frame_grid_viz/{n}.jpg.

Called automatically at the end of run_mapping_for_scene() in
rerun_realtime_mapping.py, and can also be re-run standalone against already
-built outputs (nothing here depends on re-running detection/mapping).
'''
import re
import sys
from pathlib import Path

import cv2
import numpy as np

DETECTED_MASKS_DIR = "detected_masks"
FILTERED_MASKS_DIR = "filtered_masks"
FUSED_MASKS_DIR = "fused_masks"
SCENEGRAPH_VIZ_DIR = "scenegraph_viz"
OUT_DIR = "z_frame_grid_viz"

_FRAME_NUM_RE = re.compile(r"^(\d+)")


def _frame_num(stem: str):
    m = _FRAME_NUM_RE.match(stem)
    return int(m.group(1)) if m else None


def _collect_frames(exp_out_path: Path) -> dict:
    '''Maps frame number -> {"detected": path, "filtered": path, "fused": path, "scenegraph": path}.'''
    frames = {}

    def _add(dirname, key, suffix_strip=""):
        d = exp_out_path / dirname
        if not d.is_dir():
            return
        for p in d.iterdir():
            if not p.is_file():
                continue
            stem = p.stem
            if suffix_strip and stem.endswith(suffix_strip):
                stem = stem[: -len(suffix_strip)]
            n = _frame_num(stem)
            if n is None:
                continue
            frames.setdefault(n, {})[key] = p

    _add(DETECTED_MASKS_DIR, "detected")
    _add(FILTERED_MASKS_DIR, "filtered")
    _add(FUSED_MASKS_DIR, "fused")
    _add(SCENEGRAPH_VIZ_DIR, "scenegraph", suffix_strip="_viz")
    return frames


def _letterbox(img, size):
    '''Resizes img to fit within (w, h) preserving aspect ratio, centered on a
    background-color canvas -- source images are a mix of portrait (frame masks)
    and wide-landscape (scenegraph_viz), so a plain resize would stretch/squash them.'''
    w, h = size
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 20, dtype=np.uint8)
    x_off, y_off = (w - nw) // 2, (h - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def _load_or_blank(path, size):
    w, h = size
    if path is not None:
        img = cv2.imread(str(path))
        if img is not None:
            return _letterbox(img, size)
    blank = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.putText(blank, "missing", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
    return blank


def _make_grid_image(paths: dict, cell_size=(480, 360)) -> np.ndarray:
    w, h = cell_size
    detected = _load_or_blank(paths.get("detected"), (w, h))
    filtered = _load_or_blank(paths.get("filtered"), (w, h))
    empty = np.full((h, w, 3), 20, dtype=np.uint8)
    scenegraph = _load_or_blank(paths.get("scenegraph"), (w, h))
    fused = _load_or_blank(paths.get("fused"), (2 * w, h))

    top_row = cv2.hconcat([detected, filtered, empty])
    bottom_row = cv2.hconcat([fused, scenegraph])
    return cv2.vconcat([top_row, bottom_row])


def combine_frame_viz(exp_out_path, cell_size=(480, 360)) -> Path:
    '''Builds <exp_out_path>/frame_grid_viz/{n}.jpg for every frame found in any of
    detected_masks/, filtered_masks/, fused_masks/, scenegraph_viz/. Returns the
    output directory.'''
    exp_out_path = Path(exp_out_path)
    frames = _collect_frames(exp_out_path)
    if not frames:
        print(f"combine_frame_viz: no frames found under {exp_out_path}, skipping")
        return exp_out_path / OUT_DIR

    out_dir = exp_out_path / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in sorted(frames):
        grid = _make_grid_image(frames[n], cell_size=cell_size)
        cv2.imwrite(str(out_dir / f"{n}.jpg"), grid)

    print(f"combine_frame_viz: wrote {len(frames)} frame grids to {out_dir}")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python -m conceptgraph.utils.combine_frame_viz <exp_out_path> [<exp_out_path> ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        combine_frame_viz(arg)
