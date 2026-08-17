"""
Step 2: match a pair's "before" and "after" ConceptGraphs against each other and
export the result as a SceneDiff-benchmark-compatible object_masks.pkl, so it can be
scored by scene_diff/scripts/evaluate_multiview.py (run_scene_diff_benchmark.py, step 3).

Matching: before/after ConceptGraphs share one Pi3-estimated coordinate frame (see
conceptgraph/hydra_configs/rerun_realtime_mapping.yaml), so their 3D object bounding
boxes are directly comparable. Objects are matched one-to-one via
scipy.optimize.linear_sum_assignment on a cost combining 3D centroid distance and CLIP
feature dissimilarity; unmatched-before = removed, unmatched-after = added, matched
pairs whose centroids moved more than --moved_threshold = moved. Matched pairs that
didn't move aren't a "change" and aren't exported.

object_masks.pkl schema (consumed by scene_diff/scripts/evaluate_multiview.py):
    {
        'H': int, 'W': int,
        <object_id: int>: {
            'video_1': {<frame_idx: int>: {'mask': coco_rle, 'cost': float}, ...},  # "before"
            'video_2': {<frame_idx: int>: {'mask': coco_rle, 'cost': float}, ...},  # "after"
        },
        ...
    }
An object present in only video_1 = removed, only video_2 = added, both = moved.
frame_idx values are ConceptGraph's own image_idx (already resampled the same way
scene_diff/data/scenediff_to_conceptgraph.py resampled the source video), so they line
up with evaluate_multiview.py's GT frame indices as long as the same --resample_rate is
passed to it.

Also writes a debug image, benchmark_data/moved_objects_pointcloud.png: a top-down 2D
projection of the full 3D point cloud (context in gray) with each moved object's
before/after points (orange/blue) and an arrow between their centroids -- see
save_moved_objects_pointcloud_debug_image().
"""
import argparse
import gzip
import pickle
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

AXIS_NAMES = ["X", "Y", "Z"]


def _load_rerun_mapping_config() -> dict:
    """Reads output_root/exp_suffix from rerun_realtime_mapping.yaml -- the single
    source of truth for where pipeline outputs live -- via hydra.compose(), the same
    hydra_configs/ this script's sibling rerun_realtime_mapping.py loads with
    @hydra.main(config_path="../hydra_configs/", config_name="rerun_realtime_mapping").
    hydra.initialize() resolves config_path relative to this file, so it's the same
    "../hydra_configs" used there."""
    with hydra.initialize(version_base=None, config_path="../hydra_configs"):
        cfg = hydra.compose(config_name="rerun_realtime_mapping")
    return {"output_root": Path(cfg.output_root).resolve(), "exp_suffix": cfg.exp_suffix}


def load_scene_hw(pair_name: str):
    """Reads (H, W) straight from the scene's dataset config
    (conceptgraph/dataset/dataconfigs/scenediff/<pair_name>/<variant>.yaml,
    camera_params.image_height/image_width) instead of from any detected object's mask --
    the same source rerun_realtime_mapping.py itself falls back to when
    cfg.image_height/image_width are unset (see process_cfg() in slam/utils.py). Unlike
    infer_hw(), this works even when before_objs/after_objs are both completely empty."""
    dataconfigs_dir = Path(__file__).resolve().parent.parent / "dataset" / "dataconfigs" / "scenediff" / pair_name
    for variant in ("before", "after"):
        path = dataconfigs_dir / f"{variant}.yaml"
        if path.exists():
            camera_params = OmegaConf.load(path).camera_params
            return int(camera_params.image_height), int(camera_params.image_width)
    return None


def load_objects(concept_graphs_dir: Path, variant: str, exp_suffix: str):
    pcd_path = concept_graphs_dir / variant / "exps" / exp_suffix / f"pcd_{exp_suffix}.pkl.gz"
    with gzip.open(pcd_path, "rb") as f:
        data = pickle.load(f)
    objects = [obj for obj in data["objects"] if not obj.get("is_background", False)]
    # up_axis/up_direction are the main pipeline's camera-grounded detect_up_vector()
    # result (slam/utils.py), persisted by save_pointcloud() -- .get() so pcd files
    # saved before this was added still load fine (up_axis=None triggers a fallback
    # in save_moved_objects_pointcloud_debug_image()).
    return objects, data.get("up_axis"), data.get("up_direction")


def bbox_center(obj):
    return np.asarray(obj["bbox_np"], dtype=np.float64).mean(axis=0)


def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return 1.0 - float(np.dot(a, b) / denom)


def match_objects(before_objs, after_objs, max_match_distance: float, visual_weight: float):
    """Returns (matches, unmatched_before, unmatched_after).
    matches: list of (before_idx, after_idx, spatial_dist)."""
    n_before, n_after = len(before_objs), len(after_objs)
    if n_before == 0 or n_after == 0:
        return [], list(range(n_before)), list(range(n_after))

    before_centers = np.stack([bbox_center(o) for o in before_objs])
    after_centers = np.stack([bbox_center(o) for o in after_objs])
    spatial_cost = np.linalg.norm(before_centers[:, None, :] - after_centers[None, :, :], axis=-1)

    visual_cost = np.zeros_like(spatial_cost)
    for i, bo in enumerate(before_objs):
        for j, ao in enumerate(after_objs):
            visual_cost[i, j] = cosine_distance(bo["clip_ft"], ao["clip_ft"])

    combined_cost = spatial_cost + visual_weight * visual_cost
    infeasible = spatial_cost > max_match_distance
    # Large-but-finite sentinel so linear_sum_assignment stays well-defined even if a
    # row/column has no feasible match at all.
    combined_cost = np.where(infeasible, 1e6, combined_cost)

    row_idx, col_idx = linear_sum_assignment(combined_cost)

    matches = []
    matched_before, matched_after = set(), set()
    for i, j in zip(row_idx, col_idx):
        if infeasible[i, j]:
            continue
        matches.append((int(i), int(j), float(spatial_cost[i, j])))
        matched_before.add(int(i))
        matched_after.add(int(j))

    unmatched_before = [i for i in range(n_before) if i not in matched_before]
    unmatched_after = [j for j in range(n_after) if j not in matched_after]
    return matches, unmatched_before, unmatched_after


def encode_masks(obj):
    """Returns {frame_idx: {'mask': rle, 'cost': float}} and the (H, W) of the masks."""
    per_frame = {}
    hw = None
    for frame_idx, mask in zip(obj["image_idx"], obj["mask"]):
        mask = np.asfortranarray(np.asarray(mask, dtype=np.uint8))
        hw = mask.shape
        rle = mask_utils.encode(mask)
        rle["counts"] = rle["counts"].decode("ascii")
        per_frame[int(frame_idx)] = {"mask": rle, "cost": 1.0}
    return per_frame, hw


def infer_hw(before_objs, after_objs):
    """Fallback (H, W) source for when nothing was exported into object_masks (e.g. every
    matched pair was under moved_threshold, so build_object_masks() never called
    encode_masks() and its hw stayed None) -- read it directly off any object's mask
    instead, since before_objs/after_objs are non-empty and still have real masks even
    though none of them counted as a "change"."""
    for obj in before_objs + after_objs:
        for mask in obj.get("mask", []):
            return np.asarray(mask).shape
    return None


def build_object_masks(before_objs, after_objs, matches, unmatched_before, unmatched_after, moved_threshold: float):
    object_masks = {}
    hw = None
    next_id = 0

    def add_entry(video_1_obj=None, video_2_obj=None, match_cost=None):
        nonlocal hw, next_id
        entry = {}
        if video_1_obj is not None:
            frames, this_hw = encode_masks(video_1_obj)
            if match_cost is not None:
                for v in frames.values():
                    v["cost"] = 1.0 / (1.0 + match_cost)
            entry["video_1"] = frames
            hw = hw or this_hw
        if video_2_obj is not None:
            frames, this_hw = encode_masks(video_2_obj)
            if match_cost is not None:
                for v in frames.values():
                    v["cost"] = 1.0 / (1.0 + match_cost)
            entry["video_2"] = frames
            hw = hw or this_hw
        object_masks[next_id] = entry
        next_id += 1

    for before_idx, after_idx, spatial_dist in matches:
        if spatial_dist <= moved_threshold:
            continue  # matched and didn't move -- not a change, don't export
        add_entry(before_objs[before_idx], after_objs[after_idx], match_cost=spatial_dist)

    for i in unmatched_before:
        add_entry(video_1_obj=before_objs[i])

    for j in unmatched_after:
        add_entry(video_2_obj=after_objs[j])

    return object_masks, hw


def save_moved_objects_pointcloud_debug_image(pair_name, before_objs, after_objs, moved_pairs, out_path: Path,
                                               up_axis=None, up_direction=None):
    """2D 투영 디버그 이미지: 전체 3D 점군을 회색 배경으로 깔고, moved_pairs로 판정된
    물체들의 이동 전(주황)/이동 후(파랑) 점과 중심 간 화살표를 표시한다.

    up_axis/up_direction은 메인 파이프라인의 detect_up_vector()(slam/utils.py, 카메라
    위치 기반) 결과를 pcd 파일에서 그대로 읽어온 것 -- 이 투영 평면(up-axis와 직교하는
    나머지 두 축)을 정하는 데 쓴다. up-axis 자체는 이 평면과 수직이라 실제 좌표 공간
    안에는 화살표를 그릴 수 없으므로, 방향은 좌표와 무관한 구석의 나침반 아이콘+텍스트로
    별도 표시한다. 예전 pcd 파일처럼 up_axis가 없으면(None) 점군 bbox extent가 가장
    작은 축으로 폴백(이 경우 방향은 알 수 없음)."""
    if not moved_pairs:
        print(f"[{pair_name}] no moved objects -- skipping pointcloud debug image")
        return

    if up_axis is None:
        all_points = np.concatenate(
            [o["pcd_np"] for o in before_objs + after_objs if o["pcd_np"].size > 0], axis=0
        )
        extent = all_points.max(axis=0) - all_points.min(axis=0)
        up_axis = int(np.argmin(extent))
    plane_axes = [i for i in range(3) if i != up_axis]

    moved_before_idx = {bi for bi, _, _ in moved_pairs}
    moved_after_idx = {ai for _, ai, _ in moved_pairs}

    fig, ax = plt.subplots(figsize=(10, 10))

    context_points = [
        o["pcd_np"][:, plane_axes] for i, o in enumerate(before_objs) if i not in moved_before_idx
    ] + [
        o["pcd_np"][:, plane_axes] for i, o in enumerate(after_objs) if i not in moved_after_idx
    ]
    if context_points:
        ctx = np.concatenate(context_points, axis=0)
        ax.scatter(ctx[:, 0], ctx[:, 1], s=2, c="lightgray", alpha=0.3, label="context (unchanged)")

    for before_idx, after_idx, dist in moved_pairs:
        before_pts = before_objs[before_idx]["pcd_np"][:, plane_axes]
        after_pts = after_objs[after_idx]["pcd_np"][:, plane_axes]
        ax.scatter(before_pts[:, 0], before_pts[:, 1], s=6, c="orange", alpha=0.7)
        ax.scatter(after_pts[:, 0], after_pts[:, 1], s=6, c="blue", alpha=0.7)
        before_c, after_c = before_pts.mean(axis=0), after_pts.mean(axis=0)
        ax.annotate("", xy=after_c, xytext=before_c,
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.text(after_c[0], after_c[1], f" #{before_idx}->{after_idx} ({dist:.2f}m)", fontsize=8)

    ax.scatter([], [], s=20, c="orange", label="moved: before")
    ax.scatter([], [], s=20, c="blue", label="moved: after")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(AXIS_NAMES[plane_axes[0]])
    ax.set_ylabel(AXIS_NAMES[plane_axes[1]])
    ax.legend(loc="upper right", fontsize=8)

    # up-axis is perpendicular to this top-down plane, so its direction can't be
    # drawn as a real arrow in data space -- show a small fixed-direction compass
    # icon (axes-fraction coords, unrelated to the actual point coordinates) in the
    # opposite corner from the legend instead, naming the real 3D axis + sign.
    if up_direction is not None:
        axis_label = f"{'+' if up_direction > 0 else '-'}{AXIS_NAMES[up_axis]}"
        ax.annotate("", xy=(0.06, 0.95), xytext=(0.06, 0.87), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))
        ax.text(0.06, 0.965, f"UP ({axis_label})", transform=ax.transAxes,
                ha="center", fontsize=9, fontweight="bold")
        ax.set_title(f"{pair_name} -- moved objects (up-axis: {axis_label})")
    else:
        ax.text(0.06, 0.95, f"up-axis: {AXIS_NAMES[up_axis]}\n(direction unknown)", transform=ax.transAxes,
                ha="center", fontsize=8, style="italic")
        ax.set_title(f"{pair_name} -- moved objects (up-axis auto-detected: {AXIS_NAMES[up_axis]}, direction unknown)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{pair_name}] moved-objects pointcloud debug image -> {out_path}")


def convert(pair_name: str, concept_graphs_dir: Path, benchmark_data_dir: Path, exp_suffix: str,
            max_match_distance: float, moved_threshold: float, visual_weight: float):
    before_objs, before_up_axis, before_up_direction = load_objects(concept_graphs_dir, "before", exp_suffix)
    after_objs, after_up_axis, after_up_direction = load_objects(concept_graphs_dir, "after", exp_suffix)
    # before/after share one Pi3-estimated coordinate frame, so these should agree --
    # prefer "before"'s, falling back to "after"'s for pcd files saved before either
    # variant persisted this (see save_pointcloud()'s up_axis/up_direction params).
    up_axis = before_up_axis if before_up_axis is not None else after_up_axis
    up_direction = before_up_direction if before_up_axis is not None else after_up_direction

    matches, unmatched_before, unmatched_after = match_objects(
        before_objs, after_objs, max_match_distance, visual_weight
    )
    object_masks, hw = build_object_masks(
        before_objs, after_objs, matches, unmatched_before, unmatched_after, moved_threshold
    )
    if hw is None:
        # No moved/removed/added objects, but before_objs/after_objs may still be
        # non-empty (e.g. everything matched and stayed put) -- get (H, W) straight from
        # any object's mask so we can still write a valid (empty) object_masks.pkl
        # instead of aborting the whole scene pair.
        hw = infer_hw(before_objs, after_objs)
    if hw is None:
        # before_objs/after_objs are both completely empty (no non-background object
        # survived mapping in either variant) -- infer_hw() has no mask to read either.
        # Fall back to the scene's own dataset config so the pair still produces a valid
        # (fully empty) object_masks.pkl instead of aborting.
        hw = load_scene_hw(pair_name)

    if hw is None:
        raise RuntimeError(
            f"No changed objects with any observed frames found for '{pair_name}' -- "
            "nothing to export."
        )
    H, W = hw
    result = {"H": int(H), "W": int(W), **object_masks}

    out_path = benchmark_data_dir / "object_masks.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    moved_pairs = [(bi, ai, d) for bi, ai, d in matches if d > moved_threshold]
    print(
        f"[{pair_name}] before={len(before_objs)} after={len(after_objs)} "
        f"moved={len(moved_pairs)} removed={len(unmatched_before)} added={len(unmatched_after)} "
        f"-> {out_path}"
    )

    save_moved_objects_pointcloud_debug_image(
        pair_name, before_objs, after_objs, moved_pairs,
        benchmark_data_dir / "moved_objects_pointcloud.png",
        up_axis=up_axis, up_direction=up_direction,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair_name", required=True)
    parser.add_argument("--max_match_distance", type=float, default=1.5,
                         help="Max 3D centroid distance (meters) to even consider two objects the same (else always removed+added)")
    parser.add_argument("--moved_threshold", type=float, default=0.3,
                         help="3D centroid distance (meters) above which a matched pair counts as 'moved'")
    parser.add_argument("--visual_weight", type=float, default=1.0,
                         help="Weight of CLIP cosine-distance term relative to spatial distance (meters) in the matching cost")
    args = parser.parse_args()

    cfg = _load_rerun_mapping_config()
    scene_root = cfg["output_root"] / args.pair_name

    convert(
        pair_name=args.pair_name,
        concept_graphs_dir=scene_root / "concept_graphs",
        benchmark_data_dir=scene_root / "benchmark_data",
        exp_suffix=cfg["exp_suffix"],
        max_match_distance=args.max_match_distance,
        moved_threshold=args.moved_threshold,
        visual_weight=args.visual_weight,
    )


if __name__ == "__main__":
    main()
