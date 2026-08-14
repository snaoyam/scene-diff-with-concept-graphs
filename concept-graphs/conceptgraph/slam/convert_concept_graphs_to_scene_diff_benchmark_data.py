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
"""
import argparse
import gzip
import pickle
from pathlib import Path

import hydra
import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


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


def load_objects(concept_graphs_dir: Path, variant: str, exp_suffix: str):
    pcd_path = concept_graphs_dir / variant / "exps" / exp_suffix / f"pcd_{exp_suffix}.pkl.gz"
    with gzip.open(pcd_path, "rb") as f:
        data = pickle.load(f)
    objects = [obj for obj in data["objects"] if not obj.get("is_background", False)]
    return objects


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


def convert(pair_name: str, concept_graphs_dir: Path, benchmark_data_dir: Path, exp_suffix: str,
            max_match_distance: float, moved_threshold: float, visual_weight: float):
    before_objs = load_objects(concept_graphs_dir, "before", exp_suffix)
    after_objs = load_objects(concept_graphs_dir, "after", exp_suffix)

    matches, unmatched_before, unmatched_after = match_objects(
        before_objs, after_objs, max_match_distance, visual_weight
    )
    object_masks, hw = build_object_masks(
        before_objs, after_objs, matches, unmatched_before, unmatched_after, moved_threshold
    )

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

    n_moved = sum(1 for _, _, d in matches if d > moved_threshold)
    print(
        f"[{pair_name}] before={len(before_objs)} after={len(after_objs)} "
        f"moved={n_moved} removed={len(unmatched_before)} added={len(unmatched_after)} "
        f"-> {out_path}"
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
