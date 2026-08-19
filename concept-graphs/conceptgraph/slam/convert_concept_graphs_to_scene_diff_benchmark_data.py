"""
Step 2: match a pair's "before" and "after" ConceptGraphs against each other and
export the result as a SceneDiff-benchmark-compatible object_masks.pkl, so it can be
scored by scene_diff/scripts/evaluate_multiview.py (run_scene_diff_benchmark.py, step 3).

Matching: before/after ConceptGraphs share one Pi3-estimated coordinate frame (see
conceptgraph/hydra_configs/rerun_realtime_mapping.yaml), so their 3D geometry is
directly comparable. Two stages:

  1. Geometry. Among pairs whose AABBs overlap, trimmed_surface_distance (pooled
     two-way nearest-neighbour distances, worst tail trimmed) at or under
     scenediff_geometric_match_max_distance matches them 1:1 outright, skipping
     appearance entirely. This is the "it simply did not move" case, and settling it
     here keeps it off the semantic threshold -- it also works where appearance cannot,
     since an open-vocabulary detector will label one physical object "coffee table" in
     one scan and "remote" in the other.
  2. Appearance, for whatever stage 1 left. The score is the mean cosine similarity
     over EVERY (before frame, after frame) pair, computed as one dot product of the
     objects' mean unit features (see semantic_similarity). Many-to-many: an object may
     match several on the other side, which a node split in one scan requires.

Classification is then per side, and only from objects marked recognition_trusted --
an object whose recognition evidence was too thin cannot assert a change, though every
object regardless of trust is available to match against:

    trusted before object, no match -> removed
    trusted after  object, no match -> added
    matched, centroid moved > scenediff_moved_threshold -> moved
    matched and still in place -> unchanged, not a change, not exported

Because matching is many-to-many and each side judges for itself, one physical move can
surface as several moved pairs. They are grouped into connected components of the match
graph and exported as ONE change per component, with the masks of every node on a side
unioned per frame (see _group_moved_pairs).

A trusted object the other scan's camera never had in view is left unchanged rather
than called removed/added -- the trajectories differ, and "never looked there" is not
evidence of change. This applies symmetrically to both directions.

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
projection with every object colored by its classification -- unchanged (gray), not
visible on the other side (light gray), moved (blue, with an arrow to its new
position), removed (red), added (green). Always written, even when nothing changed --
see save_change_pointcloud_debug_image().
"""
import argparse
import gzip
import pickle
from pathlib import Path

import hydra
from conceptgraph.utils.general_utils import EXP_SUFFIX
from conceptgraph.slam.geometric_fusion import (
    _bbox_gate_vector,
    aabb_from_points,
    count_visible_frames,
    geometry_fusion_params_from_cfg,
    trimmed_surface_distance,
)
from conceptgraph.dataset.datasets_common import get_dataset
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from pycocotools import mask as mask_utils

AXIS_NAMES = ["X", "Y", "Z"]


def _load_rerun_mapping_config(output_root_override: str | None = None) -> dict:
    """Reads output_root from rerun_realtime_mapping.yaml -- the single source of
    truth for where pipeline outputs live -- via hydra.compose(), the same
    hydra_configs/ this script's sibling rerun_realtime_mapping.py loads with
    @hydra.main(config_path="../hydra_configs/", config_name="rerun_realtime_mapping").
    hydra.initialize() resolves config_path relative to this file, so it's the same
    "../hydra_configs" used there. exp_suffix itself is not read from the yaml --
    it's pinned to general_utils.EXP_SUFFIX (see there for why). output_root_override
    mirrors passing output_root=... on rerun_realtime_mapping.py's CLI, so this stage
    agrees with stage 1 on where a given run's outputs live even when that run didn't
    use the yaml's default output_root."""
    overrides = [f"output_root={output_root_override}"] if output_root_override else []
    with hydra.initialize(version_base=None, config_path="../hydra_configs"):
        cfg = hydra.compose(config_name="rerun_realtime_mapping", overrides=overrides)
    return {"output_root": Path(cfg.output_root).resolve(), "cfg": cfg}


def load_variant_dataset(cfg, pair_name: str, variant: str):
    """
    The other scan's camera trajectory and depth maps, which the visibility exception
    needs: to know whether "before" ever showed an object, this scan has to be replayed
    with the OTHER variant's poses. Same loader and same dataconfig the mapping stage
    used, so the frame indexing lines up.
    """
    dataconfig = (Path(__file__).resolve().parent.parent / "dataset" / "dataconfigs"
                  / "scenediff" / pair_name / f"{variant}.yaml")
    # cfg.image_height/width are null in the yaml -- the mapping stage fills them in via
    # process_cfg() from this same dataconfig, which never runs here, so read them the
    # way load_scene_hw() does. They must match what mapping used, or the projected
    # visibility test would be measured against a differently-scaled camera.
    height, width = cfg.image_height, cfg.image_width
    if height is None or width is None:
        height, width = load_scene_hw(pair_name)
    return get_dataset(
        dataconfig=dataconfig,
        start=cfg.start, end=cfg.end, stride=cfg.get("stride", 1),
        basedir=cfg.dataset_root, sequence=f"{pair_name}/{variant}",
        desired_height=height, desired_width=width,
        device="cpu", dtype=torch.float,
    )


def count_cross_visibility(objects, cfg, pair_name: str, other_variant: str, params):
    """Per object, how many frames of the OTHER scan should have shown it."""
    if not objects:
        return []
    dataset = load_variant_dataset(cfg, pair_name, other_variant)
    n_visible, _frames = count_visible_frames(
        [np.asarray(o["pcd_np"]) for o in objects], dataset, params,
        desc=f"visibility in {other_variant}")
    return n_visible


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
    # in save_change_pointcloud_debug_image()).
    return objects, data.get("up_axis"), data.get("up_direction")


def bbox_center(obj):
    return np.asarray(obj["bbox_np"], dtype=np.float64).mean(axis=0)


def semantic_similarity(obj_a, obj_b, clip_weight: float, dino_weight: float) -> float:
    """
    How alike two objects look, averaged over EVERY (frame of a, frame of b) pair.

    Each term is literally that average. Per-frame CLIP/DINO features arrive
    L2-normalized, and clip_ft_mean/dino_ft_mean (slam/utils.py) hold their plain
    arithmetic mean without renormalizing, so for unit vectors

        mean over all N*M frame pairs of cos(a_i, b_j) == dot(mean(a), mean(b))

    exactly -- no per-frame features have to be kept or iterated. Note this is NOT
    the same as the cosine between clip_ft/dino_ft: those get renormalized on every
    merge, which discards each object's cross-frame consistency, and that magnitude
    is exactly what makes the identity hold.
    """
    return (clip_weight * float(np.dot(_mean_feature(obj_a, "clip_ft_mean"),
                                       _mean_feature(obj_b, "clip_ft_mean")))
            + dino_weight * float(np.dot(_mean_feature(obj_a, "dino_ft_mean"),
                                         _mean_feature(obj_b, "dino_ft_mean"))))


def _mean_feature(obj, key):
    feat = obj.get(key)
    if feat is None:
        raise KeyError(
            f"Object is missing '{key}'. This pcd was produced before the mean-feature "
            "fields existed -- re-run the mapping stage for this pair. Falling back to "
            "'clip_ft'/'dino_ft' would silently score a different quantity (those are "
            "renormalized on every merge), so the comparison refuses to guess."
        )
    return np.asarray(feat, dtype=np.float64).reshape(-1)


def match_geometrically(before_objs, after_objs, max_distance: float, keep_fraction: float,
                        bbox_margin: float):
    """
    Settle the "this object simply did not move" pairs before appearance is consulted
    at all, and match them 1:1.

    Only pairs whose AABBs overlap are even scored, then trimmed_surface_distance
    decides. Greedy over the qualifying pairs, closest first, so each object is claimed
    once. Returns a list of (before_idx, after_idx, distance).

    Doing this first is what keeps the unambiguous cases off the semantic threshold's
    shoulders -- and it works where appearance cannot, since an open-vocabulary detector
    happily labels the same physical object "coffee table" in one scan and "remote" in
    the other. Measured on living_room_17: 5 pairs matched here at 0.3-1.2cm, three of
    them across disagreeing labels, against a next-nearest non-match at 4.2cm.
    """
    if not before_objs or not after_objs:
        return []

    los_a, his_a = zip(*(aabb_from_points(o["pcd_np"]) for o in after_objs))
    los_a, his_a = np.stack(los_a), np.stack(his_a)

    qualifying = []
    for i, bo in enumerate(before_objs):
        lo_b, hi_b = aabb_from_points(bo["pcd_np"])
        for j in np.nonzero(_bbox_gate_vector(lo_b, hi_b, los_a, his_a, bbox_margin))[0]:
            distance = trimmed_surface_distance(
                bo["pcd_np"], after_objs[int(j)]["pcd_np"], keep_fraction)
            if distance <= max_distance:
                qualifying.append((distance, i, int(j)))

    matches, used_before, used_after = [], set(), set()
    for distance, i, j in sorted(qualifying):
        if i in used_before or j in used_after:
            continue
        used_before.add(i)
        used_after.add(j)
        matches.append((i, j, float(distance)))
    return matches


def match_semantically(before_objs, after_objs, skip_before, skip_after,
                       max_match_distance: float, sim_threshold: float,
                       clip_weight: float, dino_weight: float):
    """
    Appearance matching over whatever the geometric stage left unclaimed, many-to-many:
    one object may match several on the other side (a node split in one scan, one object
    genuinely resembling two). Returns a list of (before_idx, after_idx, similarity).

    Candidates are visited nearest-centroid-first and the scan for a given object stops
    at the first one past max_match_distance, since everything after it is further still.
    """
    matches = []
    if not before_objs or not after_objs:
        return matches

    after_live = [(j, ao) for j, ao in enumerate(after_objs) if j not in skip_after]
    for i, bo in enumerate(before_objs):
        if i in skip_before:
            continue
        center_b = bbox_center(bo)
        by_distance = sorted(
            after_live, key=lambda ja: float(np.linalg.norm(center_b - bbox_center(ja[1]))))
        for j, ao in by_distance:
            if float(np.linalg.norm(center_b - bbox_center(ao))) > max_match_distance:
                break
            similarity = semantic_similarity(bo, ao, clip_weight, dino_weight)
            if similarity >= sim_threshold:
                matches.append((i, j, float(similarity)))
    return matches


def encode_masks(*objs):
    """
    Returns {frame_idx: {'mask': rle, 'cost': float}} and the (H, W) of the masks.

    Several objects may be passed, in which case their masks are unioned per frame --
    that is how a move involving more than one node on a side is exported as one
    change (see _group_moved_pairs). Frames only some of them were seen in still come
    through, carrying just the objects that were there.
    """
    merged = {}
    hw = None
    for obj in objs:
        for frame_idx, mask in zip(obj["image_idx"], obj["mask"]):
            mask = np.asarray(mask, dtype=bool)
            hw = mask.shape
            frame_idx = int(frame_idx)
            merged[frame_idx] = mask if frame_idx not in merged else (merged[frame_idx] | mask)

    per_frame = {}
    for frame_idx, mask in merged.items():
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        per_frame[frame_idx] = {"mask": rle, "cost": 1.0}
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


def classify_changes(before_objs, after_objs, matches, moved_threshold: float,
                     before_visible_in_after, after_visible_in_before):
    """
    Turn the match set into (moved_groups, removed_idx, added_idx).

    Each side is judged independently, and only from its own TRUSTED objects -- an
    object whose recognition evidence was too thin to trust cannot assert that
    something was removed or added, but is still a perfectly good thing for the other
    side to match against, which is why matching used every object regardless.

        trusted before object, no match  -> removed
        trusted after  object, no match  -> added
        matched (either side)            -> unchanged, or moved if it travelled

    Matches are many-to-many, so an object with several counterparts counts as
    unchanged if ANY of them is within moved_threshold: something is still sitting
    where it was, whatever else also matched.

    A trusted object the other scan's camera never had in view is left alone
    (unchanged) rather than called removed/added. The two trajectories differ, so
    "never looked there" is not evidence of change -- and this cuts both ways, which is
    why the caller supplies visibility counts for both directions.
    """
    before_matches, after_matches = {}, {}
    for i, j, _score in matches:
        before_matches.setdefault(i, []).append(j)
        after_matches.setdefault(j, []).append(i)

    centers_b = [bbox_center(o) for o in before_objs]
    centers_a = [bbox_center(o) for o in after_objs]

    def travelled(i, j):
        return float(np.linalg.norm(centers_b[i] - centers_a[j]))

    def is_trusted(obj):
        return bool(obj.get("recognition_trusted", True))

    moved_pairs, removed_idx, added_idx = {}, [], []

    for i, bo in enumerate(before_objs):
        if not is_trusted(bo):
            continue
        counterparts = before_matches.get(i, [])
        if not counterparts:
            if before_visible_in_after[i] > 0:
                removed_idx.append(i)
            continue
        if any(travelled(i, j) <= moved_threshold for j in counterparts):
            continue  # still where it was
        j = min(counterparts, key=lambda j: travelled(i, j))
        moved_pairs[(i, j)] = travelled(i, j)

    for j, ao in enumerate(after_objs):
        if not is_trusted(ao):
            continue
        counterparts = after_matches.get(j, [])
        if not counterparts:
            if after_visible_in_before[j] > 0:
                added_idx.append(j)
            continue
        if any(travelled(i, j) <= moved_threshold for i in counterparts):
            continue
        i = min(counterparts, key=lambda i: travelled(i, j))
        # Keyed by the pair, so a move both sides noticed is recorded once.
        moved_pairs[(i, j)] = travelled(i, j)

    return _group_moved_pairs(moved_pairs, centers_b, centers_a), removed_idx, added_idx


def _group_moved_pairs(moved_pairs, centers_b, centers_a):
    """
    Collapse the moved pairs into connected components of the bipartite match graph, so
    one physical move is one exported change no matter how many nodes it involves.

    Matching is many-to-many, and both sides judge independently, so a single move can
    surface as several pairs: one before object matching two after objects yields a
    "moved" verdict from each after object, and an object split across scans yields one
    per fragment. Exporting those separately would report one change several times and
    hand the same mask to several entries. Everything reachable through shared endpoints
    is therefore merged into a single group whose point clouds (and, downstream, whose
    per-frame masks) are unioned.

    Returns a list of (before_indices, after_indices, distance), where distance is
    between the two merged centroids -- how far the group as a whole travelled.
    """
    if not moved_pairs:
        return []

    neighbours = {}
    for i, j in moved_pairs:
        neighbours.setdefault(("b", i), set()).add(("a", j))
        neighbours.setdefault(("a", j), set()).add(("b", i))

    groups, seen = [], set()
    for start in neighbours:
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in neighbours[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        before_idx = sorted(idx for side, idx in component if side == "b")
        after_idx = sorted(idx for side, idx in component if side == "a")
        center_b = np.mean([centers_b[i] for i in before_idx], axis=0)
        center_a = np.mean([centers_a[j] for j in after_idx], axis=0)
        groups.append((before_idx, after_idx, float(np.linalg.norm(center_b - center_a))))

    return sorted(groups, key=lambda g: (g[0], g[1]))


def build_object_masks(before_objs, after_objs, moved_groups, removed_idx, added_idx):
    object_masks = {}
    hw = None
    next_id = 0

    def add_entry(video_1_objs=(), video_2_objs=(), match_cost=None, confidence=None):
        nonlocal hw, next_id
        entry = {}
        for key, objs in (("video_1", video_1_objs), ("video_2", video_2_objs)):
            if not objs:
                continue
            frames, this_hw = encode_masks(*objs)
            if match_cost is not None:
                for v in frames.values():
                    v["cost"] = 1.0 / (1.0 + match_cost)
            elif confidence is not None:
                for v in frames.values():
                    v["cost"] = confidence
            entry[key] = frames
            hw = hw or this_hw
        object_masks[next_id] = entry
        next_id += 1

    for before_idx, after_idx, spatial_dist in moved_groups:
        add_entry([before_objs[i] for i in before_idx],
                  [after_objs[j] for j in after_idx],
                  match_cost=spatial_dist)

    for i in removed_idx:
        add_entry(video_1_objs=[before_objs[i]],
                  confidence=before_objs[i].get("recognition_confidence", 1.0))

    for j in added_idx:
        add_entry(video_2_objs=[after_objs[j]],
                  confidence=after_objs[j].get("recognition_confidence", 1.0))

    return object_masks, hw


def _categorize_side(n_objs, moved_idx, changed_idx, visible_in_other):
    """
    Partitions one side's object indices into exactly one of four buckets, in the same
    priority order classify_changes itself resolves in:

        1. part of a moved group          -> "moved"
        2. in removed_idx / added_idx      -> "changed" (removed on the before side,
                                               added on the after side)
        3. other camera should never have seen it (visible_in_other[i] == 0)
                                            -> "not_visible"
        4. everything else                 -> "unchanged"

    An untrusted, unmatched object that visibility would otherwise allow to be
    removed/added falls to "unchanged" here -- trust is a separate axis from these four
    buckets, and lumping it in with "confirmed unchanged" is the conservative choice
    rather than inventing a fifth category this request didn't ask for.
    """
    buckets = {"moved": set(), "changed": set(), "not_visible": set(), "unchanged": set()}
    for i in range(n_objs):
        if i in moved_idx:
            buckets["moved"].add(i)
        elif i in changed_idx:
            buckets["changed"].add(i)
        elif visible_in_other[i] == 0:
            buckets["not_visible"].add(i)
        else:
            buckets["unchanged"].add(i)
    return buckets


def save_change_pointcloud_debug_image(pair_name, before_objs, after_objs, moved_groups,
                                       removed_idx, added_idx, before_visible_in_after,
                                       after_visible_in_before, out_path: Path,
                                       up_axis=None, up_direction=None):
    """2D 투영 디버그 이미지: 모든 물체를 5개 카테고리로 분류해 색으로 구분한다--
    unchanged(회색), not visible on the other side(연한 회색), moved(파랑, 화살표),
    removed(빨강), added(초록). classify_changes와 정확히 같은 판정 순서를 쓴다
    (_categorize_side 참고), 그래서 이 그림이 실제로 export된 변화와 항상 일치한다.

    변화가 전혀 없어도(moved/removed/added 모두 0건) 언제나 파일을 생성한다 --
    unchanged/not-visible 두 색만으로라도. before_objs와 after_objs가 둘 다 완전히
    비어 그릴 점 자체가 없을 때만 건너뛴다.

    up_axis/up_direction은 메인 파이프라인의 detect_up_vector()(slam/utils.py, 카메라
    위치 기반) 결과를 pcd 파일에서 그대로 읽어온 것 -- 이 투영 평면(up-axis와 직교하는
    나머지 두 축)을 정하는 데 쓴다. up-axis 자체는 이 평면과 수직이라 실제 좌표 공간
    안에는 화살표를 그릴 수 없으므로, 방향은 좌표와 무관한 구석의 나침반 아이콘+텍스트로
    별도 표시한다. 예전 pcd 파일처럼 up_axis가 없으면(None) 점군 bbox extent가 가장
    작은 축으로 폴백(이 경우 방향은 알 수 없음)."""
    if not before_objs and not after_objs:
        print(f"[{pair_name}] no objects on either side -- skipping pointcloud debug image")
        return

    if up_axis is None:
        all_points = np.concatenate(
            [o["pcd_np"] for o in before_objs + after_objs if o["pcd_np"].size > 0], axis=0
        )
        extent = all_points.max(axis=0) - all_points.min(axis=0)
        up_axis = int(np.argmin(extent))
    plane_axes = [i for i in range(3) if i != up_axis]

    moved_before_idx = {i for before_idx, _, _ in moved_groups for i in before_idx}
    moved_after_idx = {j for _, after_idx, _ in moved_groups for j in after_idx}
    before_buckets = _categorize_side(len(before_objs), moved_before_idx, set(removed_idx),
                                      before_visible_in_after)
    after_buckets = _categorize_side(len(after_objs), moved_after_idx, set(added_idx),
                                     after_visible_in_before)

    fig, ax = plt.subplots(figsize=(10, 10))

    def scatter_bucket(key, objs, color, label, **kw):
        idx = before_buckets[key] if objs is before_objs else after_buckets[key]
        if not idx:
            return
        pts = np.concatenate([objs[i]["pcd_np"] for i in sorted(idx)], axis=0)[:, plane_axes]
        ax.scatter(pts[:, 0], pts[:, 1], c=color, label=label, **kw)

    # Background categories first, so a change never ends up visually buried under them.
    for objs in (before_objs, after_objs):
        scatter_bucket("unchanged", objs, "gray", None, s=2, alpha=0.3)
        scatter_bucket("not_visible", objs, "lightgray", None, s=2, alpha=0.3)
    ax.scatter([], [], s=20, c="gray", label="unchanged")
    ax.scatter([], [], s=20, c="lightgray", label="not visible on other side")

    scatter_bucket("changed", before_objs, "red", "removed", s=8, alpha=0.8)
    scatter_bucket("changed", after_objs, "green", "added", s=8, alpha=0.8)

    for before_idx, after_idx, dist in moved_groups:
        # A group can hold several nodes per side (see _group_moved_pairs); they are one
        # change, so their points are pooled and one arrow is drawn between the merged
        # centroids. Both sides are blue -- the arrow, not the color, carries direction.
        before_pts = np.concatenate([before_objs[i]["pcd_np"] for i in before_idx])[:, plane_axes]
        after_pts = np.concatenate([after_objs[j]["pcd_np"] for j in after_idx])[:, plane_axes]
        ax.scatter(before_pts[:, 0], before_pts[:, 1], s=8, c="blue", alpha=0.8)
        ax.scatter(after_pts[:, 0], after_pts[:, 1], s=8, c="blue", alpha=0.8)
        before_c, after_c = before_pts.mean(axis=0), after_pts.mean(axis=0)
        ax.annotate("", xy=after_c, xytext=before_c,
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.text(after_c[0], after_c[1], f" #{before_idx}->{after_idx} ({dist:.2f}m)", fontsize=8)
    if moved_groups:
        ax.scatter([], [], s=20, c="blue", label="moved")

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
        ax.set_title(f"{pair_name} -- changes (up-axis: {axis_label})")
    else:
        ax.text(0.06, 0.95, f"up-axis: {AXIS_NAMES[up_axis]}\n(direction unknown)", transform=ax.transAxes,
                ha="center", fontsize=8, style="italic")
        ax.set_title(f"{pair_name} -- changes (up-axis auto-detected: {AXIS_NAMES[up_axis]}, direction unknown)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{pair_name}] change pointcloud debug image -> {out_path}")


def convert(pair_name: str, concept_graphs_dir: Path, benchmark_data_dir: Path, exp_suffix: str,
            cfg, max_match_distance: float, moved_threshold: float, sim_threshold: float):
    before_objs, before_up_axis, before_up_direction = load_objects(concept_graphs_dir, "before", exp_suffix)
    after_objs, after_up_axis, after_up_direction = load_objects(concept_graphs_dir, "after", exp_suffix)
    # before/after share one Pi3-estimated coordinate frame, so these should agree --
    # prefer "before"'s, falling back to "after"'s for pcd files saved before either
    # variant persisted this (see save_pointcloud()'s up_axis/up_direction params).
    up_axis = before_up_axis if before_up_axis is not None else after_up_axis
    up_direction = before_up_direction if before_up_axis is not None else after_up_direction

    # Fail here rather than midway through matching: a pcd written before the
    # mean-feature fields existed cannot be compared, and silently substituting
    # clip_ft/dino_ft would score a different quantity (see semantic_similarity).
    for variant, objs in (("before", before_objs), ("after", after_objs)):
        missing = [o.get("curr_obj_num") for o in objs if o.get("clip_ft_mean") is None]
        if missing:
            raise RuntimeError(
                f"[{pair_name}] {len(missing)}/{len(objs)} objects in '{variant}' have no "
                f"clip_ft_mean -- this pcd predates that field. Re-run the mapping stage "
                f"for this pair before comparing."
            )

    # Stage 1: pairs that simply did not move, settled on geometry alone.
    geometric = match_geometrically(
        before_objs, after_objs,
        max_distance=float(cfg.scenediff_geometric_match_max_distance),
        keep_fraction=float(cfg.scenediff_geometric_match_keep_fraction),
        bbox_margin=float(cfg.fusion_bbox_margin),
    )
    claimed_before = {i for i, _, _ in geometric}
    claimed_after = {j for _, j, _ in geometric}

    # Stage 2: appearance, over whatever stage 1 left, many-to-many.
    semantic = match_semantically(
        before_objs, after_objs, claimed_before, claimed_after,
        max_match_distance=max_match_distance,
        sim_threshold=sim_threshold,
        clip_weight=float(cfg.scenediff_clip_weight),
        dino_weight=float(cfg.scenediff_dino_weight),
    )
    matches = geometric + semantic

    # A trusted object the other camera never had in view is not evidence of a change.
    params = geometry_fusion_params_from_cfg(cfg)
    before_visible_in_after = count_cross_visibility(before_objs, cfg, pair_name, "after", params)
    after_visible_in_before = count_cross_visibility(after_objs, cfg, pair_name, "before", params)

    moved_groups, removed_idx, added_idx = classify_changes(
        before_objs, after_objs, matches, moved_threshold,
        before_visible_in_after, after_visible_in_before,
    )
    object_masks, hw = build_object_masks(
        before_objs, after_objs, moved_groups, removed_idx, added_idx
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

    n_trusted_b = sum(bool(o.get("recognition_trusted", True)) for o in before_objs)
    n_trusted_a = sum(bool(o.get("recognition_trusted", True)) for o in after_objs)
    print(
        f"[{pair_name}] before={len(before_objs)} (trusted {n_trusted_b}) "
        f"after={len(after_objs)} (trusted {n_trusted_a}) | "
        f"matched geometric={len(geometric)} semantic={len(semantic)} | "
        f"moved={len(moved_groups)} removed={len(removed_idx)} added={len(added_idx)} "
        f"-> {out_path}"
    )

    save_change_pointcloud_debug_image(
        pair_name, before_objs, after_objs, moved_groups, removed_idx, added_idx,
        before_visible_in_after, after_visible_in_before,
        benchmark_data_dir / "moved_objects_pointcloud.png",
        up_axis=up_axis, up_direction=up_direction,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair_name", required=True)
    # These default to None so the yaml stays the single source of truth; pass one only
    # to override it for a single run.
    parser.add_argument("--max_match_distance", type=float, default=None,
                         help="Max 3D centroid distance (meters) to even consider two objects the same")
    parser.add_argument("--moved_threshold", type=float, default=None,
                         help="3D centroid distance (meters) above which a matched pair counts as 'moved'")
    parser.add_argument("--sim_threshold", type=float, default=None,
                         help="Minimum semantic similarity (mean CLIP+DINO cosine over all frame pairs) to call two objects the same")
    parser.add_argument("--output_root", default=None,
                         help="Overrides the yaml's output_root, like output_root=... on rerun_realtime_mapping.py's CLI")
    args = parser.parse_args()

    loaded = _load_rerun_mapping_config(args.output_root)
    cfg = loaded["cfg"]
    scene_root = loaded["output_root"] / args.pair_name

    def pick(override, key):
        return float(cfg[key]) if override is None else override

    convert(
        pair_name=args.pair_name,
        concept_graphs_dir=scene_root / "concept_graphs",
        benchmark_data_dir=scene_root / "benchmark_data",
        exp_suffix=EXP_SUFFIX,
        cfg=cfg,
        max_match_distance=pick(args.max_match_distance, "scenediff_max_match_distance"),
        moved_threshold=pick(args.moved_threshold, "scenediff_moved_threshold"),
        sim_threshold=pick(args.sim_threshold, "scenediff_semantic_sim_threshold"),
    )


if __name__ == "__main__":
    main()
